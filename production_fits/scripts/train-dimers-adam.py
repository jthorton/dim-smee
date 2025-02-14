import contextlib
import datetime
import os
import pathlib

import datasets
import datasets.distributed
import tensorboardX
import torch
import torch.distributed

import descent.optim
import descent.targets.dimers
import descent.utils.loss
import descent.utils.reporting
import numpy as np

WORLD_SIZE = torch.multiprocessing.cpu_count()


@contextlib.contextmanager
def open_writer(path: pathlib.Path, rank: int) -> tensorboardX.SummaryWriter:
    if rank != 0:
        yield None
    else:
        with tensorboardX.SummaryWriter(str(path)) as writer:
            yield writer


def main(rank: int = 0):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)

    dataset_dir = pathlib.Path("dimer-datasets/DE-filtered-no-charge-small-mol/")
    output_dir = pathlib.Path(f"fits")
    fit_dir = output_dir.joinpath("DE-filtered-no-charge-small-mols")
    fit_dir.mkdir(exist_ok=True, parents=True)
    n_epochs = 3000
    lr = 0.001

    dataset = datasets.Dataset.load_from_disk(dataset_dir.as_posix())
    # format
    dataset = dataset.with_format(
        type="torch", columns=["coords", "energy"], dtype=torch.float64
    )
    dataset = datasets.distributed.split_dataset_by_node(
        dataset, rank=rank, world_size=WORLD_SIZE
    )

    force_field, topologies = torch.load(dataset_dir.joinpath("ff_and_tops.pt"))

    vdw_potential = force_field.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True
    # if using dexp we need to optimise alpha and beta
    vdw_potential.attributes.requires_grad = True

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"fit-{timestamp}"
    experiment_dir = fit_dir.joinpath(experiment_name)

    with open_writer(experiment_dir, rank) as writer:
        optimizer = torch.optim.Adam([vdw_potential.parameters, vdw_potential.attributes], lr=lr, amsgrad=True)

        if rank == 0:
            for v in tensorboardX.writer.hparams({"optimizer": "Adam", "lr": lr}, {}):
                writer.file_writer.add_summary(v)

        for i in range(n_epochs):

            y_ref, y_pred = descent.targets.dimers.predict(
                dataset, force_field, topologies
            )

            loss = ((y_pred - y_ref) ** 2).sum()
            loss.backward()

            torch.distributed.all_reduce(loss)
            torch.distributed.all_reduce(vdw_potential.parameters.grad)
            torch.distributed.all_reduce(vdw_potential.attributes.grad)
            # zero out the attributes we dont want to fit
            # print(vdw_potential.attributes)
            vdw_potential.attributes.grad[:5] = 0.0
            # zero out all non distance v-site parameters
            # force_field.v_sites.parameters.grad[:,1:] = 0.0

            if rank == 0:
                print(f"epoch={i} loss={loss:.6f}", flush=True)
                writer.add_scalar("loss", loss.detach().item(), i)
                writer.add_scalar(
                    "grad/vdw", torch.norm(vdw_potential.parameters.grad), i
                )
                writer.add_scalar('grad/a&b', torch.norm(vdw_potential.attributes.grad), i)

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                # force zero or positive parameter values and zero v-site parameters
                vdw_potential.parameters *= torch.where(
                    vdw_potential.parameters < 0.0, 0.0, 1.0
                )
                # constrain all vsites to be negative
                # force_field.v_sites.parameters[:, 0] = torch.where(force_field.v_sites.parameters[:, 0] > -0.01, -0.01, force_field.v_sites.parameters[:, 0])
            if i % 100 == 0 and rank == 0:
                descent.utils.reporting.print_force_field_summary(force_field)
                # save the ff with torch
                torch.save(
                    force_field, experiment_dir.joinpath(f"epoch_{i}_forcefield.pt")
                )
            if np.isnan(loss.item()) and rank == 0:
                print("loss is Nan saving parameters")
                descent.utils.reporting.print_force_field_summary(force_field)
                torch.save(
                    force_field, experiment_dir.joinpath(f"epoch_{i}_forcefield.pt")
                )
                exit(0)
    # vdw_potential.parameters = vdw_parameters / vdw_scale

    if rank != 0:
        exit(0)

    descent.utils.reporting.print_force_field_summary(force_field)
    torch.save(force_field, experiment_dir.joinpath(f"epoch_{i}_forcefield.pt"))

    force_field_initial, _ = torch.load(dataset_dir.joinpath("ff_and_tops.pt"))

    descent.targets.dimers.report(
        datasets.Dataset.load_from_disk(dataset_dir.as_posix()),
        {"DEXP Initial": force_field_initial, "DEXP Opt": force_field},
        {"DEXP Initial": topologies, "DEXP Opt": topologies},
        experiment_dir.joinpath("energies.html"),
    )


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.multiprocessing.spawn(main, nprocs=WORLD_SIZE, join=True)
