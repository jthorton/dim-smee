"""General trainer script with batched closure for dimers, energies/forces and smart batching for liquids which can easily be extended to fit to combinations
of targets.
Note uses the LM optimiser exclusively to avoid lots of liquid evaluations.

Assume we have run a dimer fit with the adam optimiser first with a lot of iterations, see the train-dimers-adam script

"""

import copy
import functools
import logging
import pathlib
from typing import Optional
import datetime

import click
import datasets
import openff.interchange
import openff.toolkit
import smee
import smee.converters
import smee.utils
import torch
import yaml
import pprint

import descent.optim
import descent.targets
import descent.targets.dimers
import descent.targets.thermo
import descent.train
import descent.utils.loss
import descent.utils.molecule
import descent.utils.reporting
import tqdm

logger = logging.getLogger(__name__)


def apply_parameters(
    force_field: str,
    energy_dataset: Optional[datasets.Dataset] = None,
    thermo_dataset: Optional[datasets.Dataset] = None,
    dimer_dataset: Optional[datasets.Dataset] = None,
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    unique_smiles = []
    if energy_dataset is not None:
        unique_smiles.extend(descent.targets.energy.extract_smiles(energy_dataset))
    if thermo_dataset is not None:
        unique_smiles.extend(descent.targets.thermo.extract_smiles(thermo_dataset))
    if dimer_dataset is not None:
        unique_smiles.extend(descent.targets.dimers.extract_smiles(dimer_dataset))
    unique_smiles = set(unique_smiles)
    interchanges = [
        openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField(force_field, load_plugins=True),
            openff.toolkit.Molecule.from_mapped_smiles(smiles).to_topology(),
        )
        for smiles in tqdm.tqdm(unique_smiles, desc="Creating interchanges", ncols=80)
    ]

    force_field, topologies = smee.converters.convert_interchange(interchanges)
    force_field = force_field.to("cuda")

    topologies = {
        smiles: topology.to("cuda")
        for smiles, topology in zip(unique_smiles, topologies)
    }

    for top in topologies.values():  # for some reason needed for hessian calc...
        for param in top.parameters.values():
            param.assignment_matrix = param.assignment_matrix.to_dense()

    return force_field, topologies


def report(
    step,
    x: torch.Tensor,
    loss,
    gradient,
    hessian,
    step_quality,
    accept_step,
    trainable: descent.train.Trainable,
    topologies: dict[str, smee.TensorTopology],
    # thermo_dataset: datasets.Dataset,
):
    # if accept_step:
    #     ff = trainable.to_force_field(x.detach().clone().requires_grad_(False))
    #
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         descent.targets.thermo.predict(
    #             thermo_dataset, ff, topologies, pathlib.Path(tmp_dir), None, None, True
    #         )

    logging.info(
        f"step: {step} "
        f"loss: {loss.detach().item():.5f} "
        f"quality: {step_quality.detach().item():.5f} "
        f"accept: {str(accept_step).lower()}"
    )
    logging.info(f"x: {x.detach().cpu().numpy()}")


def to_vdw_only_ff(ff: smee.TensorForceField) -> smee.TensorForceField:
    return smee.TensorForceField(
        potentials=[ff.potentials_by_type["vdW"]], v_sites=ff.v_sites
    )


def default_dimer_closure(
    trainable: "descent.train.Trainable",
    topologies: dict[str, smee.TensorTopology],
    dataset: datasets.Dataset,
    batch_size: int = 1,
) -> descent.optim.ClosureFn:
    """Return a default closure function for training against thermodynamic
    properties.

    Args:
        trainable: The wrapper around trainable parameters.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        dataset: The dataset to train against.
        batch_size: The number of dimer entries to calculate the gradient and hessian for in each batch, gradients and hessian will be averaged over the batch.

    Returns:
        The default closure function.
    """
    import tqdm
    import more_itertools
    import math

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ):
        total_loss, grad, hess = (
            torch.zeros(size=(1,), device=x.device.type),
            None,
            None,
        )
        # get the total number of dimers and configs to get the RMSE and average gradient and hessian
        n_dimers = len(dataset)
        total_points = sum([len(d["energy"]) for d in dataset])

        for batch_ids in tqdm.tqdm(
            more_itertools.batched([i for i in range(n_dimers)], batch_size),
            desc="Calculating dimers",
            ncols=80,
            total=math.ceil(n_dimers / batch_size),
        ):
            batch = dataset.select(indices=batch_ids)
            actuall_batch_size = len(batch)
            batch_configs = sum([len(d["energy"]) for d in batch])

            def loss_fn(_x):
                ff_vdw = to_vdw_only_ff(trainable.to_force_field(_x))
                y_ref, y_pred = descent.targets.dimers.predict(
                    batch, ff_vdw, topologies
                )
                return torch.sqrt(((y_pred - y_ref) ** 2).mean())

            loss = loss_fn(x)

            if compute_hessian:
                hessian = torch.autograd.functional.hessian(
                    loss_fn, x, vectorize=True, create_graph=False
                ).detach()
                if hess is None:
                    hess = hessian * actuall_batch_size
                else:
                    hess += hessian * actuall_batch_size
            if compute_gradient:
                (gradient,) = torch.autograd.grad(loss, x, create_graph=False)
                gradient = gradient.detach()
                if grad is None:
                    grad = gradient * actuall_batch_size
                else:
                    grad += gradient * actuall_batch_size

            # we want the overal rmse for reporting
            total_loss += torch.square(loss.detach()) * batch_configs

        return torch.sqrt(total_loss / total_points), grad / n_dimers, hess / n_dimers

    return closure_fn


def run_simulation(
    phase: descent.targets.thermo.Phase,
    key: descent.targets.thermo.SimulationKey,
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    output_dir: pathlib.Path,
) -> tuple[str, str, str]:
    """Run the given simulation and return the path to the frames from which the observables can be computed along with the phase and key."""
    import hashlib
    import pickle

    traj_hash = hashlib.sha256(pickle.dumps(key)).hexdigest()
    traj_name = f"{phase}-{traj_hash}-frames.msgpack"

    output_path = output_dir / traj_name

    config = descent.targets.thermo.default_config(phase, key.temperature, key.pressure)
    descent.targets.thermo._simulate(system, force_field, config, output_path)
    return (phase, key, output_path)
    # return output_path


def smart_liquid_closure(
    trainable: "descent.train.Trainable",
    topologies: dict[str, smee.TensorTopology],
    dataset: datasets.Dataset,
    output_dir: pathlib.Path,
    per_type_scales: dict[descent.targets.thermo.DataType, float] | None = None,
) -> descent.optim.ClosureFn:
    """Return a default closure function for training against thermodynamic
    properties.

    Notes:
        The closure computes the properties in batches of size one to reduce the memory footprint.
        The liquid simulations are deduplicated where possible.

    Args:
        trainable: The wrapper around trainable parameters.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        dataset: The dataset to train against.
        per_type_scales: The scale factor to apply to each data type.
        verbose: Whether to log additional information about predictions.

    Returns:
        The default closure function.
    """

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ):
        import openmm.unit
        from multiprocessing import get_context
        from concurrent.futures import ProcessPoolExecutor, as_completed

        total_loss, grad, hess = (
            torch.zeros(size=(1,), device=x.device.type),
            None,
            None,
        )

        entries = [*descent.utils.dataset.iter_dataset(dataset)]
        # plan the minimum number of required simulations
        required_simulations, entry_to_simulation = (
            descent.targets.thermo._plan_simulations(entries, topologies)
        )
        # run the simulations and store the path to the simulation data to be used later
        # detach the tensor to pass through the pool, only used for the simulation the attched tensor is used for the gradient later
        sim_ff = trainable.to_force_field(x.detach().clone())
        frames = {phase: {} for phase in required_simulations.keys()}
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
            simulations = []
            for phase, systems in required_simulations.items():
                for key, system in systems.items():
                    simulations.append(pool.submit(run_simulation, **{'phase': phase, 'key': key, 'system': system, 'force_field': sim_ff, 'output_dir': output_dir}))
            for job in tqdm.tqdm(as_completed(simulations), desc="Running simulations", total=len(simulations)):
                phase, key, sim_path = job.result()
                frames[phase][key] = sim_path

        # frames = {
        #     phase: {
        #         key: run_simulation(phase, key, system, force_field, output_dir)
        #         for key, system in systems.items()
        #     }
        #     for phase, systems in required_simulations.items()
        # }

        # remake the force field to make sure the graident is correctly attached to the tensors
        force_field = trainable.to_force_field(x)
        # load each of the set of frames and calculate the loss
        for entry, keys in tqdm.tqdm(
            zip(entries, entry_to_simulation, strict=True),
            desc="Calculating observables",
            ncols=80,
            total=len(entries),
        ):
            type_scale = per_type_scales.get(entry["type"], 1.0)
            ref = entry["value"] * type_scale
            # gather the observables for this entry
            from collections import defaultdict

            observables = defaultdict(dict)
            predicted = []
            for sim_key in keys.values():
                temperature = sim_key.temperature * openmm.unit.kelvin
                pressure = (
                    None
                    if sim_key.pressure is None
                    else sim_key.pressure * openmm.unit.atmospheres
                )
                obs = descent.targets.thermo._Observables(
                    *smee.mm.compute_ensemble_averages(
                            system=required_simulations['bulk'][sim_key],
                            force_field=force_field,
                            frames_path=frames['bulk'][sim_key],
                            temperature=temperature,
                            pressure=pressure,
                    ),
                )
                observables['bulk'][sim_key] = obs
            # print(observables)
            pred, _ = descent.targets.thermo._predict(
                        entry=entry,
                        keys=keys,
                        observables=observables,
                        systems=required_simulations,
                        )
            predicted.append(pred * type_scale)
            y_pred = torch.stack(predicted)
            print(y_pred)
            y_ref = smee.utils.tensor_like(ref, y_pred)

            loss = (y_pred - y_ref) ** 2

            if compute_hessian:
                print(x)
                print(y_pred)
                hessian = descent.utils.loss.approximate_hessian(x, y_pred).detach()
                if hess is None:
                    hess = hessian
                else:
                    hess += hessian
            if compute_gradient:
                gradient = torch.autograd.grad(loss, x, retain_graph=True)[0].detach()
                if grad is None:
                    grad = gradient
                else:
                    grad += gradient

            total_loss += loss.detach()
            # clear the graph
            torch.cuda.empty_cache()

        return total_loss, grad / len(dataset), hess / len(dataset)

    return closure_fn


@click.command()
@click.option(
    "--param",
    "param_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "--data-train",
    "data_train_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
)
# @click.option(
#     "--data-val",
#     "data_val_path",
#     type=click.Path(exists=True, path_type=pathlib.Path),
#     required=True,
# )
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
)
def main(
    param_path: pathlib.Path,
    data_train_path: pathlib.Path,
    # data_val_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    logging.basicConfig(level=logging.INFO)
    # load the parameters to fit
    param_config = yaml.safe_load(param_path.read_text())
    logger.info("Loaded parameter config")
    logger.info(pprint.pprint(param_config))
    # load the training data config
    data_config = yaml.safe_load(data_train_path.read_text())
    logger.info("Loaded training config")
    logger.info(pprint.pprint(data_config))

    # load up the dataset options
    thermo_dataset = (
        datasets.load_from_disk(data_config["thermo"]["source"])
        if "thermo" in data_config
        else None
    )
    energy_dataset = (
        datasets.load_from_disk(data_config["energy"]["source"])
        if "energy" in data_config
        else None
    )
    dimer_dataset = (
        datasets.load_from_disk(data_config["dimer"]["source"])
        if "dimer" in data_config
        else None
    )
    thermo_scales = data_config["thermo"]["scales"] if "thermo" in data_config else None

    print("Applying parameters")
    ff_initial, topologies = apply_parameters(
        dimer_dataset=dimer_dataset,
        thermo_dataset=thermo_dataset,
        energy_dataset=energy_dataset,
        force_field=param_config["initial"],
    )

    # edit the water assignment matrix to constrain the charges

    water_top = topologies['[O:1]([H:2])[H:3]']
    # we need to set both hydrogens to the same charge parameter, and the vsite to -2 times it
    print(water_top.parameters['Electrostatics'].assignment_matrix)
    # set the oxygen parameter to not be used
    water_top.parameters["Electrostatics"].assignment_matrix[0] = water_top.parameters["Electrostatics"].assignment_matrix[0] * 0.0
    # set the hydrogens to be the same
    water_top.parameters['Electrostatics'].assignment_matrix[2] = water_top.parameters['Electrostatics'].assignment_matrix[1]
    # set the vsite
    water_top.parameters['Electrostatics'].assignment_matrix[3] = water_top.parameters['Electrostatics'].assignment_matrix[1] * -2.0
    print(water_top.parameters['Electrostatics'].assignment_matrix)
    print(water_top.parameters['Electrostatics'].assignment_matrix @ ff_initial.potentials_by_type["Electrostatics"].parameters)
    print(ff_initial.v_sites)
    # make sure all tensors on the GPU
    water_top.to('cuda')

    # print vdW
    descent.utils.reporting.print_potential_summary(
        ff_initial.potentials_by_type["vdW"]
    )
    # # print Electro
    # descent.utils.reporting.print_potential_summary(
    #     ff_initial.potentials_by_type["Electrostatics"]
    # )

    # edit the include sections
    for k, v in param_config['parameters'].items():
        for param_type in ['include', 'exclude']:
            if param_type in v:
                param_config['parameters'][k][param_type] = [p for p in ff_initial.potentials_by_type[k].parameter_keys if p.id in v[param_type]]
    print(param_config)

    trainable = descent.train.Trainable(
        copy.deepcopy(ff_initial),
        parameters={
            k: descent.train.ParameterConfig(**v)
            for k, v in param_config["parameters"].items()
        },
        attributes={
            k: descent.train.AttributeConfig(**v)
            for k, v in param_config["attributes"].items()
        },
    )
    # build the combined closure
    logging.info("Creating closure function")
    closures_to_combine = {}
    if thermo_dataset is not None:
        liquid_dir = output_dir.joinpath('liquid-cache')
        liquid_dir.mkdir(parents=True, exist_ok=True)
        closures_to_combine["thermo"] = smart_liquid_closure(
            trainable=trainable, topologies=topologies, dataset=thermo_dataset, per_type_scales=thermo_scales, output_dir=liquid_dir
        )
    if dimer_dataset is not None:
        closures_to_combine["dimer"] = default_dimer_closure(
            trainable=trainable,
            topologies=topologies,
            dataset=dimer_dataset,
            batch_size=100,
        )
    if len(closures_to_combine) > 1:
        closure_fn = descent.utils.loss.combine_closures(
            closures_to_combine,
            weights={target: data_config[target]["weight"] for target in data_config},
            verbose=True,
        )
    else:
        closure_fn = list(closures_to_combine.values())[0]
    correct_fn = trainable.clamp

    report_fn = functools.partial(
        report,
        trainable=trainable,
        topologies=topologies,
        # thermo_dataset=thermo_dataset_val
    )

    lm_config = descent.optim.LevenbergMarquardtConfig(
        mode="adaptive", n_convergence_criteria=0, max_steps=10
    )
    x_final = descent.optim.levenberg_marquardt(
        trainable.to_values(), lm_config, closure_fn, correct_fn, report_fn
    )

    ff_final = trainable.to_force_field(x_final)
    descent.utils.reporting.print_potential_summary(ff_final.potentials_by_type["vdW"])
    # descent.utils.reporting.print_potential_summary(
    #     ff_final.potentials_by_type["Electrostatics"]
    # )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = output_dir.joinpath(f"fit-trainer-{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    # save the input files to the folder for history
    yaml.safe_dump(data_config, experiment_dir.joinpath("data-train.yaml").open("w"))
    # yaml.safe_dump(param_config, experiment_dir.joinpath("params.yaml").open("w"))
    torch.save(ff_final, experiment_dir.joinpath("final_ff.pt"))
    tops = {smiles: topology.to("cpu") for smiles, topology in topologies.items()}
    if dimer_dataset is not None:
        descent.targets.dimers.report(
            dimer_dataset,
            {
                "DEXP Initial": to_vdw_only_ff(ff_initial).to("cpu"),
                "DEXP Opt": to_vdw_only_ff(ff_final).to("cpu"),
            },
            {"DEXP Initial": tops, "DEXP Opt": tops},
            output_dir / "energies.html",
        )


if __name__ == "__main__":
    main()
