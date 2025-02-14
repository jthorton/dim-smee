import pathlib

import datasets
import datasets.distributed
import torch
import torch.distributed

import descent.optim
import descent.targets.dimers
import descent.utils.loss
import descent.utils.reporting
import smee
import smee.converters
import smee.utils
import functools
import multiprocessing
import tqdm
import logging
import openff.interchange
import openff.toolkit
import pandas

_LOGGER = logging.getLogger(__name__)


def compute_dimer_ref_energy(data: pandas.DataFrame, *_) -> torch.Tensor:
    return torch.tensor(data["cbs_CCSD(T)_all"].values)


def build_interchange(smiles, force_field_paths: tuple[str, ...]):
    return openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField(*force_field_paths, load_plugins=True),
        openff.toolkit.Molecule.from_mapped_smiles(smiles).to_topology(),
    )


def apply_parameters(
    dataset: datasets.Dataset, *force_field_paths: str
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    unique_smiles = descent.targets.dimers.extract_smiles(dataset)

    build_interchange_fn = functools.partial(
        build_interchange, force_field_paths=force_field_paths
    )

    with multiprocessing.Pool() as pool:
        interchanges = list(
            tqdm.tqdm(
                pool.imap(build_interchange_fn, unique_smiles),
                total=len(unique_smiles),
                desc="building interchanges",
            )
        )

    _LOGGER.info("converting interchanges to smee")

    force_field, topologies = smee.converters.convert_interchange(interchanges)
    force_field.potentials = [
        p for p in force_field.potentials if p.type in ["vdW", "Electrostatics"]
    ]

    return force_field, {
        smiles: topology for smiles, topology in zip(unique_smiles, topologies)
    }


def main():

    dataset_path = pathlib.Path("datasets/lj-filtered-no-charge-small-mol")
    force_field_initial, topologies = torch.load(
        dataset_path.joinpath("ff_and_tops.pt")
    )
    force_field_initial.potentials = [
        p for p in force_field_initial.potentials if p.type == "vdW"
    ]
    ff_final = torch.load(
        "fits/lj-filtered-no-charge-small-mols/fit-20240717-152645/epoch_2999_forcefield.pt"
    )
    ff_final.potentials = [p for p in ff_final.potentials if p.type == "vdW"]
    # force_fields_by_name = {
    #     # 'DEXP-opt': 'final-dexp-complete.offxml',
    #     # 'DEXP-pub': 'de-force-complete.1.0.2.offxml',
    #     'LJ Sage-2.2': 'sage-tip3p-contracted-hs.offxml',
    #     'Sage-Tip3p': 'openff-2.0.0.offxml',
    #     'Sage-Tip3p-CCSD-fit': 'sage-tip3p-dimer-start.offxml'
    # }

    # load the dataset and get the unique smiles
    dataset = datasets.load_from_disk(dataset_path.as_posix())

    # # filter the dataset to only keep water water interactions
    # def filter_dimers(dimer: descent.targets.dimers.Dimer) -> bool:
    #     if torch.isnan(dimer["energy"]).any():
    #         return False

    #     smiles_a, smiles_b = dimer["smiles_a"], dimer["smiles_b"]
    #     if smiles_a == smiles_b == "[O:1]":
    #         return True

    #     return False

    # dataset = dataset.filter(filter_dimers, with_indices=False, batched=False)
    # _LOGGER.info(f"kept {len(dataset)} after filtering")

    # smee_topologies = dict()

    # for name, force_field in tqdm.tqdm(force_fields_by_name.items(), desc='Converting force fields', ncols=80):
    #     tensor_force_field, topologies = apply_parameters(dataset, force_field)

    #     tensor_force_field.potentials = [
    #         p for p in tensor_force_field.potentials if p.type in ["vdW", "Electrostatics"]
    #     ]

    #     smee_topologies[name] = topologies
    #     force_fields_by_name[name] = tensor_force_field
    # print(smee_topologies.keys())

    # create the report
    descent.targets.dimers.report(
        dataset,
        {"LJ Sage-2.2": force_field_initial, "LJ Opt": ff_final},
        {"LJ Sage-2.2": topologies, "LJ Opt": topologies},
        pathlib.Path("lj-comp-ccsd-370k-water.html"),
    )


if __name__ == "__main__":
    main()
