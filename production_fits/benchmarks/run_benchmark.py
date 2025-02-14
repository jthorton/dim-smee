import openmm.unit
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
import click
import pandas as pd
import descent.targets.thermo
import descent.optim
import descent.targets
import descent.train
import pathlib
import smee
import datasets
import openff.interchange
import tqdm
import openff.toolkit

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

    # config = descent.targets.thermo.default_config(phase, key.temperature, key.pressure)
    # descent.targets.thermo._simulate(system, force_field, config, output_path)
    return (phase, key, output_path)

def apply_parameters(
    force_field: str,
    thermo_dataset: datasets.Dataset,
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    unique_smiles = descent.targets.thermo.extract_smiles(thermo_dataset)
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


@click.command()
@click.option('-f', '--force-field', help="The name of the offxml on which to run the benchmark")
@click.option('-d', '--dataset', help='The name of the descent thermo dataset to use in the benchmark')
@click.option('-o', '--output', help='The folder to write the results to.')
def main(force_field: str, dataset: str, output: str):
    """Run the benchmark on the input dataset using the provided force field
    Results are printed to terminal as a pandas table and stored in a csv.
    """
    output_folder = pathlib.Path(output)
    output_folder.mkdir(exist_ok=True)
    # write the ff to the output folder
    off = openff.toolkit.ForceField(force_field, load_plugins=True)
    off.to_file(output_folder / 'force_field.offxml')

    # load the dataset and extract the entries
    thermo_dataset = datasets.load_from_disk(dataset)
    entries = [*descent.utils.dataset.iter_dataset(thermo_dataset)]
    # create the force field and the topologies
    tensor_ff, tensor_tops = apply_parameters(force_field=force_field, thermo_dataset=thermo_dataset)
    # create a trainer to help with later
    trainable = descent.train.Trainable(
        tensor_ff,
        parameters={
        },
        attributes={}
    )

    # plan the minimum number of required simulations
    required_simulations, entry_to_simulation = (
        descent.targets.thermo._plan_simulations(entries, tensor_tops)
    )
    x = trainable.to_values()
    sim_ff = trainable.to_force_field(x.detach().clone())
    # run the simulations and store the path to the simulation data to be used later
    # detach the tensor to pass through the pool, only used for the simulation the attched tensor is used for the gradient later
    frames = {phase: {} for phase in required_simulations.keys()}
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        simulations = []
        for phase, systems in required_simulations.items():
            for key, system in systems.items():
                simulations.append(pool.submit(run_simulation, **{'phase': phase, 'key': key, 'system': system, 'force_field': sim_ff, 'output_dir': output_folder}))
        for job in tqdm.tqdm(as_completed(simulations), desc="Running simulations", total=len(simulations)):
            phase, key, sim_path = job.result()
            frames[phase][key] = sim_path

    # load each of the set of frames and calculate the loss
    results_table = []
    for entry, keys in tqdm.tqdm(
        zip(entries, entry_to_simulation, strict=True),
        desc="Calculating observables",
        ncols=80,
        total=len(entries),
    ):
        std_ref = "" if entry["std"] is None else f" ± {float(entry['std']):.3f}"
        # gather the observables for this entry
        from collections import defaultdict

        observables = defaultdict(dict)
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
                        force_field=trainable.to_force_field(x),
                        frames_path=frames['bulk'][sim_key],
                        temperature=temperature,
                        pressure=pressure,
                ),
            )
            observables['bulk'][sim_key] = obs
        # print(observables)
        pred, std = descent.targets.thermo._predict(
                    entry=entry,
                    keys=keys,
                    observables=observables,
                    systems=required_simulations,
                    )
        results_table.append(
            {
            "type": f'{entry["type"]} [{entry["units"]}]',
            "smiles_a": descent.utils.molecule.unmap_smiles(entry["smiles_a"]),
            "smiles_b": (
                ""
                if entry["smiles_b"] is None
                else descent.utils.molecule.unmap_smiles(entry["smiles_b"])
            ),
            "pred": f"{float(pred.item()):.3f} ± {float(std.item()):.3f}",
            "ref": f"{float(entry['value']):.3f}{std_ref}",
        }
        )
    
    df = pd.DataFrame(results_table)
    print(df)
    df.to_csv(output_folder / 'results_table.csv')

if __name__ == '__main__':
    main()