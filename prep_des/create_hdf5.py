import h5py
from openff.toolkit.topology import Molecule
import os
import csv
import tqdm
import numpy
from openmm import unit


def main():
    """
    Create a HDF5 of the dataset.
    """

    csv_file = "DESS66x8/DESS66x8.csv"
    geometries = "DESS66x8/geometries/"

    ref_db = h5py.File("DESS66x8.hdf5", "w")

    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm.tqdm(reader, ncols=80, total=reader.line_num, desc="Create ref database"):
            position_file = os.path.join(geometries, row["system_id"], f"DESS66x8_{row['geom_id']}.mol")
            off_mol = Molecule.from_file(position_file)
            ccsd_energy = float(row["cbs_CCSD(T)_all"])
            k_index = int(row["k_index"])
            name = f"{row['system_id']}-{row['group_id']}-{row['geom_id']}"
            # create a group for the entry
            group = ref_db.create_group(name=name)
            group.create_dataset("smiles", data=[off_mol.to_smiles(mapped=True)], dtype=h5py.string_dtype())
            conformations = group.create_dataset("conformation", data=[off_mol.conformers[0].value_in_unit(unit.angstrom)], dtype=numpy.float64)
            conformations.attrs["units"] = "angstrom"
            group.create_dataset("atomic_numbers", data=[atom.atomic_number for atom in off_mol.atoms], dtype=numpy.int16)
            int_energy = group.create_dataset("interaction_energy", data=[ccsd_energy], dtype=numpy.float64)
            int_energy.attrs["units"] = "kcal / mol"
            group.create_dataset("kindex", data=[k_index], dtype=numpy.int16)
            group.create_dataset("title", data=[row['system_name']], dtype=h5py.string_dtype())


if __name__ == "__main__":
    main()
