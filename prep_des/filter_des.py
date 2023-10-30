"""
Filter the DES dataset to only contain records with elements of interest
"""
import os

import pandas as pd
from rdkit import Chem

# I, P and Br not covered by the ANA2B model
elements_to_keep = ["H", "C", "N", "O", "F", "S", "Cl"]
# Ions can not be used with ANA2B. Most will be caught by the element filter just need to remove some halogen ones
ions_to_remove = ["[F-]", "[Cl-]", "[H][H]"]


def unwanted_molecules_filter(row) -> bool:
    """Pandas filter function which returns True if the record contains unwanted elements, ions or charged molecules."""
    for element in row["elements"].split():
        if element not in elements_to_keep:
            return True
    # check for ions
    for smiles in [row["smiles0"], row["smiles1"]]:
        if smiles in ions_to_remove:
            return True

    # slow check last - remove charged
    for smiles in [row["smiles0"], row["smiles1"]]:
        rdkit_mol = Chem.MolFromSmiles(smiles)
        total_charge = sum([atom.GetFormalCharge() for atom in rdkit_mol.GetAtoms()])
        if total_charge != 0:
            return True

    return False


def main():
    dataset_name = "DES370K"

    raw_dataset = pd.read_csv(os.path.join(dataset_name, dataset_name + ".csv"))

    unwanted_rows = raw_dataset.apply(unwanted_molecules_filter, axis=1)

    filtered_des = raw_dataset[~unwanted_rows]

    filtered_des.to_csv(dataset_name + "-filtered.csv")

    # work out the total number of unique systems and the total number of data points
    print(
        f"Total number of unique systems after filtering {len(filtered_des['system_id'].unique())}"
    )
    print(f"Total number of data points after filtering {len(filtered_des)}")


if __name__ == "__main__":
    main()
