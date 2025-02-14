import datasets
import descent.utils.molecule

dataset = datasets.load_from_disk('binary-combined-training-small-v1-water-co-opt')
wanted = [('CC(C)O', 'O'), ('CO', 'O'), ('O', None)]
to_keep = []
for i, entry in enumerate(dataset):
    smiles_a = descent.utils.molecule.unmap_smiles(entry['smiles_a']) if entry['smiles_a'] is not None else None
    smiles_b = descent.utils.molecule.unmap_smiles(entry['smiles_b']) if entry['smiles_b'] is not None else None
    smiles = (smiles_a, smiles_b)
    if smiles in wanted and entry['temperature'] == 298.15:
        to_keep.append(i)
new_dataset = dataset.select(indices=to_keep)
new_dataset.save_to_disk('lj-small-test')
