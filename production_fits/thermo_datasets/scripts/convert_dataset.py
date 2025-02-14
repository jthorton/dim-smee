from descent.targets.thermo import create_from_evaluator
import pathlib

dataset_file = pathlib.Path("../evaluator_data/sage-train-v1.json")
liquid_dataset = create_from_evaluator(dataset_file)

liquid_dataset.save_to_disk("../sage-train-v1")
