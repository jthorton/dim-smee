import typing

import openff
import pandas
import smee
import smee.converters
import torch
import tqdm
from rdkit import Chem
import os


class Dimer(typing.NamedTuple):
    smiles_a: str
    smiles_b: str

    group_id: str
    geometry_ids: tuple[str, ...]

    distances: torch.Tensor

    mol_a_topology: smee.TensorTopology
    mol_b_topology: smee.TensorTopology
    dimer_topology: smee.TensorTopology

    ccsd: torch.Tensor

    sapt_all: torch.Tensor
    sapt_ex: torch.Tensor
    sapt_exind: torch.Tensor
    sapt_disp: torch.Tensor
    sapt_exdisp_os: torch.Tensor
    sapt_exdisp_ss: torch.Tensor
    sapt_delta_hf: torch.Tensor
    # ANA2B terms
    ana2b_v_ana: torch.Tensor
    ana2b_v_esp: torch.Tensor
    ana2b_v_pol: torch.Tensor
    ana2b_v_d3: torch.Tensor

    coords: torch.Tensor


def apply_parameters(
    metadata: pandas.DataFrame, forcefield: openff.toolkit.ForceField, dataset_name: str
) -> tuple[smee.TensorForceField, dict[str, list[smee.TensorTopology]]]:
    interchanges = {}

    system_ids = [*metadata["system_id"].unique()]

    for system_id in tqdm.tqdm(system_ids, desc="applying parameters"):
        row = metadata[metadata["system_id"] == system_id].iloc[0]

        group_orig = row["group_orig"]
        geometry_id = row["geom_id"]

        rdkit_dimer = Chem.MolFromMolFile(
            f"{dataset_name}/geometries/{system_id}/DES{group_orig}_{geometry_id}.mol",
            removeHs=False,
        )
        rdkit_mol_a, rdkit_mol_b = Chem.GetMolFrags(rdkit_dimer, asMols=True)

        openff_mol_a = openff.toolkit.Molecule.from_rdkit(rdkit_mol_a)
        openff_mol_b = openff.toolkit.Molecule.from_rdkit(rdkit_mol_b)

        openff_dimer = openff.toolkit.Topology()
        openff_dimer.add_molecule(openff_mol_a)
        openff_dimer.add_molecule(openff_mol_b)

        interchanges[system_id] = [
            openff.interchange.Interchange.from_smirnoff(forcefield, topology)
            for topology in [
                openff_mol_a.to_topology(),
                openff_mol_b.to_topology(),
                openff_dimer,
            ]
        ]

    print("converting to tensors", flush=True)

    tensor_ff, tensor_topologies = smee.converters.convert_interchange(
        [interchange for system in system_ids for interchange in interchanges[system]]
    )
    tensor_topologies = {
        system_id: tensor_topologies[i * 3 : (i + 1) * 3]
        for i, system_id in enumerate(system_ids)
    }

    print("converted to tensors", flush=True)

    return tensor_ff, tensor_topologies


def load_dimers(
    dataset_name: str, forcefield: openff.toolkit.ForceField
) -> tuple[smee.TensorForceField, list[Dimer]]:
    """Load dimers from a DESXXX set."""

    metadata = pandas.read_csv(os.path.join(dataset_name, f"{dataset_name}.csv"), index_col=False)

    tensor_ff, tensor_topologies = apply_parameters(
        metadata=metadata, forcefield=forcefield, dataset_name=dataset_name
    )

    dimers = []

    system_ids = metadata["system_id"].unique()

    for system_id in system_ids:
        system_data = metadata[metadata["system_id"] == system_id]

        (smiles_a,) = system_data["smiles0"].unique()
        (smiles_b,) = system_data["smiles1"].unique()

        group_ids = metadata[metadata["system_id"] == system_id]["group_id"].unique()

        mol_a_topology, mol_b_topology, dimer_topology = tensor_topologies[system_id]

        for group_id in group_ids:
            group_data = system_data[system_data["group_id"] == group_id]

            geometry_ids = tuple(group_data["geom_id"].values)

            distances = torch.tensor(group_data["k_index"].values * 0.1)  # angstrom

            conformers = []

            for _, row in group_data.iterrows():
                group_orig = row["group_orig"]
                geometry_id = row["geom_id"]

                dimer_mol = Chem.MolFromMolFile(
                    f"DESS66x8/geometries/{system_id}/DES{group_orig}_{geometry_id}.mol",
                    removeHs=False,
                )
                conformers.append(dimer_mol.GetConformer().GetPositions().tolist())

            dimer = Dimer(
                smiles_a,
                smiles_b,
                group_id=group_id,
                geometry_ids=geometry_ids,
                distances=distances,
                mol_a_topology=mol_a_topology,
                mol_b_topology=mol_b_topology,
                dimer_topology=dimer_topology,
                ccsd=torch.tensor(group_data["cbs_CCSD(T)_all"].values),
                sapt_all=torch.tensor(group_data["sapt_all"].values),
                sapt_ex=torch.tensor(group_data["sapt_ex"].values),
                sapt_exind=torch.tensor(group_data["sapt_exind"].values),
                sapt_disp=torch.tensor(group_data["sapt_disp"].values),
                sapt_exdisp_os=torch.tensor(group_data["sapt_exdisp_os"].values),
                sapt_exdisp_ss=torch.tensor(group_data["sapt_exdisp_ss"].values),
                sapt_delta_hf=torch.tensor(group_data["sapt_delta_HF"].values),
                ana2b_v_d3=torch.tensor(group_data["ANA2B_V_D3"].values),
                ana2b_v_ana=torch.tensor(group_data["ANA2B_V_ana"].values),
                ana2b_v_esp=torch.tensor(group_data["ANA2B_V_esp"].values),
                ana2b_v_pol=torch.tensor(group_data["ANA2B_V_pol"].values),
                coords=torch.tensor(conformers),
            )
            dimers.append(dimer)

    return tensor_ff, dimers


def compute_base_sapt_energy(dimer: Dimer) -> torch.Tensor:
    delta_ccsd_sapt = dimer.ccsd - dimer.sapt_all

    sapt_terms = (
        dimer.sapt_ex
        + dimer.sapt_exind
        + dimer.sapt_disp
        + dimer.sapt_exdisp_os
        + dimer.sapt_exdisp_ss
        + dimer.sapt_delta_hf
    )

    return sapt_terms + delta_ccsd_sapt


def compute_dimer_vdw_energy(
    dimer: Dimer, potential: smee.TensorPotential, ff: smee.TensorForceField
) -> torch.Tensor:
    """Compute the non-electrostatic vdw energy of a dimer for any functional form. Also works with vsites"""
    mol_a_coords = dimer.coords[:, : dimer.mol_a_topology.n_atoms, :]
    mol_b_coords = dimer.coords[:, dimer.mol_a_topology.n_atoms :, :]

    dimer_coords = dimer.coords
    print(dimer_coords)
    if dimer.dimer_topology.v_sites is not None:
        if dimer.dimer_topology.v_sites.keys:
            dimer_coords = smee.add_v_site_coords(v_sites=dimer.dimer_topology.v_sites, conformer=dimer.coords, force_field=ff)
        if dimer.mol_a_topology.v_sites.keys:
            mol_a_coords = smee.add_v_site_coords(v_sites=dimer.mol_a_topology.v_sites, conformer=mol_a_coords, force_field=ff)
        if dimer.mol_b_topology.v_sites.keys:
            mol_b_coords = smee.add_v_site_coords(v_sites=dimer.mol_b_topology.v_sites, conformer=mol_b_coords, force_field=ff)

    dimer_energies = smee.compute_energy_potential(
        system=dimer.dimer_topology, conformer=dimer_coords, potential=potential
    )

    mol_a_energies = smee.compute_energy_potential(
        system=dimer.mol_a_topology, conformer=mol_a_coords, potential=potential
    )

    mol_b_energies = smee.compute_energy_potential(
        system=dimer.mol_b_topology, conformer=mol_b_coords, potential=potential
    )

    return dimer_energies - mol_a_energies - mol_b_energies


def mol_to_image(
    smiles_a: str, smiles_b: str, width: int = 400, height: int = 200
) -> str:
    import base64

    from rdkit.Chem import Draw

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    for atom in mol_a.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom in mol_b.GetAtoms():
        atom.SetAtomMapNum(0)

    dimer = Chem.CombineMols(mol_a, mol_b)
    dimer = Draw.PrepareMolForDrawing(Chem.Mol(dimer), forceCoords=True)

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(dimer)
    drawer.FinishDrawing()

    data = base64.b64encode(drawer.GetDrawingText().encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}"></img>'


def plot_energies(distances: torch.Tensor, energies: dict[str, torch.Tensor]) -> str:
    import base64
    import io

    from matplotlib import pyplot

    figure, axis = pyplot.subplots(1, 1, figsize=(4.0, 4.0))

    colors = ["red", "green", "blue", "black"]
    markers = ["x", "o", "+", "^"]

    for i, (k, v) in enumerate(energies.items()):
        axis.plot(
            distances.detach().numpy(),
            v.detach().numpy(),
            label=k,
            linestyle="none",
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
        )

    axis.set_xlabel("Distance [Å]")
    axis.set_ylabel("Energy [kcal / mol]")

    axis.legend()

    figure.tight_layout()

    with io.BytesIO() as stream:
        figure.savefig(stream, format="svg")
        data = base64.b64encode(stream.getvalue()).decode()

    pyplot.close(figure)

    return f'<img src="data:image/svg+xml;base64,{data}"></img>'