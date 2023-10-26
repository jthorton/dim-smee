import functools
import multiprocessing

import Electrostatics
import numpy as np
import pandas
import Polarization
from AMPParams import AMPParams
from Constants import BOHR_TO_ANGSTROM, H_TO_KJ
from dftd3.interface import DispersionModel, RationalDampingParam
from GraphBuilder import GraphBuilder
from HybridFF import HybridFF
from rdkit import Chem
from scipy.spatial.distance import cdist
from tqdm import tqdm
from Utilities import to_Z


def calculate_d3_dimer(
    coords_1, coords_2, elements_1, elements_2, functional="pbe0", damping_params=None
):
    elements_1, elements_2 = to_Z(elements_1), to_Z(elements_2)
    energy_monomer_1, energy_monomer_2, energy_dimer = [], [], []
    if damping_params is None:
        damping_params = RationalDampingParam(method=functional)
    for monomer_1 in coords_1:
        model = DispersionModel(elements_1, monomer_1 / BOHR_TO_ANGSTROM)
        e_1 = model.get_dispersion(damping_params, grad=False)["energy"]
        energy_monomer_1.append(e_1)

    for monomer_2 in coords_2:
        model = DispersionModel(elements_2, monomer_2 / BOHR_TO_ANGSTROM)
        e_2 = model.get_dispersion(damping_params, grad=False)["energy"]
        energy_monomer_2.append(e_2)

    dimer_coords = np.concatenate((coords_1, coords_2), axis=1)
    dimer_elements = np.concatenate((elements_1, elements_2))
    for dimer in dimer_coords:
        model = DispersionModel(dimer_elements, dimer / BOHR_TO_ANGSTROM)
        e_dimer = model.get_dispersion(damping_params, grad=False)["energy"]
        energy_dimer.append(e_dimer)

    energy_monomer_1, energy_monomer_2, energy_dimer = (
        np.hstack(energy_monomer_1),
        np.hstack(energy_monomer_2),
        np.hstack(energy_dimer),
    )
    return np.float32((energy_dimer - energy_monomer_1 - energy_monomer_2) * H_TO_KJ)


def calculate_ana2b_energies(row, dataset):
    system_id = row["system_id"]
    group_orig = row["group_orig"]
    geometry_id = row["geom_id"]

    dimer = Chem.MolFromMolFile(
        f"{dataset}/geometries/{system_id}/DES{group_orig}_{geometry_id}.mol",
        removeHs=False,
    )
    monomer_1, monomer_2 = Chem.GetMolFrags(dimer, asMols=True)

    coords_1 = monomer_1.GetConformer().GetPositions().astype(np.float32)
    coords_2 = monomer_2.GetConformer().GetPositions().astype(np.float32)

    elements_1 = np.array([atom.GetSymbol() for atom in monomer_1.GetAtoms()])
    elements_2 = np.array([atom.GetSymbol() for atom in monomer_2.GetAtoms()])

    builder = GraphBuilder()
    model_params = AMPParams()
    hybrid_ff = HybridFF(pol0=False)

    distance_matrix = cdist(coords_1, coords_2).astype(np.float32)[None]
    # Construct molecular graphs (can be constructed once for each molecular topology)
    graph_1 = builder.from_coords(coords_1, elements_1)
    graph_2 = builder.from_coords(coords_2, elements_2)
    coords_1, coords_2 = coords_1[None], coords_2[None]
    # Predict atomic multipoles (geometric graphs)
    graph, _ = model_params.predict(coords_1, elements_1)
    monos_1, dipos_1, quads_1, ratios_1 = (
        graph.monos,
        graph.dipos,
        graph.quads,
        graph.ratios,
    )
    graph, _ = model_params.predict(coords_2, elements_2)
    monos_2, dipos_2, quads_2, ratios_2 = (
        graph.monos,
        graph.dipos,
        graph.quads,
        graph.ratios,
    )
    multipoles = (monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2)
    alpha_coeffs_1, alpha_coeffs_2 = (
        hybrid_ff.Alpha(graph_1, 1)[None],
        hybrid_ff.Alpha(graph_2, 1)[None],
    )
    alphas_1, alphas_2 = alpha_coeffs_1 * ratios_1, alpha_coeffs_2 * ratios_2
    # Calculate interaction terms
    V_esp, F1, F2 = Electrostatics.esp_dimer(
        coords_1, coords_2, distance_matrix, multipoles, with_field=True
    )
    V_pol, mu_ind_1, mu_ind_2 = Polarization.ind_dimer(
        coords_1, coords_2, alphas_1, alphas_2, F1, F2
    )
    V_D3 = calculate_d3_dimer(coords_1, coords_2, elements_1, elements_2)
    V_ana = hybrid_ff.ANA(
        graph_1,
        graph_2,
        coords_1,
        coords_2,
        distance_matrix,
        multipoles,
        (mu_ind_1, mu_ind_2),
        coords_1.shape[0],
    )

    KJ_TO_KCAL = 1.0 / 4.184

    return {
        **row,
        "ANA2B_V_esp": float(V_esp) * KJ_TO_KCAL,
        "ANA2B_V_pol": float(V_pol) * KJ_TO_KCAL,
        "ANA2B_V_D3": float(V_D3) * KJ_TO_KCAL,
        "ANA2B_V_ana": float(V_ana) * KJ_TO_KCAL,
    }


def main():
    dataset = "DESS66x8"

    metadata = pandas.read_csv(f"{dataset}/{dataset}.csv", index_col=False)

    rows_initial = [row for _, row in metadata.iterrows()]

    energy_fn = functools.partial(calculate_ana2b_energies, dataset=dataset)

    with multiprocessing.Pool() as pool:
        rows = list(tqdm(pool.imap(energy_fn, rows_initial), total=len(rows_initial)))

    metadata = pandas.DataFrame(rows)
    metadata.to_csv(f"{dataset}-ANA2B.csv", index=False)


if __name__ == "__main__":
    main()