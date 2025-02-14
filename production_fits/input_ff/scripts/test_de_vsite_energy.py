import pathlib
import datasets
import torch
import descent.targets.dimers
import descent.utils.dataset
import smee
import openff.toolkit
import openff.interchange
import smee.converters
import openff.units

water = openff.toolkit.Molecule.from_smiles('O')
water.generate_conformers()

de = openff.toolkit.ForceField('../de-force-small-mol.offxml', load_plugins=True, allow_cosmetic_attributes=True)
water_conformer = torch.tensor(water.conformers[0].m_as(openff.units.unit.angstrom))
interchange = openff.interchange.Interchange.from_smirnoff(de, water.to_topology())
tip3p_tensor_ff, [tip3p_tensor_topology] = smee.converters.convert_interchange(
    interchange
)
# tip3p_tensor_ff.potentials = [p for p in tip3p_tensor_ff.potentials if p.type == 'vdW']
vdw_potential = tip3p_tensor_ff.potentials_by_type['vdW']
vdw_potential.parameters.requires_grad = True
tip3p_tensor_ff.v_sites.parameters.requires_grad=True
electro_potential = tip3p_tensor_ff.potentials_by_type['Electrostatics']
electro_potential.parameters.requires_grad = True
tip3p_dimer = smee.TensorSystem(
    [tip3p_tensor_topology], n_copies=[2], is_periodic=False
)

tip3p_conformers = torch.stack(
    [
        torch.vstack([water_conformer, water_conformer + torch.tensor(1.5 + i * 0.05)])
        for i in range(1)
    ]
)
descent.utils.reporting.print_potential_summary(
        tip3p_tensor_ff.potentials_by_type["vdW"]
    )
descent.utils.reporting.print_potential_summary(
        tip3p_tensor_ff.potentials_by_type["Electrostatics"]
    )

# compute with descent to handle the vsites
energy = descent.targets.dimers.compute_dimer_energy(
    topology_a=tip3p_tensor_topology, 
    topology_b=tip3p_tensor_topology, 
    force_field=tip3p_tensor_ff, 
    coords=tip3p_conformers)

print(energy)
energy.backward()
print(vdw_potential.parameter_cols)
print(vdw_potential)
print(vdw_potential.parameters)
print(vdw_potential.parameters.grad)
print(tip3p_tensor_ff.v_sites.parameters.grad)
print(electro_potential.parameters.grad)
print(tip3p_tensor_topology.parameters['Electrostatics'].assignment_matrix.to_dense() @ electro_potential.parameters)