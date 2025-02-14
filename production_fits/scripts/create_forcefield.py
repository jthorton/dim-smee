# update the parameters into a forcefield we can use to run MD
import torch
from openff.toolkit import ForceField
import click
from descent.utils import reporting
import smee

@click.command()
@click.option(
    '-bff',
    '--base-ff',
    help='The path to the base offxml force field the new parameters should be saved into'
)
@click.option(
    '-fp',
    '--final-parameters',
    help='The path to the final force field fitting parameters from smee'
)
@click.option(
    '-o',
    '--output',
    help='The name of the offxml the extracted parameters should be written to.'
)
def main(base_ff: str, final_parameters: str, output: str):
    """Convert the final force field parameters into an offxml file."""
    # need load plugins for dexp
    ff: ForceField = ForceField(base_ff, load_plugins=True)
    smee_forcefield: smee.TensorForceField = torch.load(final_parameters)
    reporting.print_force_field_summary(smee_forcefield)

    for potential in smee_forcefield.potentials:
        ff_handler_name = potential.parameter_keys[0].associated_handler
        if potential.type != "vdW":
            continue
        ff_handler = ff.get_parameter_handler(ff_handler_name)

        # check if we have handler attributes to update
        attribute_names = potential.attribute_cols
        attribute_units = potential.attribute_units
        
        if potential.attributes is not None:
            opt_attributes = potential.attributes.detach().cpu().numpy()
            for j, (p, unit) in enumerate(zip(attribute_names, attribute_units)):
                setattr(ff_handler, p, opt_attributes[j] * unit)


        parameter_names = potential.parameter_cols
        parameter_units = potential.parameter_units

        for i in range(len(potential.parameters)):
            smirks = potential.parameter_keys[i].id
            if "EP" in smirks:
                # skip fitted sites to dimers, we only have water and it should be 0 anyway
                continue
            ff_parameter = ff_handler[smirks]
            opt_parameters = potential.parameters[i].detach().cpu().numpy()
            for j, (p, unit) in enumerate(zip(parameter_names, parameter_units)):
                setattr(ff_parameter, p, opt_parameters[j] * unit)

    # # only grab the vsites if present in the smee force field
    # v_site_handler = ff.get_parameter_handler('VirtualSites')

    # parameter_names = vdw_potential.parameter_cols
    # parameter_units = vdw_potential.parameter_units

    # v_site_units = smee_forcefield.v_sites.parameter_units

    # for i in range(len(vdw_potential.parameters)):
    #     smirks = vdw_potential.parameter_keys[i].id

    #     if "EP" not in smirks:
    #         parameter = ff_handler[smirks]
    #         opt_parameters = vdw_potential.parameters[i].detach().numpy()
    #         for j, (p, unit) in enumerate(zip(parameter_names, parameter_units)):
    #             setattr(parameter, p, opt_parameters[j] * unit)
    #     elif 'EP' in smirks:
    #         smirk = smirks.split()[0]
    #         parameter = v_site_handler[smirk]
    #         opt_parameters = vdw_potential.parameters[i].detach().numpy()
    #         for j, (p, unit) in enumerate(zip(parameter_names, parameter_units)):
    #             setattr(parameter, p, opt_parameters[j] * unit)
    #         # update the distance
    #         for i, param_key in enumerate(smee_forcefield.v_sites.keys):
    #             if param_key.id == smirks:
    #                 print(smirks, smee_forcefield.v_sites.parameters[i][0] * v_site_units['distance'] )
    #                 parameter.distance = smee_forcefield.v_sites.parameters[i][0] * v_site_units['distance'] 
    #                 break
    ff.to_file(output)

if __name__ == '__main__':
    main()