from openff.toolkit import ForceField
from openff.toolkit.utils.exceptions import ParameterLookupError
import copy

sage_small = ForceField('../openff-2.2.0-small_mol.offxml')
de_force = ForceField('de-force-1.0.1.offxml', load_plugins=True)
old_handler = de_force.get_parameter_handler('DoubleExponential')

new_ff = copy.deepcopy(de_force)

# loop over the sage terms if they are not in the de_force translate and add them
new_ff.deregister_parameter_handler('DoubleExponential')
new_handeler = new_ff.get_parameter_handler('DoubleExponential')
# copy the attributes
new_handeler.alpha = old_handler.alpha
new_handeler.beta = old_handler.beta
for i, parameter in enumerate(sage_small.get_parameter_handler('vdW').parameters):
    try:
        de_param = old_handler[parameter.smirks]
        de_param.id = f'de_{i}'
        new_handeler.add_parameter(parameter=de_param)
    except ParameterLookupError:
        # create a new parameter from sage
        data = {
            'smirks': parameter.smirks,
            'id': f'de_{i}',
            'epsilon': parameter.epsilon,
            'r_min': (parameter.sigma ** 6 * 2) ** (1 / 6)
        }
        new_handeler.add_parameter(parameter_kwargs=data)

new_ff.to_file('de-force-small-mol.offxml')