"""
Create a version of sage where the H LJ centers are contracted towards the bonded heavy atom while the charge is left
at the normal location. The contracted H terms are modeled as bond charge virtual site type.
"""

def main():

    from openff.toolkit import ForceField
    from openff.units import unit

    openff_ff = ForceField("openff-2.1.0.offxml")

    lj_handler = openff_ff.get_parameter_handler('vdW')
    vsites = openff_ff.get_parameter_handler("VirtualSites")

    # define all of the current h related smirks parameters and the conversion to a vsite smirks
    smirks_to_convert = [
        ("[#1:1]", "[#1:1]~[*:2]"),
        ("[#1:1]-[#6X4]", "[#1:1]-[#6X4:2]"),
        ("[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]", "[#1:1]-[#6X4:2]-[#7,#8,#9,#16,#17,#35]"),
        ("[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", "[#1:1]-[#6X4:2](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]"),
        ("[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", "[#1:1]-[#6X4:2](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]"),
        ("[#1:1]-[#6X4]~[*+1,*+2]", "[#1:1]-[#6X4:2]~[*+1,*+2]"),
        ("[#1:1]-[#6X3]", "[#1:1]-[#6X3:2]"),
        ("[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]", "[#1:1]-[#6X3:2]~[#7,#8,#9,#16,#17,#35]"),
        ("[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]", "[#1:1]-[#6X3:2](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]"),
        ("[#1:1]-[#6X2]", "[#1:1]-[#6X2:2]"),
        ("[#1:1]-[#7]", "[#1:1]-[#7:2]"),
        ("[#1:1]-[#8]", "[#1:1]-[#8:2]"),
        ("[#1:1]-[#16]", "[#1:1]-[#16:2]"),
        ("[#1:1]-[#8X2H2+0]-[#1]", "[#1:1]-[#8X2H2+0:2]-[#1]")  # water
    ]

    for current_lj, new_vsite in smirks_to_convert:
        # remove the current handler
        parameter_index = lj_handler._index_of_parameter(key=current_lj)
        current_parameter = lj_handler._parameters.pop(parameter_index)

        # add a new vsite
        vsites.add_parameter(
            {
                "smirks": new_vsite,
                "epsilon": current_parameter.epsilon,
                "sigma": current_parameter.sigma,
                "type": "BondCharge",
                "match": "all_permutations",
                "distance": -0.1 * unit.angstrom,
                "charge_increment": [0, 0] * unit.elementary_charge,
            }
        )

    # save back to file
    openff_ff.to_file("contracted_h_sage_lj.offxml")


if __name__ == "__main__":
    main()