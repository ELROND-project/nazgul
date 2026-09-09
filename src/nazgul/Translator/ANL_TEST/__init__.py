# Analytical Test - to verify the pipeline
simsuite_name = "ANL_TEST"
simsuite_short_name ="ANLT"

# Available simulations:
sim     = ["SIS","SIE"]

# Define name of accepted particles
part_type_list = ["part"]

def check_part_type(part_type):
    assert type(part_type)==str
    if part_type != part_type_list[0]:
        warnings.warn(f"Ignoring {part_type} as there is only one particle type for this analytical profile: {part_type_list}")
    return part_type_list[0]