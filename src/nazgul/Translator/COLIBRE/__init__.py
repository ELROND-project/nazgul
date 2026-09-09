simsuite_name ="COLIBRE"
simsuite_short_name ="CLB" 
sim   = ["L0050N0752"]#L0025N0752"] # , L0050N1504
subsim = {"L0025N0752":["THERMAL_AGN_m5"],
          "L0050N0752":["THERMAL_AGN_m6"],
         "L0050N1504":["THERMAL_AGN_m5"]}

# Define name of accepted particles
plural_part_type_list = ["stars","gas","dark_matter","black_holes"]
singular_part_type_list = ["star","black_hole"]
part_type_list = plural_part_type_list

def check_part_type(part_type):
    assert type(part_type)==str
    if part_type not in plural_part_type_list:
        if part_type in singular_part_type_list:
            return part_type+"s"
        else:
            raise RuntimeError(f"Particle type {part_type} not an accepted particle type:\n{plural_part_type_list}")
    return part_type
