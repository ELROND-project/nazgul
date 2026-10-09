from   nazgul.Translator import std_sim,std_subsim,std_simsuite,test_sim,tutorial_sim
import nazgul.configurations as conf

path_nazgul        = conf.nazgul_path
path_nazgul_origin = conf.nazgul_path_origin if conf.nazgul_path_origin is not None else conf.nazgul_path

std_data_dir = path_nazgul/"RingBearer"
# (base)/tmp will be a collector of intermediate, mildly useful plots/results, with the advantage of being easily accessible
tmp_dir = path_nazgul/"tmp"
#mkdir(tmp_dir)

# result directory
results_dir = path_nazgul/"results"
#mkdir(results_dir)

std_simsuite_dir = std_data_dir/std_simsuite # which simulation suite
std_sim_dir      = std_simsuite_dir/std_sim  # which simulation

# path to LensPop directory
LensPop_dir = path_nazgul/"LensPop"
