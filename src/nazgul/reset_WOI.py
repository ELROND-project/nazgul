"""
Ad-hoc solution to reset WOI that somehow got hanged
"""
import os,sys
import argparse
from glob import glob
from pathlib import Path

from nazgul.Modelling.lib_models import  model_res_base
from nazgul.combined_modelling_results import get_res_dir
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from python_tools.tools_WOI import is_someone_workin_on_it,set_workin_on_it

def get_time(file_path):
    return os.path.getmtime(file_path)
    
def verify_if_completed_model(res_dir):
    is_completed = False
    fin_res_file = res_dir+"/mcmc_post.pdf"
    init_file = res_dir+"/kappa_gal.png"
    if len(glob(fin_res_file))==1:
        is_completed = True
    if is_completed:
        # verify if it has not restarted:
        if get_time(init_file)<get_time(fin_res_file):
            return is_completed
        else:
            # is being re-run
            return False
    return is_completed 
    
def reset_WOI_path(model_path):
    # get all results directories
    all_res_dirs = glob(str(model_path)+"/snap_*/")
    # over each of them
    for res_dir in all_res_dirs:
        # check if WOI is True
        if is_someone_workin_on_it(res_dir):
            # if so, check if actually the model is completed
            print(res_dir," WOI True")
            if verify_if_completed_model(res_dir):
                print(res_dir," completed True")
                # if so, it mean the WOI got hanged wrong, set it to work_to_set
                set_workin_on_it(res_dir,wrk=False)
                print(res_dir," WOI ",is_someone_workin_on_it(res_dir))
                print(" ") 

def reset_WOI_path_to_continue_run(model_path):
    """
    Assume you had a run that ended due to server errors (e.g. memory errors)
    In this case you want to restart it, without rerunning completed models (WOI=True), 
    but rerunning those that were only started but never finished (WOI=False)
    """
    # get all results directories
    all_res_dirs = glob(str(model_path)+"/snap_*/")
    # over each of them
    for res_dir in all_res_dirs:
        # check if they were successful:
        if verify_if_completed_model(res_dir):
            # if so, set WOI to True
            print(f"{res_dir}: completed, setting WOI=True")
            set_workin_on_it(res_dir,wrk=True)
        else:
            # botched models set WOI to False
            print(f"{res_dir}: botched, setting WOI=False")
            set_workin_on_it(res_dir,wrk=False)
                
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Reset WOI")
    parser.add_argument('-m','--model',type=str,
                        dest="model",default=None,
                        help=f"Name of type of model (def.: all)")
    parser.add_argument('-cnt_run','--continue_run',default=False,action="store_true",
                        dest="continue_run",help=f"If the run has to be continue - WARNING: If activated, it effectively does the opposite of the standard: set completed models to WOI=True and unfinished models to WOI=False")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    
    args       = parser.parse_args()
    model_name = args.model
    sim      = args.sim
    subsim   = args.subsim
    simsuite = args.simsuite
    continue_run =  args.continue_run
    kw_sim = {"simsuite":simsuite,
              "sim":sim,
              "subsim":subsim}
    if model_name is None:
        model_paths = glob(str(model_res_base)+"/*/")
    else:
        res_dir = get_res_dir(model_name,kw_sim=kw_sim)
        model_paths = glob(str(res_dir)+"/")

    if continue_run:
        print("Warning: Setting WOI=True for completed models, False for incompleted ones")
        print("Confirm reading:")
        input()
    
    for model_path in model_paths:
        if not continue_run:
            reset_WOI_path(model_path)
        else:
            reset_WOI_path_to_continue_run(model_path)