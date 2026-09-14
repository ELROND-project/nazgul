# Execute the query and get catalogues of galaxies 
# output a table of indexes to locate them, with 
# - coordinates
# - mass
# - redshift
# - Group and Subgroup

# Tutorial version, will not require EAGLE account
import dill
import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

from nazgul.configurations import nazgul_path
from nazgul.Translator import tutorial_sim as std_sim
# the tutorial ships its own mini data tree, not RingBearer
std_gal_dir = nazgul_path/"Tutorial/data_Tuto"/std_sim/"Gals"
# from sql_connect import exec_query -> this would be needed for the complete 
from python_tools.tools import short_SciNot
from python_tools.get_res import load_whatever

def get_gals(sim=std_sim,min_mass = "1e12",min_z="0",max_z="2",save_pkl=True,plot=True,check_prev=True,verbose=True,
            gal_dir=std_gal_dir):
    min_mass = short_SciNot(min_mass)
    min_z    = str(min_z)
    max_z    = str(max_z)
    
    cat_path = get_catpath(min_mass=min_mass,\
                           min_z=min_z,max_z=max_z,\
                           gal_dir=gal_dir)
    # select higher masses bc 1) lenses 2) else we have too many points
    myQuery = get_query(sim=sim,min_mass=min_mass,\
                        min_z=min_z,max_z=max_z)
    
    # NOTE: center of mass is in comoving coord.(cMpc)
    # Execute
    
    #check_prev = False
    #save_pkl  = False
    
    if check_prev:
        found_prev = False
        if cat_path.exists():
            try:
                myData = load_whatever(cat_path)
                #formatting might be slightly diff.
                if myData["query"].replace(" ","") == myQuery.replace(" ",""):
                    
                    found_prev =True
                    if verbose:
                        print(f"Found previous pickled catalogue:\n{cat_path}")
                else:
                    if verbose:
                        print(f"Not same query in prev. cat.:\n{cat_path}\nRerunning and overwriting.")
            except Exception as e:
                print(f"Tried and failed to load previous results :{cat_path}\nBecause of {e}\nRerunning SQL query.")
                check_prev = False    
                    
        if not found_prev:
            check_prev = False            
    if check_prev is False:
        raise RuntimeError("This version of the code cannot download new data")
        
    if plot:
        _plot(myData)
    return myData

def get_query(sim=std_sim,min_mass = "1e12",min_z="0",max_z="2"):
    min_mass = short_SciNot(min_mass)
    min_z    = str(min_z)
    max_z    = str(max_z)
    
    myQuery = "SELECT \
        gal.GroupNumber as Gn, \
        gal.SubGroupNumber as SGn, \
        gal.Redshift as z, \
        gal.Mass as M, \
        gal.CentreOfMass_x as CMx, \
        gal.CentreOfMass_y as CMy, \
        gal.CentreOfMass_z as CMz  \
    FROM \
        %s_Subhalo as gal \
    WHERE \
        (gal.Redshift between %s and %s) and \
        gal.Mass > %s \
    ORDER BY \
        gal.Redshift"%(sim,min_z,max_z,min_mass)
    return myQuery

def get_catpath(gal_dir=std_gal_dir,min_mass = "1e12",min_z="0",max_z="2"):
    min_mass = short_SciNot(min_mass)
    min_z    = str(min_z)
    max_z    = str(max_z)
    
    gal_dir = Path(gal_dir)
    cat_name_base = "CatGal" #old massive_gals.pkl
    cat_name = f"{cat_name_base}_minM{min_mass}_minZ{short_SciNot(min_z)}_maxZ{short_SciNot(max_z)}.pkl"
    cat_path = gal_dir/cat_name 
    return cat_path
    
def _plot(myData):
    """Plot of informative statistic data
    """
    logMass = np.log(myData["M"])
    str_logMass = r'log$_{10}$M${_*}$[M$_{\odot}$]'
    zGal   = myData["z"]
    plt.hist(logMass)
    plt.title("Mass of Galaxies selected")
    plt.xlabel(str_logMass)
    plt.savefig("hist_gal_mass.png")
    plt.close()
    plt.hist(zGal)
    plt.title("Redshift of Galaxies selected")
    plt.xlabel(r'z')
    plt.savefig("hist_gal_z.png")
    plt.close()
    
    
    plt.scatter(zGal,logMass,marker=".")
    plt.title("Mass at redshift")
    plt.xlabel(r'z')
    plt.ylabel(str_logMass)
    plt.savefig("gal_mvsz.png")