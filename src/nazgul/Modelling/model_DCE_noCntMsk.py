# Model all lenses where LOS IS simulated AND w. LOS in the model (adapted from model_SNS_DCE)
# using DCE(PLP) - Drifting Cored Elliptical power law potential
# further more, finally implement the source as elliptical sersic instead of simple circular
# here the center is not masked, to try to fit the central image to constrain theta_core
import sys
import numpy as np
from python_tools.tools import mkdir
from nazgul.Modelling.pathfinder import model_res_base

# LOS simulated as dependent on gal name (not z)
from nazgul.mount_doom.fuel_tank import lensfuel_los_gal as lensfuel

lens_model_list   = ['DCEPLP','LOS_MINIMAL']
source_model_list = ["SERSIC_ELLIPSE"]
res_dir_base      = model_res_base/"DCE_noCntMsk/"
mkdir(res_dir_base)
description_model = "Simulate variable LOS, and model the lens with cored EPL with drifting center + LOS minimal, and an elliptical sersic source and not masking the center - this is only considering the lenses from the golden samples of the DCE model"

from nazgul.Modelling.model_DCE import get_kwargs_params

kw_model = {"lens_model_list":lens_model_list,
            "source_model_list":source_model_list,
            "get_kwargs_params":get_kwargs_params}

from nazgul.Translator import std_kw_sim
from nazgul.ultimate_loader import ultimate_lenssys_loader
from nazgul.select_golden_sample import get_GS_cat,get_GS_names

if __name__=="__main__":
    from nazgul.Modelling.lib_models import run_model
    # The following is specific for the first run of this model: we want to model specifically the subsample of GS of the DCE model, as a start
    # to consider later to skip and do all lenses:
    GS_lenses_model_name = "DCE"
    snap = 110
    kw_GS_lenses         = get_GS_cat(model=GS_lenses_model_name)
    names_GS_lenses      = get_GS_names(kw_GS_lenses)
    
    lenses2model = [ultimate_lenssys_loader(str(gs_lens),lensFuel=lensfuel,prj=None,kw_sim=std_kw_sim,
                                             snap=snap) for gs_lens in names_GS_lenses]

    run_model(res_dir_base,
              lensfuel=lensfuel,
              kw_model=kw_model,
              gauss_tE_prior=True,
              prog_name=sys.argv[0],
              mask_center=False,
              lenses2model= lenses2model,
              description=description_model)
