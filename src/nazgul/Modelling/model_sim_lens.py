import json,dill
import numpy as np
from pathlib import Path
from copy import copy,deepcopy
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST

from python_tools.read_fits import load_fits
from python_tools.tools import mkdir,to_dimless


from nazgul.plot_PL import plot_all
from nazgul.pathfinder import tmp_dir
from nazgul.masking import mask_SEAGLE,mask_center
from nazgul.mount_doom.generate_gal_lens import wrapper_get_rnd_lens

lens_model_list   = ['EPL','SHEAR_GAMMA_PSI']
source_model_list = ["SERSIC"]
res_dir_base      = tmp_dir/"modelling_sim_lenses/"

if __name__=="__main__":
    lens = wrapper_get_rnd_lens(kw_lenspart={"min_thetaE":.9},
                                reload=False)
    res_dir = Path(f"{res_dir_base}/{lens.name}")
    mkdir(res_dir)
    plot_all(lens,skip_caustic=False)
    """
    band_HST = HST(band='WFC3_F160W', psf_type="GAUSSIAN")
    """
    band_HST = HST(band='WFC3_F160W', psf_type="PIXEL")
    psf_path = Path(f"./ObsData/HST/WFC3/F160W/PSFSTD_WFC3IR_F160W.fits")
    # the following contain 9 PSF depending on their position in the CCD
    # we will consider the central position -> i think it should be the second to last (not too important here )
    psf = load_fits(psf_path)[-2]
    # we can supersample it
    pssf    = 3
    psf_ss  = zoom(psf,pssf)
    kwargs_psf_HST = {"kernel_point_source":psf_ss,
                      "point_source_supersampling_factor":pssf}

    multi_band_list = lens.sim_multi_band_list(band=band_HST,
                                               kwargs_psf=kwargs_psf_HST)
    
    # models
    kwargs_model = {'lens_model_list': lens_model_list,
                    'source_light_model_list': source_model_list}
    
    # masking inner and outer of thetaE -> nope, follow SEAGLE approach
    #image = kwargs_data["image_data"]
    mask_HD =  mask_SEAGLE(lens)*mask_center(lens)
    _lns  = deepcopy(lens)
    _lns.image_sim = multi_band_list[0][0]["image_data"]
    _lns.pixel_num = _lns.image_sim.shape[0]
    # deltapix is now a property - def. by pixel_num
    #_lns.deltaPix  = to_dimless(_lns.radius)*2/_lns.pixel_num
    mask_LD = mask_SEAGLE(_lns)*mask_center(_lns)
    #DEBUG 
    plt.close()
    plt.close("all")
    kw_extents = lens.kw_extents
    extent_arcsec = kw_extents["extent_arcsec"]
    kw_plot = {"cmap":"hot","extent":extent_arcsec,"origin":"lower"}
    fig,axes = plt.subplots(2,2, figsize=(10, 10))
    ax =axes[0][0]
    im0 = ax.imshow(np.log10(lens.image_sim),**kw_plot)
    ax.set_title("Image")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')   
    
    ax =axes[0][1]
    ax.set_title("Masked Image")
    mask_nan = copy(mask_HD)
    mask_nan[np.where(mask_HD==0)] = np.nan
    im0 = ax.imshow(np.log10(mask_nan*lens.image_sim),**kw_plot)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')   

    ax =axes[1][0]
    im0 = ax.imshow(np.log10(_lns.image_sim),**kw_plot)
    ax.set_title("Image (Realistic)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')   
    
    ax =axes[1][1]
    ax.set_title("Masked Image (Realistic)")
    mask_nan = copy(mask_LD)
    mask_nan[np.where(mask_LD==0)] = np.nan
    im0 = ax.imshow(np.log10(mask_nan*_lns.image_sim),**kw_plot)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')   
    
    nm = f"{res_dir}/{lens.name}_masked_im.png"
    print(f"Saving {nm}")
    plt.savefig(nm)
    mask = mask_LD
    #print("mask shape",mask.shape)
    #print("ISSUE: the mask is defined on the high-res simulated image, not on the low res., realistic one") 

    
    kwargs_likelihood = {'check_bounds': True, # punish out-of-bound soulutions
                     #'force_no_add_image': False,
                     'source_marg': False, # marginalization addition on the imaging likelihood based on the covariance of the inferred linear coefficients 
                     #'image_position_uncertainty': 0.004,
                     #'check_matched_source_position': True,
                     #'source_position_tolerance': 0.001,
                     'source_position_sigma': 0.01,
                     #'prior_lens': prior_lens
                      "image_likelihood_mask_list": [mask]  
                             }


    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    # Params:
    # initial guess of non-linear parameters, we chose different starting parameters than the truth #
    kwargs_lens_init = [{'theta_E': .5, 
                    'e1': 0, 'e2': 0, 'gamma': 2., 
                    'center_x': 0., 'center_y': 0},
                    {'gamma_ext': 0.01, 'psi_ext': 0.}]
    kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 1., 'center_x': 0, 'center_y': 0}]
    
    # initial spread in parameter estimation #
    kwargs_lens_sigma = [{'theta_E': 0.3, 
                          'e1': 0.2, 'e2': 0.2, 'gamma': .2, 
                          'center_x': 0.1, 'center_y': 0.1},
                        {'gamma_ext': 0.1, 'psi_ext': np.pi}]
    kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1}]
    
    # hard bound lower limit in parameter space #
    kwargs_lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10},
        {'gamma_ext': 0., 'psi_ext': -np.pi}]
    kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10}]
    # hard bound upper limit in parameter space #
    kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10},
        {'gamma_ext': 0.3, 'psi_ext': np.pi}]
    kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10}]
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, {'ra_0': 0, 'dec_0': 0}], kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}

    kwargs_constraints = {#'joint_source_with_point_source': [[0, 0]], 
    #    'joint_source_with_point_source': list [[i_point_source, k_source], [...], ...],
    #     joint position parameter between lens model and source light model
                           #   'num_point_source_list': [4],
                              'solver_type':'NONE'# 'PROFILE_SHEAR',  # 'PROFILE', \
                          #'PROFILE_SHEAR', 'ELLIPSE', 'CENTER', 'NONE'
                              }

    
    # actual fit:
    from lenstronomy.Workflow.fitting_sequence import FittingSequence
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 400}]
                      ,
                       ['MCMC', {'n_burn': 100, 'n_run': 400, 'walkerRatio': 5, 'sigma_scale': .1}]
        ]
    kw_input = {"kwargs_data_joint":   kwargs_data_joint,
                "kwargs_model":        kwargs_model, 
                "kwargs_constraints":  kwargs_constraints, 
                "kwargs_likelihood":   kwargs_likelihood, 
                "kwargs_params":       kwargs_params,
                "fitting_kwargs_list": fitting_kwargs_list
               }
    nm_input = f"{res_dir}/kw_input.dll"
    with open(nm_input,"wb") as f:
        dill.dump(kw_input,f)
    print(f"Saving input in kw: {nm_input}")
    
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    print("kwargs_result",kwargs_result)
    nm_res = f"{res_dir}/kw_res.json"
    with open(nm_res,"w") as f:
        json.dump(kwargs_result,f)
    print(f"Saving result output in kw:{nm_res}")

    print(kwargs_result)
    # we need to extract the updated multi_band_list object since the coordinate shifts were updated in the kwargs_data portions of it
    multi_band_list_out = fitting_seq.multi_band_list
    
    modelPlot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result, 
                          arrow_size=0.02, cmap_string="gist_heat",
                          image_likelihood_mask_list=kwargs_likelihood["image_likelihood_mask_list"])

    band_index_plot = 0
    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
    
    modelPlot.data_plot(ax=axes[0,0], band_index=band_index_plot)#,v_min=,v_max=band_i.vmax)
    modelPlot.model_plot(ax=axes[0,1], band_index=band_index_plot)#,v_min=band_i.vmin,v_max=band_i.vmax)
    modelPlot.normalized_residual_plot(ax=axes[0,2], band_index=band_index_plot)#,v_min=band_i.res_vmin,v_max=band_i.res_vmax)
    modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100, band_index=band_index_plot)#,v_min=band_i.vmin,v_max=band_i.vmax)
    modelPlot.convergence_plot(ax=axes[1, 1], band_index=band_index_plot,v_max=1)
    modelPlot.magnification_plot(ax=axes[1, 2], band_index=band_index_plot)
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    nm = f'{res_dir}/mass_model.pdf'
    plt.savefig(nm)
    print(f"Saving {nm}")
    
    f, axes = plt.subplots(1,2, figsize=(8, 4), sharex=False, sharey=False)

    modelPlot.decomposition_plot(ax=axes[0], band_index=band_index_plot, text='Source light', source_add=True, unconvolved=True)#,v_min=band_i.vmin,v_max=band_i.vmax)
    modelPlot.decomposition_plot(ax=axes[1], band_index=band_index_plot, text='Source light convolved', source_add=True)#,v_min=band_i.vmin,v_max=band_i.vmax)

    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    #plt.show()
    nm = f'{res_dir}/light_model.pdf'
    plt.savefig(nm)
    print(f"Saving {nm}")
