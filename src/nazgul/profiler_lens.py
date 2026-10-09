import warnings
from copy import copy,deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from chainconsumer import Chain ,ChainConsumer,Truth
from chainconsumer.plotting import plot_contour,plot_dist #,plot_truths

from python_tools.tools import mkdir
from python_tools.get_res import load_whatever 
from nazgul.ultimate_loader import ultimate_gal_loader,ultimate_gallens_loader,ultimate_lenssys_loader,_get_gallens_path

from nazgul.Translator import std_kw_sim
from nazgul.lib_plot import circle
from nazgul.lib_stat import compute_tension,tension_multidim
from nazgul.isocont_stat import get_DPA, get_xi_tE
from nazgul.isocont_stat_plots import _pretty_iso_name,fit_isofit,plot_fit_iso,_plot_fit_iso
from nazgul.combined_modelling_results import get_lensfuel,get_full_chain,get_partial_chain,prettify_prm,get_emcee
from nazgul.Modelling.lib_models import get_red_chi2,get_model_plot,plot_modelplot_massmodel
from nazgul.Modelling.pathfinder import get_res_dir
from nazgul.pathfinder import results_dir

profile_dir = results_dir/"profiles"
mkdir(profile_dir)


class ProfileLens:
    """
    A class that will contain all the other
    a "ring of power" class
    it will have all the results
    """
    def __init__(self,
                 name,
                 snap,
                 model,
                 prj = None,
                 kw_sim=std_kw_sim,
                 kwargs_lenssystem={},
                 check_if_workin_on_it=True,
                 verbose=False):
        
        self.model  = model
        self.snap   = snap
        self.kw_sim = kw_sim
        self.gal    = ultimate_gal_loader(name,
                        snap=snap,
                        kw_sim=kw_sim,
                        verbose=verbose)
        self.lens = ultimate_gallens_loader(name=name,
                        snap=snap,
                        prj=prj,
                        kw_sim=kw_sim,
                        verbose=verbose)
        
        self.lensfuel = get_lensfuel(model=model)
        
        self.lens_system = ultimate_lenssys_loader(name=name,
                                                   snap=snap,
                                                   lensFuel=self.lensfuel,
                                                   prj=prj,
                                                   kwargs_lenssystem=kwargs_lenssystem,
                                                   kw_sim=kw_sim,
                                                   verbose=verbose)
        kw_truths = load_whatever(self.lens_system.path_kw_truths)
        # convert truths in something comparable to the model parameters
        self.lens_system.kw_truths = _standardise_truths(kw_truths)
        #self.lens_system.kw_ugly_truths = _standardise_truths(kw_truths,pretty=False)
        try:
            model_res_dir = _get_gallens_path(name,model=model,snaps=[snap],
                                              check_if_workin_on_it=check_if_workin_on_it)
        except RuntimeError:
            model_res_dir = None   
            warnings.warn("No lens model results found")
        self.lens_system.model_res_dir = model_res_dir
        ### where to store it
        self.savedir = get_res_dir(profile_dir,run_type=0,**kw_sim)/self.name
        mkdir(self.savedir)
    
    def __getattr__(self,name):
        # lazily import all attributs from lens_system
        return getattr(self.lens_system,name)

    def _get_path_kwisores(self,isotype="psi"):
        assert isotype in ["psi","dens"]
        nm_file = f"kw_res_iso{isotype}_prj{self.lens.proj_index}.dll"
        kw_iso_res_list = [g for g in self.lens.pkl_path.parent.glob(nm_file)]
        if len(kw_iso_res_list)==0:
            warnings.warn("No results found for kw fit")
            return None
        if len(kw_iso_res_list)>1:
            raise RuntimeError("Found multiple results for kw fit")
        path_kw_iso_res = kw_iso_res_list[0]
        return path_kw_iso_res
        
    @property
    def name(self):
        lens_name = self.lens.name.split('Lens_')[1]
        name = f"Prof_Mod_{self.model}_{lens_name}"
        return name
    
    def _get_full_chain(self):
        if self.lens_system.model_res_dir is None:
            return None
        full_chain = get_full_chain(lens=self.lens_system,
                                    model=self.model,
                                    do_prettify_prm_nm=True)
        return full_chain

    def _get_Chain(self,full_chain):
        if full_chain is None:
            return None
        chain = Chain(samples=full_chain, name=self.name, 
              shade=True, color='#2c7fb8', smooth=1, bins=30,
              shade_gradient = 0.4, linewidth=3.0)
        return chain
        
    def _get_emcee(self):
        if self.model_res_dir is None:
            return None
        #emcee_path = [pth for pth in self.model_res_dir.glob("emcee*")]
        #assert len(emcee_path)==1
        #emcee_file = emcee_path[0]
        emcee_data = get_emcee(self.lens_system,self.model,kw_sim = self.kw_sim)
        return emcee_data

    @property
    def params_orig(self):
        # warning: shear params names have to be changed to "gamma1_los_lens1","gamma2_los_lens1
        emcee = self._get_emcee()
        return emcee[-2]
        
    def get_full_chain(self,reload=False):
        if not hasattr(self,"full_chain") or reload:
            self.full_chain = self._get_full_chain()
        return self.full_chain

    def get_Chain(self,full_chain=None,reload=False):
        if hasattr(self,"Chain") and not reload:
            return self.Chain
        if full_chain is None:
            full_chain = self.get_full_chain(reload=reload)
        self.Chain = self._get_Chain(full_chain)
        return self.Chain
        
    def show_corner_post(self,full_chain=None,reload=False):
        if full_chain is None:
            full_chain = self.get_full_chain(reload=reload)
        chain = self.get_Chain(full_chain)
        if chain is None:
            warnings.warn("No lens model results found")
            return 0
        c = ChainConsumer()
        c.add_chain(chain)
        fig = c.plotter.plot()
        #fig.savefig("tmp/del.pdf")
        return fig

    def show_single_post(self,param,true=None,full_chain=None,reload=False):
        param = prettify_prm(param)
        if full_chain is None:
            full_chain = self.get_full_chain(reload=reload)
        chain = self.get_Chain(full_chain)
        if chain is None:
            warnings.warn("No lens model results found")
            return 0
        fig,ax = plt.subplots()
        plot_dist(ax,chain, px=param)
        ax.set_xlabel(param)
        true = False
        for prm in self.lens_system.kw_truths.keys():
            if param==prm:
                # we add the "true" value
                true=True
                ax.axvline(self.lens_system.kw_truths[prm],ls="--",c="r",label=param+"$_{\rm{Truth}$")
        if true:
            ax.legend()
        #if kw_truths is not None:
        #    kw_pretty_truths = _prettify_truth(kw_truths)
        #    true = kw_pretty_truths.get(param)
        #if true is not None:
        #    ax.axvline(true,ls="--",c="k",label=r"True "+param)
        return fig
        
    def show_contour_post_2(self,param1,param2,full_chain=None,reload=False):
        """
        Special case for only 2 params as it's faster
        """
        param1,param2 = prettify_prm([param1,param2])
        if full_chain is None:
            full_chain = self.get_full_chain(reload=reload)
        chain = self.get_Chain(full_chain,reload=reload)
        if chain is None:
            warnings.warn("No lens model results found")
            return 0
        fig,ax = plt.subplots(2,2)#,sharex=True,sharey=True)
        # Remove vertical space between Axes
        fig.subplots_adjust(hspace=0,wspace=0)
        plot_contour(ax[1][0], chain, px=param1, py=param2)
    
        plot_dist(ax[0][0],chain, px=param1)
        
        plot_dist(ax[1][1],chain, px=param2)
        
        true1 = self.lens_system.kw_truths.get(param1)
        true2 = self.lens_system.kw_truths.get(param2)
        if true1 is not None:
            ax[0][0].axvline(true1,ls="--",c="k",label=param1+r"$_{\rm{,\,Truth}}$="+str(np.round(true1,2)))
            ax[1][0].axvline(true1,ls="--",c="k")
            ax[0][0].legend()
        if true2 is not None:
            ax[1][1].axvline(true2,ls="--",c="k",label=param2+r"$_{\rm{,\,Truth}}$="+str(np.round(true2,2)))
            ax[1][0].axhline(true2,ls="--",c="k")
            ax[1][1].legend()
        lim_p1 = ax[1][0].get_xlim()
        lim_p2 = ax[1][0].get_ylim()
        ax[0][0].set_xlim(*lim_p1)
        ax[1][1].set_xlim(*lim_p2)

        ax[1][1].set_yticks([])
        ax[0][0].set_title(param1)
        ax[0][0].set_xticks([])
        ax[1][0].set_ylabel(param2)
        ax[1][0].set_xlabel(param1)
        ax[1][1].set_xlabel(param2)
        ax[1][1].set_title(param2)
        ax[0][1].remove()
        return fig

    def get_partial_chain(self,param_list):
        if self.lens_system.model_res_dir is None:
            return None
        partial_chain = get_partial_chain(lens=self.lens_system,
                                    model=self.model,
                                    wanted_param_list = param_list,
                                    do_prettify_prm_nm=True)
        return partial_chain
        
    def show_contour_post(self,param_list,partial_chain=None,reload=False):
        if len(param_list)==2:
            return self.show_contour_post_2(param_list[0],param_list[1],reload=reload)
        if partial_chain is None:
            partial_chain = self.get_partial_chain(param_list)
        chain = self._get_Chain(partial_chain)
        if chain is None:
            warnings.warn("No lens model results found")
            return 0
        c = ChainConsumer()
        c.add_chain(chain)
        limited_truth = {}
        for kw in chain.data_columns:
            if kw in self.lens_system.kw_truths.keys():
                limited_truth[kw] = float(self.lens_system.kw_truths[kw])
        c.add_truth(Truth(location=limited_truth))
        # Verify Truths are within range of plotting
        extents = {}
        for kw in limited_truth.keys():
            _truth = limited_truth[kw]
            _arr = chain.data_samples[kw].array
            min_arr = _arr.min()
            max_arr = _arr.max()
            Darr = max_arr-min_arr
            padd = Darr/10
            if _truth<min_arr:
                extents[kw] = (_truth-padd,max_arr)
            if _truth>min_arr:
                extents[kw] = (min_arr,_truth+padd)         
        c.plotter.config.extents=extents
        fig = c.plotter.plot()
        #fig.savefig("tmp/del.pdf")
        return fig
        
    @property
    def modelPlot(self):
        res_dir = self.lens_system.model_res_dir
        if res_dir is None:
            return None
        return get_model_plot(res_dir=res_dir)
        
    def show_simplot(self):
        modelPlot = self.modelPlot
        if modelPlot is None:
            warnings.warn("No lens model results found")
            return 0
        model_band = modelPlot._band_plot_list[0]
        kw_modelplot = {#"vmin":-5,
                        #"vmax":-1,
                        "extent":model_band._image_extent,
                        "origin":"lower"} #"cmap"="hot"
        # Sim Image
        fig,ax = plt.subplots()
        ax.set_ylabel(self.name)
        
        ax.set_title(f"Sim image")
        ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        cmap = matplotlib.cm.gist_heat
        cmap.set_bad('black',1.)
        im0 = ax.imshow(np.log10(model_band._data), 
                        cmap=cmap,**kw_modelplot)
        
        #im0 = ax.imshow(img,)
        fig.colorbar(im0,  orientation='vertical',label=r"flux$_{\rm{data}}$")
        return fig
        
    def show_iso_comparison(self,do_thin_out=False):
        kw_isopsi_pth = self._get_path_kwisores("psi")
        if kw_isopsi_pth is None:
            return None
        kw_isopsi = load_whatever(kw_isopsi_pth)
        isolist = kw_isopsi["isofit"]["isolist"]
        
        # we skip the very first isophote
        isolist.__dict__["_list"] = isolist.__dict__["_list"][1:]
        lens   = self.lens
        RE_pix = lens.thetaE.value/lens.deltaPix.value
        sma_RE = isolist.sma/RE_pix
        i_01tE = np.argmin(np.abs(sma_RE-0.1))
        i_tE   = np.argmin(np.abs(sma_RE-1))

        fig,axis = plt.subplots(5,2,figsize=(7,15))
        fig.suptitle(lens.name.replace("Sub_Lens_",""))
        RE    = lens.thetaE.value/lens.arcXkpc.value #kpc
        q     = 1-isolist.eps
        dPA   = get_DPA(isolist,i_tE)
        xi_RE = get_xi_tE(isolist,RE_pix)
        ln_dt  = xi_RE.shape[0]    
        thin_index = np.arange(0, ln_dt)
        if do_thin_out:
            thin_out_sample = 30
            thin_index = np.arange(0, ln_dt, int(np.round(ln_dt/thin_out_sample)) )
            
        dx0_RE = (isolist.x0-isolist.x0[0])/RE_pix
        dy0_RE = (isolist.y0-isolist.y0[0])/RE_pix
        ax = axis[1][0]
        ax.scatter(xi_RE,isolist.a3,marker=".",c="k",label=r"$\psi$")
        ax.set_title("a3")
        
        ax = axis[1][1]
        ax.scatter(xi_RE,isolist.b3,marker=".",c="k",label=r"$\psi$")
        ax.set_title("b3")
        
        ax = axis[2][0]
        ax.scatter(xi_RE,isolist.a4,marker=".",c="k",label=r"$\psi$")
        ax.set_title("a4")
        
        ax = axis[2][1]
        ax.scatter(xi_RE,isolist.b4,marker=".",c="k",label=r"$\psi$")
        ax.set_title("b4")

        ax = axis[3][0]
        ax.scatter(xi_RE,(isolist.x0-isolist.x0[0])/RE_pix,marker=".",c="k",label=r"$\psi$")
        ax.set_title(r"(x-x$_0$)/R$_{\rm{E}}$")
        
        ax = axis[3][1]
        ax.scatter(xi_RE,(isolist.y0-isolist.y0[0])/RE_pix,marker=".",c="k",label=r"$\psi$")
        ax.set_title(r"(y-y$_0$)/R$_{\rm{E}}$")

        ax = axis[0][0]
        ax.scatter(xi_RE,np.log(q),marker=".",c="k",label=r"$\psi$")
        ax.set_title("ln(q)")
        
        ax = axis[4][0]
        ax.scatter(xi_RE,dPA,marker=".",c="k",label=r"$\psi$")
        ax.set_title(r"P.A. - P.A.($\theta_E$)")
        
        kw_isodens_path   = self._get_path_kwisores("dens")
        if kw_isodens_path is not None:
            kw_isodens   = load_whatever(kw_isodens_path)
            isolist_dens = kw_isodens["isofit"]["isolist"]
            isolist_dens.__dict__["_list"] = isolist_dens.__dict__["_list"][1:]

            kappa_res    = kw_isodens["isofit"]["map"]-kw_isodens["isofit"]["model"]
            vm = 0.15
            ax = axis[0][1]
            im = ax.imshow(kappa_res,cmap="bwr",vmin=-vm,vmax=+vm)
            ax.scatter(*circle(isolist_dens.x0[0],isolist_dens.y0[0],RE_pix,n_points=10),c="k",marker="x",label=r"$\theta_E$")
            ax.set_title(r"$\kappa_{map}-\kappa_{model}$")
            fig.colorbar(im)
            q_dens     = 1-isolist_dens.eps
            xi_dens_RE = get_xi_tE(isolist_dens,RE_pix)
            ###
            ax = axis[1][0]
            ax.scatter(xi_dens_RE,isolist_dens.a3,marker=".",c="b",label=r"$\kappa$")
            ax = axis[1][1]
            ax.scatter(xi_dens_RE,isolist_dens.b3,marker=".",c="b",label=r"$\kappa$")
            ax = axis[2][0]
            ax.scatter(xi_dens_RE,isolist_dens.a4,marker=".",c="b",label=r"$\kappa$")
            ax = axis[2][1]
            ax.scatter(xi_dens_RE,isolist_dens.b4,marker=".",c="b",label=r"$\kappa$")
            ax = axis[0][0]
            ax.scatter(xi_dens_RE,np.log(q_dens),marker=".",c="b",label=r"$\kappa$")
            
            ax = axis[3][0]
            ax.scatter(xi_dens_RE,(isolist_dens.x0-isolist_dens.x0[0])/RE_pix,marker=".",c="b",label=r"$\kappa$")
            ax = axis[3][1]
            ax.scatter(xi_dens_RE,(isolist_dens.y0-isolist_dens.y0[0])/RE_pix,marker=".",c="b",label=r"$\kappa$")

            ax = axis[4][0]
            i_tE_dens = np.argmin(np.abs(xi_dens_RE-1))
            Dpa_dens = get_DPA(isolist_dens,i_tE_dens)
            ax.scatter(xi_dens_RE,Dpa_dens,marker=".",c="b",label=r"$\kappa$")
            ax.set_title(r"P.A. - P.A.($\theta_E$)")

        # limit the a3,b3,a4,b4
        for axi in axis[1],axis[2]:
            for ax in axi:
                min_y,max_y = ax.get_ylim()
                if min_y<-.3:
                    min_y=-.3
                if max_y>.3:
                    max_y=.3
                ax.set_ylim(min_y,max_y)
        for i,axi in enumerate(axis):
            for j,axii in enumerate(axi):
                #_,max_x = axii.get_xlim()
                #axii.set_xlim(0.1,max_x)
                if i==0 and j==1: # the image
                    axii.set_xticks([])
                    axii.set_yticks([])
                elif i==4 and j==1: # the bottom corner image
                    #axii = self.show_fit_isofit("psi").get_axes()[0]
                    kw_res_fit = self.get_fit_isofit("psi")
                    axii = _plot_fit_iso(kw_res_fit,axii)
                else:
                    axii.set_xlabel(r"$\xi/\theta_E$ []")
                    lbl = None
                    if i==0 and j==0:
                        lbl = r"$\xi/\theta_E$<0.1"
                    orig_lmy = np.array(axii.get_ylim())
                    lmy     = copy(orig_lmy)
                    dlmy   =  np.abs(np.diff(orig_lmy))*.3
                    lmy[0] -= dlmy
                    lmy[1] += dlmy
                    axii.fill_betweenx(np.arange(*lmy,0.01),0.1,facecolor="k",alpha=.4,label=lbl)
                    axii.set_ylim(*orig_lmy)
                axii.legend()
        for ax in [axis[4][0]]:
            min_y,max_y = ax.get_ylim()
            if min_y<-20:
                min_y=-20
            if max_y>20:
                max_y=20
            ax.set_ylim(min_y,max_y)
        #axis[-1][-1].remove()
        fig.tight_layout()
        return fig

    def get_red_chi2(self):
        if self.modelPlot is None:
            warnings.warn("No lens model results found")
            return 0
        reduced_chi2  = get_red_chi2(self.modelPlot,verbose=False)
        return reduced_chi2
    
    def show_lens_model_plot(self):
        modelPlot = self.modelPlot
        if modelPlot is None:
            warnings.warn("No lens model results found")
            return 0
        band_index_plot = 0
        f = plot_modelplot_massmodel(modelPlot=modelPlot,band_index_plot=band_index_plot)
        return f

    def show_isocont(self,isotype="psi",n_isoc=None): 
        path_kw_iso_res = self._get_path_kwisores(isotype=isotype)
        if path_kw_iso_res is None:
            return None
        kw_iso  = load_whatever(path_kw_iso_res)
        isolist = kw_iso["isofit"]["isolist"]
        isolist.__dict__["_list"] = isolist.__dict__["_list"][1:]
        iso_map = kw_iso["isofit"]["map"]

        pretty_isoname = _pretty_iso_name(isotype)

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.suptitle("Iso-contour of "+pretty_isoname)
        im0 = ax.imshow(iso_map,origin="lower",cmap="gist_heat")
        isos = []
        if n_isoc is None:
            frq_isoc = 1 
        else:
            frq_isoc = int(np.round(len(isolist.sma)/n_isoc))
        smas = isolist.sma[::frq_isoc]
        for sma in smas:
            iso = isolist.get_closest(sma)
            isos.append(iso)
            x, y, = iso.sampled_coordinates()
            plt.plot(x, y, color='white')
        iso = isos[0]
        x0,y0 = iso.x0,iso.y0
        RE_pix = self.lens.thetaE.value/self.lens.deltaPix.value
        plt.scatter(*circle(x0,y0,r=RE_pix),c="w",marker="x",label=r"$\theta_E$")
        #plt.xlim(x0-50,x0+50)
        #plt.ylim(y0-50,y0+50)
        fig.colorbar(im0,orientation='vertical',label=pretty_isoname, fraction=0.046, pad=0.04)
        ax.legend()
        return fig

    def get_fit_isofit(self,isotype="psi"):
        kw_res_isofit =  fit_isofit(self.lens,isotype=isotype)
        return kw_res_isofit
        
    def show_fit_isofit(self,isotype="psi"):
        return plot_fit_iso(self.lens,isotype=isotype)
        
    def tension_computer(self,param,true_value=None):
        if true_value is None:
            true_value = self.lens_system.kw_truths[prettify_prm(param)]
        prm_mcmc = self.get_lensmodel_param(param)
        val = prm_mcmc.mean()
        sig = prm_mcmc.std()
        return compute_tension([val,sig],true_value)

    def tension_computer_multi(self,param_list,true_value_list=None):
        if true_value_list is None:
            true_value_list = [self.lens_system.kw_truths[prettify_prm(p)] for p in param_list]

        partial_chain = self.get_partial_chain(param_list)
        distr = np.asarray([partial_chain[prettify_prm(p)].array for p in param_list])
        return tension_multidim(distr,true_value_list)
    

    def get_lensmodel_param(self,param,full_chain=None,reload=False):
        if full_chain is None:
            full_chain = self.get_full_chain(reload=reload)
        prm_mcmc = full_chain[prettify_prm(param)].values
        return prm_mcmc

    
    def get_isofit_param(self,param,isotype="psi"):
        kw_iso_res = self.get_fit_isofit(isotype=isotype)
        if param=="theta_core":
             param_val = kw_iso_res["kw_popt"]["x_core"]*self.lens.thetaE.value
        elif param=="sharpness":
             param_val = kw_iso_res["kw_popt"]["s"]
        else:
             param_val = kw_iso_res["kw_popt"][param]
        return param_val

def _prettify_truth(kw_truths):
    kw_pretty_truths = {}
    for kw in kw_truths:
        kw_pretty_truths[prettify_prm(kw)] = kw_truths[kw]
    return kw_pretty_truths

def _standardise_truths(kw_truths,pretty=True):
    #Very ad-hoc solution
    kw_stnd_truths = {}
    for k in kw_truths:
        if k=="theta_E_lens0":
            kw_stnd_truths[k] = kw_truths[k]
        elif k=="kwargs_source":
            for kk in kw_truths[k]:
                kw_stnd_truths[kk+"_source0"] = kw_truths[k][kk]
        elif k=="kwargs_add_lenses":
            kw_lenses = kw_truths[k]["kwargs_lens"]
            for i,kl in enumerate(kw_lenses):
                for kk in kl:
                    kw_stnd_truths[kk+"_lens"+str(i+1)] = kl[kk]
        else:
            raise RuntimeError("this truth was not standardised")
    if pretty:
        kw_stnd_truths = _prettify_truth(kw_stnd_truths)
    return kw_stnd_truths
            