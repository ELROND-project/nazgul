from python_tools.tools import short_SciNot
def get_galname(kw_gal,**kwargs):
    theta_E = kw_gal["theta_E"]
    n_smpl  = short_SciNot(kw_gal["n_smpl"])
    prof    = kw_gal["profile"]
    galname =  f"TEST_GAL_{prof}_tE{theta_E}_nS{n_smpl}"
    if prof == "SIS":
        return galname
    elif prof == "SIE":
        galname += f"_e1{kw_gal['e1']}_e2{kw_gal['e2']}"
        return galname
    else:
        raise ValueError("kw_gal['profile'] either SIS or SIE, not "+prof)