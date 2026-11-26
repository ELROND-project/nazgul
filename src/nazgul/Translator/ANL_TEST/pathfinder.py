from python_tools.tools import short_SciNot,to_uid_base64

def get_galname(kw_gal,**kwargs):
    theta_E = kw_gal["theta_E"]
    n_smpl  = short_SciNot(kw_gal["n_smpl"])
    prof    = kw_gal["profile"]
    galname =  f"TEST_GAL_{prof}_tE{theta_E}_nS{n_smpl}"
    # additional kwargs are read as hash
    kw_add= {} | kwargs
    for k in kw_gal:
        if not k in ["theta_E","profile","n_smpl"]:
            kw_add[k] = kw_gal[k]

    hash_kwg = "_"+to_uid_base64(kw_add)
    if prof == "SIS":
        return galname+hash_kwg
    elif prof == "SIE":
        galname += f"_e1{kw_gal['e1']}_e2{kw_gal['e2']}"
        return galname+hash_kwg
    else:
        raise ValueError("kw_gal['profile'] either SIS or SIE, not "+prof)
