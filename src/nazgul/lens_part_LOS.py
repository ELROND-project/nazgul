"""
Add LOS effects
""" 
import pandas as pd 

kw_los_path = "/cosma/home/do019/dc-quei1/analosis/analosis/results/datasets/golden_sample_input_kwargs.csv"
def get_kw_los(kw_los_path=kw_los_path,index=0,wrap_around=True):
    kw   = pd.read_csv(kw_los_path)
    los_cols = ['kappa_os', 'gamma1_os', 'gamma2_os', 'omega_os',
    'kappa_od', 'gamma1_od', 'gamma2_od', 'omega_od',
    'kappa_ds', 'gamma1_ds', 'gamma2_ds', 'omega_ds',
    'kappa_los', 'gamma1_los', 'gamma2_los', 'omega_los']
    los  = kw.loc[:, los_cols]
    list_los = los.to_dict('records')
    if wrap_around:
        # wrap around the list if index is > len(list_los)
        index = index%len(list_los)
    kw_los = list_los[index]
    return kw_los
