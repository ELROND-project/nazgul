# general library for some plotting tools
import matplotlib
base_colors = ["red","green","blue","yellow","black","magenta","cyan",
               "darkorange","darkviolet","lawngreen","violet"] 

warm_colors  = ['#fdcc8a', '#fc8d59', '#d7301f']

matplotlib.use('Agg') 

# Example of chainconsumer use to plot 2D posterior
"""
import pandas as pd
from chainconsumer import Chain, ChainConsumer
    c = ChainConsumer()
    prms_nms = ["Core",r"$\gamma$"]
    c.add_chain(Chain(
        samples=pd.DataFrame(np.array([cores,gammas]).T,columns=prms_nms),
        name="",
        shade=True,
        color=warm[0],
        shade_gradient = 0.8, linewidth=3.0) )
    fig = c.plotter.plot(columns=prms_nms)
    nm_fig = f"{res_dir}/distr_coreVsGamma_1Ddens.png"
    plt.savefig(nm_fig)
    print(f"Saving {nm_fig}")
    plt.close(fig)
"""
import numpy as np
# to plot a circle - basic but always useful
def circle(x0,y0,r,n_points=50):
    x,y = [],[]
    for phi in np.linspace(0,2*np.pi,n_points):
        x.append(x0 + r*np.cos(phi))
        y.append(y0 + r*np.sin(phi))
    return x,y