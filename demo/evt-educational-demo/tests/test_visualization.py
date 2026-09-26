import matplotlib
matplotlib.use("Agg")
import numpy as np
from evt_demo.visualization import plot_time_series_with_extremes

def test_plot_saved(tmp_path):
    x=np.arange(20.); mask=x>17; target=tmp_path/"plot.png"
    fig,ax=plot_time_series_with_extremes(x,mask,save_path=target)
    assert target.exists() and fig is ax.figure
