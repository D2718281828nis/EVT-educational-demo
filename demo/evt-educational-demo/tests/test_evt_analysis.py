import numpy as np
import pytest
from evt_demo.evt_analysis import EVTAnalyzer, GEVFit

def test_fits_and_detection():
    x=np.random.default_rng(3).standard_t(5,4000); analyzer=EVTAnalyzer()
    gev=analyzer.fit_gev_block_maxima(x,100); assert len(gev.block_maxima)==40 and gev.dropped_points==0 and gev.sigma>0
    gpd=analyzer.fit_gpd_peaks_over_threshold(x,threshold_percentile=95); assert len(gpd.exceedances)==200 and np.allclose(gpd.exceedances,x[gpd.indices]-gpd.threshold)
    mask=analyzer.detect_extremes_via_pot(x,gpd,.1); assert mask.dtype==bool and mask.shape==x.shape and np.all(x[mask]>gpd.threshold)
    diag=analyzer.diagnostics(gpd); assert len(diag["empirical_quantiles"])==200

def test_return_levels_gumbel():
    fit=GEVFit(1.,2.,0.,np.linspace(0,5,30),np.empty((30,2),int),0)
    result=EVTAnalyzer().calculate_gev_return_levels(fit,[10,50],n_bootstrap=20,seed=1)
    expected=1-2*np.log(-np.log1p(-1/np.array([10.,50.])))
    assert np.allclose(result.return_level,expected)

def test_too_few_exceedances():
    with pytest.raises(ValueError,match="20"):
        EVTAnalyzer().fit_gpd_peaks_over_threshold(np.arange(100),threshold=95)
