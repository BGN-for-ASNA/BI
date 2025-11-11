# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
from BI import bi, jnp

m=bi()
m.data('iris.csv', sep=',') # Data is already scaled
m.data_on_model = dict(
    X=jnp.array(m.df.iloc[:,0:-2].values)
)
m.fit(m.models.pca(type="classic"), progress_bar=False) # or robust, sparse, classic, sparse_robust_ard

m.models.pca.plot(
    X=m.df.iloc[:,0:-2].values,
    y=m.df.iloc[:,-2].values, 
    feature_names=m.df.columns[0:-2], 
    target_names=m.df.iloc[:,-1].unique(),
    color_var=m.df.iloc[:,0].values,
    shape_var=m.df.iloc[:,-2].values
)
#
#
#
#
#
from BI import bi, jnp

m = bi()
m.data('mastectomy.csv', sep=',').head()
m.df.metastasized = (m.df.metastasized.values == "yes").astype(jnp.int64)

# Import time-steps and events
m.models.survival.import_time_even(
    m.df.time.values, 
    m.df.event.values, interval_length=3
)

# To import time-fixed covariates
m.models.survival.import_covF(
    m.df.metastasized.values, ['metastasized']
) 

# To import time-varying covariates ⚠️ Experimental feature
# m.models.survival.import_covV 

m.fit(m.models.survival.model, num_samples=500) 

m.summary()

m.models.survival.plot_surv( beta = 'Hazard_rate_metastasized')
#
#
#
#
#
#
from BI import bi
from sklearn.datasets import make_blobs
m = bi()

# Generate synthetic data
data, true_labels = make_blobs(
    n_samples=500, centers=8, cluster_std=0.8,
    center_box=(-10,10), random_state=101
)

m.data_on_model = {"data": data,"K": 8 }
m.fit(m.models.gmm) # Optimize model parameters through MCMC sampling
m.plot(X=data,sampler=m.sampler) # Prebuild plot function for GMM ⚠️ Experimental feature
#
#
#
#
#
#
#
#
#
from BI import bi
from sklearn.datasets import make_blobs
m = bi()

# Generate synthetic data
data, true_labels = make_blobs(
    n_samples=500, centers=8, cluster_std=0.8,
    center_box=(-10,10), random_state=101
)
m.data_on_model = dict(data=data,T=10)
m.fit(m.models.dpmm)
m.plot(data,m.sampler)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
