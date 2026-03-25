#%%
from BI import bi,jnp

m = bi(platform='cpu')
data_path = m.load.mastectomy(only_path = True)
m.data(data_path)
m.df.metastasized = (m.df.metastasized == "yes").astype(np.int64)
# %%
m.df.event = jnp.array(m.df.event.values, dtype=jnp.int32)
m.models.survival.surv_object(time='time', event='event', cov='metastasized', interval_length=1) # if interval = 3 we get the same as in PyMC notebook
m.data_on_model['metastasized']
# %%
