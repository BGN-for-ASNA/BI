# Prior predictive checks
#%%
import arviz as az
import seaborn as sns
from code.model_write import *
from code.model_fit import *

def model_output_to_df(sample):
    l = {}
    for key in sample.keys():
        l[key] = list(tf.squeeze(sample[key]).numpy())
    post_df = pd.DataFrame.from_dict(l)
    return post_df

def plot_prior_dist(model, N = 100):
    samples = model.sample(N)
    prob = model.log_prob(samples)
    post_df = model_output_to_df(samples)
    
    fig, axs = plt.subplots(ncols=post_df.shape[1])
    for a in range(post_df.shape[1]-1): 
            sns.histplot(post_df.iloc[:,a], 
                 kde=True, stat="density",
                 edgecolor=(1, 1, 1, .4), 
                 ax=axs[a]).set_title(post_df.columns[a]) 
    
    sns.histplot(list(prob.numpy()), kde=True, stat="density",
                 edgecolor=(1, 1, 1, .4)).set_title("${\\rm logit}$")
    return fig

def model_check(posterior, trace, sample_stats, params):
    posterior, axes = plt.subplots(1, len(params), figsize=(8, 4))
    axes = az.plot_posterior(trace, var_names=params, ax=axes)
    axes.flatten()[0].get_figure() 
    
    autocor = az.plot_autocorr(trace, var_names=params)
    
    traces = az.plot_trace(trace, compact=False)
    
    rank, axes = plt.subplots(1, len(params))
    az.plot_rank(trace, var_names=params, ax=axes)
    
    forest = az.plot_forest(trace, var_names = params)

    summary = az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)
    
    return posterior, autocor, traces, rank, forest, summary

#%%


#%%
#plot_prior_dist(model)
#samples = model.sample(**posterior)
#model_check(posterior, trace, sample_stats, params = ['s', 'alpha', 'beta'])   

##%%
#plt.plot(samples['y'][1].numpy(), weight)
#
## %%
#sample_alpha = tf.squeeze(posterior["alpha"][0])
#sample_beta = tf.squeeze(posterior["beta"][0])
#sample_sigma = tf.squeeze(posterior["sigma"][0])
#
#samples_flat = tf.stack([sample_alpha, sample_beta, sample_sigma], axis=0)
## %%
#sns.scatterplot(x = weight, y = observed_data)
#
## %%
#weight = d.weight - d.weight.mean()# weight can be replace by any value
#plt.scatter(d.weight,observed_data, label="real data")
#plt.scatter(d.weight,np.array(model.sample(**posterior)['y'].numpy()).mean(axis=0), label="estimation")
#
#plt.legend(loc="upper left")
## %%

# %%
