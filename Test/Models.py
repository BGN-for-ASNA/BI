from BI import bi
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

# Helper to get the absolute path to resources
RESOURCES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Resources")) + os.sep

def get_1_continuous_variable():
    m = bi(print_devices_found=False)
    data_path = m.load.howell1(only_path=True)
    m.data(data_path, sep=';')
    m.df = m.df[m.df.age > 18]
    m.scale(['weight'])
    
    def model(weight, height):
        alpha = m.dist.normal(178, 20, name='a')
        beta = m.dist.log_normal(0, 1, name='b')
        sigma = m.dist.uniform(0, 50, name='s')
        m.dist.normal(alpha + beta * weight, sigma, obs=height)
        
    m.data_to_model(['weight', 'height'])
    return m, model

def get_2_categorical_variable():
    m = bi(print_devices_found=False)
    data_path = m.load.milk(only_path=True)
    m.data(data_path, sep=';')
    m.index(["clade"])
    m.scale(['kcal_per_g'])
    
    def model(kcal_per_g, index_clade):
        a = m.dist.normal(0, 0.5, shape=(4,), name='a')
        s = m.dist.exponential(1, name='s')
        mu = a[index_clade]
        m.dist.normal(mu, s, obs=kcal_per_g)
        
    m.data_to_model(['kcal_per_g', "index_clade"])
    return m, model

def get_3_continuous_interactions():
    m = bi(print_devices_found=False)
    df = pd.read_csv(RESOURCES_PATH + 'tulips.csv', sep=';')
    df['blooms_scaled'] = df.blooms / df.blooms.max()
    df['water_scaled'] = df.water - df.water.mean()
    df['shade_scaled'] = df.shade - df.shade.mean()
    m.df = df
    
    def model(blooms_scaled, water_scaled, shade_scaled):
        sigma = m.dist.exponential(1, name='sigma')
        bws = m.dist.normal(0, 0.25, name='bws')
        bs = m.dist.normal(0, 0.25, name='bs')
        bw = m.dist.normal(0, 0.25, name='bw')
        a = m.dist.normal(0.5, 0.25, name='a')
        mu = a + bw*water_scaled + bs*shade_scaled + bws*water_scaled*shade_scaled
        m.dist.normal(mu, sigma, obs=blooms_scaled)
        
    m.data_to_model(['blooms_scaled', 'water_scaled', 'shade_scaled'])
    return m, model

def get_4_binomial():
    m = bi(print_devices_found=False)
    df = pd.read_csv(RESOURCES_PATH + 'chimpanzees.csv', sep=';')
    m.df = df
    
    def model(pulled_left):
        alpha = m.dist.normal(0, 10, name='a')
        m.dist.binomial(total_count=1, logits=alpha, obs=pulled_left)
        
    m.data_on_model = {'pulled_left': jnp.array(df.pulled_left.values)}
    return m, model

def get_5_binomial_with_indices():
    m = bi(print_devices_found=False)
    df = pd.read_csv(RESOURCES_PATH + 'chimpanzees.csv', sep=';')
    df['prosoc_left'] = df.prosoc_left.values
    df['condition'] = df.condition.values
    df['actor'] = df.actor.values - 1
    m.df = df
    
    def model(actor, prosoc_left, condition, pulled_left):
        a = m.dist.normal(0, 10, shape=(7,), name='a')
        bp = m.dist.normal(0, 10, name='bp')
        bpc = m.dist.normal(0, 10, name='bpc')
        logits = a[actor] + (bp + bpc * condition) * prosoc_left
        m.dist.binomial(total_count=1, logits=logits, obs=pulled_left)
        
    m.data_to_model(['prosoc_left', 'condition', 'actor', 'pulled_left'])
    return m, model

def get_6_poisson():
    m = bi(print_devices_found=False)
    df = pd.read_csv(RESOURCES_PATH + 'Kline.csv', sep=';')
    df['population_scaled'] = (df['population'] - df['population'].mean()) / df['population'].std()
    df['cid'] = (df['contact'] == "high").astype(int)
    m.df = df
    
    def model(cid, population_scaled, total_tools):
        a = m.dist.normal(3, 0.5, shape=(2,), name='a')
        b = m.dist.normal(0, 0.2, shape=(2,), name='b')
        lambda_ = jnp.exp(a[cid] + b[cid] * population_scaled)
        m.dist.poisson(lambda_, obs=total_tools)
        
    m.data_to_model(['cid', 'population_scaled', 'total_tools'])
    return m, model

def get_7_negative_binomial():
    m = bi(print_devices_found=False)
    np.random.seed(1)
    num_days = 300
    y_days = np.random.poisson(1.5, num_days)
    num_weeks = 40
    y_weeks = np.random.poisson(0.5 * 7, num_weeks)
    y_all = np.concatenate([y_days, y_weeks])
    exposure = np.concatenate([np.repeat(1, num_days), np.repeat(7, num_weeks)])
    monastery = np.concatenate([np.repeat(0, num_days), np.repeat(1, num_weeks)])
    df = pd.DataFrame({'y': y_all, 'days': exposure, 'monastery': monastery})
    df['log_days'] = np.log(df['days'])
    m.df = df
    
    def model(y, log_days, monastery):
        a = m.dist.normal(0, 1, name='a')
        b = m.dist.normal(0, 1, name='b')
        lambda_ = jnp.exp(log_days + a + b * monastery)
        m.dist.poisson(lambda_, obs=y)
        
    m.data_to_model(['y', 'log_days', 'monastery'])
    return m, model

def get_8_multinomial():
    m = bi(print_devices_found=False)
    N = 500
    income = np.array([1, 2, 5])
    score = 0.5 * income
    p_true = jax.nn.softmax(score)
    np.random.seed(1)
    career = np.random.choice([0, 1, 2], size=N, p=np.array(p_true))
    df = pd.DataFrame({'career': career})
    unique_income = np.array([1, 2, 5])
    m.df = df
    
    def model(career, unique_income):
        a = m.dist.normal(0, 1, shape=(2,), name='a')
        b = m.dist.half_normal(0.5, name='b')
        s1 = a[0] + b * unique_income[0]
        s2 = a[1] + b * unique_income[1]
        s3 = jnp.zeros(1)
        p = jax.nn.softmax(jnp.stack([s1, s2, s3[0]]))
        m.dist.categorical(probs=p, obs=career)
        
    m.data_on_model = {
        'career': jnp.array(df.career.values),
        'unique_income': jnp.array(unique_income).astype(jnp.int32)
    }
    return m, model

def get_9_beta_binomial():
    m = bi(print_devices_found=False)
    df = pd.read_csv(RESOURCES_PATH + 'UCBadmit.csv', sep=';')
    df["gid"] = (df["applicant.gender"] != "male").astype(int)
    m.df = df
    
    def model(gid, applications, admit):
        phi = m.dist.exponential(1.0, name='phi')
        alpha = m.dist.normal(0.0, 1.5, shape=(2,), name='alpha')
        theta = phi + 2.0
        pbar = jax.nn.sigmoid(alpha[gid])
        concentration1 = pbar * theta
        concentration0 = (1.0 - pbar) * theta
        m.dist.beta_binomial(total_count=applications, concentration1=concentration1, concentration0=concentration0, obs=admit)
        
    m.data_on_model = {
        'gid': jnp.array(df.gid.values),
        'applications': jnp.array(df.applications.values),
        'admit': jnp.array(df.admit.values)
    }
    return m, model

def get_10_zero_inflated():
    from jax.scipy.special import expit
    m = bi(print_devices_found=False)
    prob_drink = 0.2
    rate_work = 1
    N = 365
    np.random.seed(42)
    drink = np.random.binomial(1, prob_drink, N)
    y = (1 - drink) * np.random.poisson(rate_work, N)
    df = pd.DataFrame({'y': y})
    m.df = df
    
    def model(y):
        ap = m.dist.normal(-1.5, 1.0, name='ap')
        p = expit(ap)
        al = m.dist.normal(1.0, 0.5, name='al')
        lambda_ = jnp.exp(al)
        m.dist.zero_inflated_poisson(p, lambda_, obs=y)
        
    m.data_on_model = {'y': jnp.array(y)}
    return m, model

def get_11_varying_intercepts():
    m = bi(print_devices_found=False)
    df = pd.read_csv(RESOURCES_PATH + 'reedfrogs.csv', sep=';')
    df["tank"] = np.arange(len(df))
    m.df = df
    
    def model(tank, surv, density):
        sigma = m.dist.exponential(1.0, name='sigma')
        bar_alpha = m.dist.normal(0.0, 1.5, name='a_bar')
        alpha = m.dist.normal(bar_alpha, sigma, shape=(48,), name='alpha')
        p = alpha[tank]
        m.dist.binomial(total_count=density, logits=p, obs=surv)
        
    m.data_on_model = {
        'tank': jnp.array(df.tank.values),
        'surv': jnp.array(df.surv.values),
        'density': jnp.array(df.density.values)
    }
    return m, model

def get_12_varying_effects():
    m_sim = bi(print_devices_found=False)
    a, b = 3.5, -1.0
    sigma_a, sigma_b, rho = 1.0, 0.5, -0.7
    Mu = jnp.array([a, b])
    sigmas = jnp.array([sigma_a, sigma_b])
    Rho_sim = jnp.array([[1, rho], [rho, 1]])
    Sigma = jnp.diag(sigmas) @ Rho_sim @ jnp.diag(sigmas)
    N_cafes, N_visits = 20, 10
    vary_effects = m_sim.dist.multivariate_normal(Mu, Sigma, shape=(N_cafes,), sample=True)
    a_cafe_sim, b_cafe_sim = vary_effects[:, 0], vary_effects[:, 1]
    afternoon = jnp.tile(jnp.arange(2), N_visits * N_cafes // 2)
    cafe_id = jnp.repeat(jnp.arange(N_cafes), N_visits)
    mu_sim = a_cafe_sim[cafe_id] + b_cafe_sim[cafe_id] * afternoon
    sigma_sim = 0.5
    wait = m_sim.dist.normal(mu_sim, sigma_sim, sample=True)
    d = pd.DataFrame(dict(cafe=cafe_id, afternoon=afternoon, wait=wait))
    
    m = bi(print_devices_found=False)
    m.df = d
    m.data_on_model = dict(
        cafe=jnp.array(d.cafe.values, dtype=jnp.int32),
        wait=jnp.array(d.wait.values, dtype=jnp.float32),
        N_cafes=N_cafes,
        afternoon=jnp.array(d.afternoon.values, dtype=jnp.float32)
    )
    
    def model(cafe, wait, N_cafes, afternoon):
        alpha = m.dist.normal(5.0, 2.0, name='a')
        beta = m.dist.normal(-1.0, 0.5, name='b')
        sigma = m.dist.exponential(1.0, name='sigma')
        sigma_cafe = m.dist.exponential(1.0, shape=(2,), name='sigma_cafe')
        Rho = m.dist.lkj(2, 2.0, name='Rho')
        cov = jnp.outer(sigma_cafe, sigma_cafe) * Rho
        a_b_cafe = m.dist.multivariate_normal(jnp.stack([alpha, beta]), cov, shape=(N_cafes,), name='a_b_cafe')
        a_cafe, b_cafe = a_b_cafe[:, 0], a_b_cafe[:, 1]
        mu = a_cafe[cafe] + b_cafe[cafe] * afternoon
        m.dist.normal(mu, sigma, obs=wait)
        
    return m, model

def get_13_gaussian_processes():
    N_islands = 10
    Dm = jnp.array([
        [0, 0.48, 0.72, 3.59, 4.34, 4.38, 4.8, 5.25, 5.37, 5.86],
        [0.48, 0, 0.32, 3.2, 3.96, 4, 4.41, 4.88, 5.01, 5.49],
        [0.72, 0.32, 0, 2.9, 3.66, 3.7, 4.12, 4.6, 4.73, 5.23],
        [3.59, 3.2, 2.9, 0, 1, 1.13, 1.5, 1.95, 2.22, 2.71],
        [4.34, 3.96, 3.66, 1, 0, 0.22, 0.61, 1, 1.45, 1.93],
        [4.38, 4, 3.7, 1.13, 0.22, 0, 0.4, 0.9, 1.25, 1.74],
        [4.8, 4.41, 4.12, 1.5, 0.61, 0.4, 0, 0.51, 1.01, 1.49],
        [5.25, 4.88, 4.6, 1.95, 1, 0.9, 0.51, 0, 0.61, 1.1],
        [5.37, 5.01, 4.73, 2.22, 1.45, 1.25, 1.01, 0.61, 0, 0.49],
        [5.86, 5.49, 5.23, 2.71, 1.93, 1.74, 1.49, 1.1, 0.49, 0]
    ])
    f_true, a_true, b_true, g_true = 3.0, 1.0, 1.0, 0.1
    P_vals = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
    m_sim = bi(print_devices_found=False)
    K_sim = a_true**2 * jnp.exp(-b_true**2 * Dm**2) + jnp.diag(jnp.repeat(g_true**2 + 1e-2, N_islands))
    k_sim = m_sim.dist.multivariate_normal(jnp.zeros(N_islands), K_sim, sample=True)
    lambda_sim = jnp.exp(f_true + k_sim) * P_vals
    D_sim = m_sim.dist.poisson(lambda_sim, sample=True)
    
    m = bi(print_devices_found=False)
    m.df = pd.DataFrame(dict(island=jnp.arange(N_islands), P=P_vals, D=D_sim))
    m.data_on_model = dict(
        P=P_vals.astype(jnp.float32),
        D=D_sim.astype(jnp.int32),
        Dm=Dm.astype(jnp.float32),
        N=N_islands
    )
    
    def model(P, D, Dm, N):
        a = m.dist.exponential(1.0, name='a')
        b = m.dist.exponential(1.0, name='b')
        g = m.dist.exponential(1.0, name='g')
        f = m.dist.normal(3.0, 1.0, name='f')
        K = a**2 * jnp.exp(-b**2 * Dm**2) + jnp.diag(jnp.repeat(g**2 + 1e-2, N))
        k = m.dist.multivariate_normal(jnp.zeros(N), K, name='k')
        lambda_ = jnp.exp(f + k) * P
        m.dist.poisson(lambda_, obs=D)
        
    return m, model

all_models = {
    "continuous_variable": get_1_continuous_variable,
    "categorical_variable": get_2_categorical_variable,
    "continuous_interactions": get_3_continuous_interactions,
    "binomial": get_4_binomial,
    "binomial_with_indices": get_5_binomial_with_indices,
    "poisson": get_6_poisson,
    "negative_binomial": get_7_negative_binomial,
    "multinomial": get_8_multinomial,
    "beta_binomial": get_9_beta_binomial,
    "zero_inflated": get_10_zero_inflated,
    "varying_intercepts": get_11_varying_intercepts,
    "varying_effects": get_12_varying_effects,
    "gaussian_processes": get_13_gaussian_processes
}
