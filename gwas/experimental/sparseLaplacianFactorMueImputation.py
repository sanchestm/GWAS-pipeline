# --- Import Libraries ---
import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax import random, jit, value_and_grad
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

# --- Simulate Genomic Positions and Hotspot Map ---
def simulate_recombination_map(n_variants, n_hotspots=5, hotspot_strength=10.0, seed=42):
    np.random.seed(seed)
    pos = np.sort(np.random.randint(0, 50_000_000, n_variants))  # 50Mb region
    hotspot_pos = np.random.choice(pos, size=n_hotspots, replace=False)
    theta = np.ones(n_variants)
    for h in hotspot_pos:
        mask = np.abs(pos - h) < 500_000  # 500kb around hotspot
        theta[mask] *= hotspot_strength
    return pos, theta

# --- Simulate Genotype-like Counts ---
def simulate_zip_data(n_variants, n_samples, n_factors, pos, theta, seed=0):
    key = random.PRNGKey(seed)
    key_w, key_z, key_noise = random.split(key, 3)
    
    W_true = random.normal(key_w, (n_variants, n_factors)) * 0.5
    Z_true = random.normal(key_z, (n_factors, n_samples)) * 0.5
    mu_true = random.normal(key_noise, (n_variants,)) * 1.0
    
    lam = jnp.exp(W_true @ Z_true + mu_true[:, None])
    key_zip = random.split(key, 1)[0]
    counts = random.poisson(key_zip, lam)
    
    zi_prob = jax.nn.sigmoid(-3.0 + 0.1 * theta)
    key_mask = random.PRNGKey(seed + 1)
    mask = random.bernoulli(key_mask, zi_prob[:, None], shape=counts.shape)
    counts = counts * (1 - mask)
    
    return counts

# --- Vectorized Sparse Graph Construction ---
def vectorized_sparse_graph(pos, theta, gamma=10.0, window=2_000_000):
    n = pos.shape[0]
    pos = jnp.array(pos)
    diffs = jnp.abs(pos[:, None] - pos[None, :])
    triu_mask = jnp.triu(diffs, k=1)
    valid_mask = (triu_mask > 0) & (triu_mask <= window)
    row_idx, col_idx = jnp.where(valid_mask)
    dists = diffs[row_idx, col_idx]

    r = 0.5 * (1.0 - jnp.exp(dists * (-2e-6)))
    theta_i = theta[row_idx]
    theta_j = theta[col_idx]
    r_ij = 0.5 * (theta_i + theta_j) * r

    w_ij = jnp.exp(-gamma * r_ij)

    rows = jnp.concatenate([row_idx, col_idx])
    cols = jnp.concatenate([col_idx, row_idx])
    vals = jnp.concatenate([w_ij, w_ij])

    return np.array(rows), np.array(cols), np.array(vals), np.array(dists)

# --- KL Divergence ---
def kl_normal(mu, logvar):
    return 0.5 * jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar)

# --- ZIP Log-Likelihood ---
def zip_log_prob(x, lam, pi):
    eps = 1e-8
    zero_mask = (x == 0)
    log_prob_zero = jnp.log(pi + (1 - pi) * jnp.exp(-lam) + eps)
    log_prob_nonzero = jnp.log(1 - pi + eps) - lam + x * jnp.log(lam + eps)
    return jnp.where(zero_mask, log_prob_zero, log_prob_nonzero)

# --- Laplacian Penalty ---
def laplacian_penalty_sparse(W_factor, row_idx, col_idx, L_vals):
    dot_prods = jnp.sum(W_factor[row_idx] * W_factor[col_idx], axis=1)
    penalty = jnp.sum(L_vals * dot_prods)
    return penalty

# --- Variational Sampling ---
def sample_z(key, mu_z, logvar_z):
    eps = random.normal(key, mu_z.shape)
    return mu_z + jnp.exp(0.5 * logvar_z) * eps

def sample_theta(key, mu_theta, logvar_theta):
    eps = random.normal(key, mu_theta.shape)
    log_theta = mu_theta + jnp.exp(0.5 * logvar_theta) * eps
    return jnp.exp(log_theta)

# --- Batched ELBO Loss Function ---
def batched_elbo_loss(params, var_params, X, batch_idx, row_idx, col_idx, dists, gamma, alpha, key):
    W, mu, pi_logit = params
    mu_z, logvar_z, mu_theta, logvar_theta = var_params

    z_key, theta_key = random.split(key)
    z = sample_z(z_key, mu_z[:, batch_idx], logvar_z[:, batch_idx])
    theta = sample_theta(theta_key, mu_theta, logvar_theta)

    lam = jnp.exp(W @ z + mu[:, None])
    pi = jax.nn.sigmoid(pi_logit)[:, None]
    log_likelihood = jnp.sum(zip_log_prob(X[:, batch_idx], lam, pi))

    kl_z = kl_normal(mu_z[:, batch_idx], logvar_z[:, batch_idx])
    kl_t = kl_normal(mu_theta, logvar_theta)

    r = 0.5 * (1.0 - jnp.exp(dists * (-2e-6)))
    r_ij = 0.5 * (theta[row_idx] + theta[col_idx]) * r
    w_ij = jnp.exp(-gamma * r_ij)
    smooth_penalty = laplacian_penalty_sparse(W, row_idx, col_idx, w_ij)

    return -(log_likelihood - (kl_z + kl_t) - alpha * smooth_penalty)

# --- Parameter Initialization ---
def init_params(key, n_variants, n_samples, n_factors):
    key_w, key_mu, key_pi = random.split(key, 3)
    W = random.normal(key_w, (n_variants, n_factors)) * 0.01
    mu = jnp.zeros((n_variants,))
    pi_logit = jnp.zeros((n_variants,))
    return W, mu, pi_logit

def init_variational_params(key, n_samples, n_factors, n_variants):
    key_mu_z, key_logvar_z, key_mu_theta, key_logvar_theta = random.split(key, 4)
    mu_z = random.normal(key_mu_z, (n_factors, n_samples)) * 0.01
    logvar_z = random.normal(key_logvar_z, (n_factors, n_samples)) - 5.0
    mu_theta = jnp.zeros((n_variants,))
    logvar_theta = jnp.zeros((n_variants,)) - 5.0
    return mu_z, logvar_z, mu_theta, logvar_theta

# --- Batch Update Step ---
@jit
def update_batch(params, var_params, opt_state, X, batch_idx, row_idx, col_idx, dists, gamma, alpha, key):
    loss, grads = value_and_grad(batched_elbo_loss, argnums=(0, 1))(
        params, var_params, X, batch_idx, row_idx, col_idx, dists, gamma, alpha, key
    )
    updates, opt_state = optimizer.update(grads, opt_state, (params, var_params))
    params, var_params = optax.apply_updates((params, var_params), updates)
    return params, var_params, opt_state, loss

# --- Full Training Pipeline ---
def train_model(X, pos, n_factors=10, n_epochs=5000, gamma=10.0, alpha=1.0, learning_rate=1e-2, batch_size=128, seed=42):
    key = random.PRNGKey(seed)
    n_variants, n_samples = X.shape

    params = init_params(key, n_variants, n_samples, n_factors)
    var_params = init_variational_params(key, n_samples, n_factors, n_variants)

    global optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init((params, var_params))

    theta_init = jnp.ones(n_variants)
    rows, cols, vals, dists = vectorized_sparse_graph(pos, theta_init, gamma)

    n_batches = n_samples // batch_size

    for epoch in range(n_epochs):
        key, subkey = random.split(key)
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        params, var_params, opt_state, loss = update_batch(
            params, var_params, opt_state, X, batch_indices, rows, cols, dists, gamma, alpha, subkey
        )
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return params, var_params

# --- Fast MC Imputation ---
def impute_training_data_fast(params, var_params, n_samples_mc=100):
    W, mu, _ = params
    mu_z, logvar_z, _, _ = var_params
    
    keys = random.split(random.PRNGKey(0), n_samples_mc)
    eps = jax.vmap(lambda key: random.normal(key, mu_z.shape))(keys)
    z_samples = mu_z[None, :, :] + jnp.exp(0.5 * logvar_z[None, :, :]) * eps

    lam_samples = jnp.exp(jnp.einsum('vf,mfs->mvs', W, z_samples) + mu[:, None, None])
    imputed_mean = jnp.mean(lam_samples, axis=0)
    imputed_std = jnp.std(lam_samples, axis=0)
    return imputed_mean, imputed_std

# --- Hotspot Plotting ---
def plot_recombination_hotspots(pos, true_theta, inferred_mu_theta):
    inferred_theta = jnp.exp(inferred_mu_theta)

    plt.figure(figsize=(12, 6))
    plt.plot(pos / 1e6, true_theta, label="True Theta (Hotspots)", color="blue")
    plt.plot(pos / 1e6, inferred_theta, label="Inferred Theta", color="red", alpha=0.7)
    plt.xlabel("Position (Mb)")
    plt.ylabel("Recombination Modifier (Theta)")
    plt.title("Recombination Hotspots: True vs Inferred")
    plt.legend()
    plt.show()

# ------------------ 🧬 Run the Full System 🧬 ------------------
if __name__ == "__main__":
    n_variants = 1000
    n_samples = 500
    n_factors = 10
    gamma = 10.0
    alpha = 1.0
    learning_rate = 1e-2
    n_epochs = 2000
    batch_size = 128

    pos, true_theta = simulate_recombination_map(n_variants, n_hotspots=5, hotspot_strength=10.0)
    X = simulate_zip_data(n_variants, n_samples, n_factors, pos, true_theta)

    params, var_params = train_model(X, pos, n_factors, n_epochs, gamma, alpha, learning_rate, batch_size)

    imputed_mean, imputed_std = impute_training_data_fast(params, var_params)

    print("Training complete!")

    plot_recombination_hotspots(pos, true_theta, var_params[2])
