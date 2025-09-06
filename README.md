# Denoising Diffusion Probabilistic Models (DDPM) — Linear Noise Scheduler

A minimal, math-first summary of DDPM (Ho et al., 2020) with a **linear β schedule**.  
Use this as a reference for implementing training and sampling loops.

---

## Notation

- $x_0$: clean data (e.g., images)  
- $x_t$: noised sample at step $t$  
- $\epsilon \sim \mathcal N(0,\mathbf I)$: Gaussian noise  
- $t \in \{1,\dots,T\}$: diffusion timestep (paper uses 1-based; code below is 0-based)  
- $\beta_t \in (0,1)$, $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s=1}^t \alpha_s$

---

## Linear Beta Schedule

$$
\beta_t
=
\beta_{\text{start}}
+
\frac{t-1}{T-1}\bigl(\beta_{\text{end}}-\beta_{\text{start}}\bigr)
$$

Common choices in practice:  
$\beta_{\text{start}}\in[10^{-4},10^{-3}]$, $\beta_{\text{end}}\in[0.02,0.05]$.

---

## Alphas and Cumulative Product

$$
\alpha_t = 1-\beta_t,
\qquad
\bar\alpha_t = \prod_{s=1}^{t}\alpha_s
$$

Precompute $\{\alpha_t\}_{t=1}^T$ and $\{\bar\alpha_t\}_{t=1}^T$ once.

---

## Forward (Diffusion) Process *(Ho et al., Eq. 4)*

$$
q(x_t\mid x_0)=\mathcal N\!\left(
x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\mathbf I
\right),
\qquad
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon
$$

---

## Reverse (Denoising) Mean *(Ho et al., Eq. 11)*

Let $\epsilon_\theta(x_t,t)$ be the network that predicts the added noise.  
Then the mean of the reverse transition $p_\theta(x_{t-1}\mid x_t)$ is

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\,\epsilon_\theta(x_t,t)
\right).
$$

---

## Posterior (Sampling) Variance *(Ho et al., Eq. 15)*

$$
\sigma_t^2
=
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,
\quad (\text{with }\bar\alpha_0:=1)
$$

At $t=1$, sampling often uses $\sigma_1^2=\tilde\beta_1$; at $t>1$, sample
$x_{t-1}\sim\mathcal N(\mu_\theta,\sigma_t^2\mathbf I)$.

---

## Minimal PyTorch Snippets

### 0) Schedules (0-based indexing in code)

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 1000
beta_start, beta_end = 1e-4, 0.02

betas = torch.linspace(beta_start, beta_end, T, device=device)     # [T]
alphas = 1.0 - betas                                              # [T]
alphas_cumprod = torch.cumprod(alphas, dim=0)                     # \bar{α}_t
alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

# Useful square-roots
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# Posterior variance (tilde beta)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

# (Optional) numerical safety
eps = 1e-12
alphas_cumprod = torch.clamp(alphas_cumprod, min=eps, max=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=eps))
