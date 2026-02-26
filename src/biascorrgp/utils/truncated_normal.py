import torch
from torch.special import erf

def truncmom_upper_multi(mu: torch.Tensor, Sigma: torch.Tensor, sigma_corr: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns first and second moment of the upper truncated multivariate normal distribution with mean mu and covariance matrix Sigma.
    
    :param mu: mean value of the Normal distribution before truncation
    :param Sigma: covariance matrix of the Normal distribution before truncation
    :param sigma_corr: whether to apply correction of truncated variances for stability, default is True
    """
    a = 0.0
    sigma2 = torch.diagonal(Sigma)
    sigma = torch.sqrt(sigma2)
    alph = (a - mu) / (torch.sqrt(torch.tensor(2.0, dtype=mu.dtype)) * sigma)

    istab = (alph <= 3).nonzero(as_tuple=False).squeeze()
    astab = (alph > 3).nonzero(as_tuple=False).squeeze()

    hatx = torch.zeros_like(mu)
    hatx2 = torch.zeros_like(mu)

    if istab.numel() > 0:
        alf_stable = alph[istab]
        pom = (1 - erf(alf_stable)) * torch.sqrt(torch.tensor(torch.pi / 2, dtype=mu.dtype))
        gam = (-torch.exp(-alf_stable**2)) / pom
        del_val = (-a * torch.exp(-alf_stable**2)) / pom
        hatx[istab] = mu[istab] - sigma[istab] * gam
        hatx2[istab] = sigma[istab]**2 + mu[istab] * hatx[istab] - sigma[istab] * del_val

    if astab.numel() > 0:
        hatx[astab] = a**2 - sigma[astab]**2 / mu[astab]
        hatx2[astab] = a**2 - 2 * a * sigma[astab]**2 / mu[astab] + \
            (2 * sigma[astab]**4 - 2 * a**2 * sigma[astab]**2) / (mu[astab]**2)

    faulta = (hatx < a).nonzero(as_tuple=False).squeeze()
    if faulta.numel() > 0:
        print("tN:(mv < a)")
        hatx[faulta] = 0.0
        hatx2[faulta] = torch.abs(hatx[faulta])

    if sigma_corr:
        varx = torch.clamp(hatx2 - hatx**2, min=0.0)
        o = torch.sqrt(varx / sigma2)
        o_diag = torch.diag(o)
        hatxxT = hatx.unsqueeze(1) @ hatx.unsqueeze(0) + o_diag @ Sigma @ o_diag
    else:
        varx = hatx2 - hatx**2
        hatxxT = hatx.unsqueeze(1) @ hatx.unsqueeze(0) + torch.diag(varx)

    return hatx, hatxxT