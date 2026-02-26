import torch

from src.biascorrgp.utils.truncated_normal import truncmom_upper_multi


class lsapc_sourceTerm:
    """
    Class for LS-APC model of a source term.

    Attributes:
    p: numebr of measurements
    n: number of discretization steps of the source term
    alpha0, beta0: prior parameters of upsilon
    nu0, rho0: prior parameters of omega
    zeta0, eta0: prior parameters of psi
    l0: prior mean of l
    Sigma_x, mu_x: posterior covariance and mean of the source_term
    alpha, beta: posterior parameters of upsilon
    Sigma_l, mu_l: posterior covariance and mean of l
    zeta, eta: posterior parameters of psi
    nu, rho: posterior parameters of omega
    hat_omega: first moment of omega
    hat_ups: first moment of upsilon
    hat_x: first moment of the source term
    hat_xxt: second moment of the source term
    hat_psi: first moment of psi
    omega_opt: boolean indicating whether omega should be optimized in the optimization step.

    Adapted from Tichý, O., Šmídl, V., Hofman, R., Stohl, A., 2016. LS-APC v1.0: a tuning-free method for the linear inverse problem and its application to source-term determination. Geoscientific Model Development 9, 4297–4311.
    """

    def __init__(self, M: torch.Tensor, gamma: float = 1e-2, omega_opt: bool = False) -> None:
        """
        :param M: SRS matrix
        :param gamma: scales the prior mean of upsilon
        :param omega_opt: boolean indicating whether omega should be optimized in the optimization step
        """
        (self.p, self.n) = M.shape

        #prior parameters
        self.alpha0 = 1e-10 #1e-1 for ruthenium
        self.beta0 = 1e-10 #1e-1 for ruthenium
        self.nu0 = 1e-10
        self.rho0 = 1e-10
        self.zeta0 = 1e-2
        self.eta0 = 1e-2
        self.l0 = -torch.ones(self.n-1, dtype = torch.float64)


        #posterior parameters
        self.Sigma_x = torch.diag(torch.ones(self.n, dtype = torch.float64)) #the initial value not used
        self.mu_x = torch.ones(self.n, dtype = torch.float64) #the initial value not used
        self.alpha = self.alpha0 + 1/2 * torch.ones(self.n, dtype = torch.float64)
        self.beta = self.alpha0 + 1/2 * torch.ones(self.n, dtype = torch.float64)
        self.Sigma_l = torch.diag(torch.zeros(self.n-1, dtype = torch.float64))
        self.mu_l = torch.zeros(self.n-1, dtype = torch.float64)
        self.zeta = self.zeta0 + 1/2
        self.eta = self.zeta0 + 1/2
        self.nu = self.nu0 + self.p/2
        self.rho = self.nu0 + self.p/2

        #expectations
        self.hat_omega = torch.max(M.T @ M)**(-1)
        self.hat_ups = gamma * torch.ones(self.n, dtype = torch.float64)
        self.hat_x = torch.ones(self.n, dtype = torch.float64) #initial value not used
        self.hat_xxt = self.mu_x.unsqueeze(0) @ self.mu_x.unsqueeze(1) #initial value not used
        self.hat_psi = torch.ones(self.n-1, dtype = torch.float64) #varsigma

        self.omega_opt = omega_opt #whether we should optimize omega iniside ls_apc optimization step 

    def reinit_alpha_beta(self, alpha0: float | None = None, beta0: float | None = None) -> None:
        """
        Reinitializes alpha and beta for given alpha0 and beta0 and sets alpha0 and beta0 values. 
        If they are not given, it reinitializes alpha and beta for current values of alpha0 and beta0.
        
        :param alpha0: alpha0 value if it should be changed
        :param beta0: beta0 value if it should be changed
        """
        if alpha0 is not None:
            self.alpha0 = alpha0
        if beta0 is not None:
            self.beta0 = beta0
        self.alpha = self.alpha0 + 1/2 * torch.ones(self.n, dtype = torch.float64)
        self.beta = self.alpha0 + 1/2 * torch.ones(self.n, dtype = torch.float64)

    def reinint_zeta_eta(self, zeta0: float | None = None, eta0: float | None = None) -> None:
        """
        Reinitializes zeta and eta for given zeta0 and eta0 and sets zeta0 and eta0 values. 
        If they are not given, it reinitializes zeta and eta for current values of zeta0 and eta0.
        
        :param zeta0: zeta0 value if it should be changed
        :param eta0: eta0 value if it should be changed
        """
        if zeta0 is not None:
            self.zeta0 = zeta0
        if eta0 is not None:
            self.eta0 = eta0
        self.zeta = self.zeta0 + 1/2
        self.eta = self.zeta0 + 1/2

    def reinit_nu_rho(self, nu0: float | None = None, rho0: float | None = None) -> None:
        """
        Reinitializes nu and rho for given nu0 and rho0 and sets nu0 and rho0 values. 
        If they are not given, it reinitializes nu and rho for current values of nu0 and rho0.
        
        :param nu0: nu0 value if it should be changed
        :param rho0: rho0 value if it should be changed
        """
        if nu0 is not None:
            self.nu0 = nu0
        if rho0 is not None:
            self.rho0 = rho0
        self.nu = self.nu0 + self.p/2
        self.rho = self.nu0 + self.p/2

    def get_L(self, ls: torch.Tensor) -> torch.Tensor:
        """
        Returns the matrix L for a given vector of ls.
        param ls: vector of expected value of l.
        """
        L = torch.diag(torch.ones(self.n))
        L[1:].diagonal().copy_(ls) 
        return L
    

    def optimization_step(self, M: torch.Tensor, MtM: torch.Tensor, y: torch.Tensor):
        """
        Performs one update of LS-APC parameters as described in the paper.
        :param M: SRS matrix (first moment of the corrected one if used with correction algorithm)
        :param MtM: M.T @ M (second moment of the corrected M.T @ M if used with correction algorithm)
        :param y: measurement vector
        """
        #moments of l required for further computations
        hat_llt = self.mu_l.unsqueeze(1) @ self.mu_l.unsqueeze(0) + self.Sigma_l 
        hat_LNuLt = self.get_L(self.hat_ups[:-1]*self.mu_l)
        hat_LNuLt = hat_LNuLt + hat_LNuLt.T
        new_diag = self.hat_ups.clone()
        new_diag[1:] += self.hat_ups[:-1] * torch.diagonal(hat_llt, 0)
        hat_LNuLt.diagonal().copy_(new_diag)

        #update of source term 
        self.Sigma_x = torch.linalg.inv(self.hat_omega * MtM + hat_LNuLt + 0*1e-4*torch.eye(self.n))
        self.mu_x = self.Sigma_x @ (self.hat_omega * M.T @ y.unsqueeze(1))

        self.hat_x, self.hat_xxt = truncmom_upper_multi(self.mu_x.reshape(-1), self.Sigma_x, sigma_corr = True)
        hat_xx = torch.diagonal(self.hat_xxt, offset = -1) #first subdiagonal
        diag_hatx2 = torch.diag(self.hat_xxt)
        low_diag = 2*self.mu_l * hat_xx + torch.diagonal(hat_llt) * diag_hatx2[1:]
        hat_diagLtxxtL = diag_hatx2 + torch.cat([low_diag, torch.tensor([0.], dtype=low_diag.dtype)])

        #update of upsilon
        self.beta = self.beta0 + 1/2 * hat_diagLtxxtL
        self.hat_ups = self.alpha / self.beta
        
        #update of l
        self.Sigma_l = torch.diag((self.hat_ups[:-1] * diag_hatx2[1:] + self.hat_psi) ** (-1)) #diagonal
        self.mu_l = self.Sigma_l @ (-self.hat_ups[:-1] * hat_xx + self.l0 * self.hat_psi)

        hat_llt = self.mu_l.unsqueeze(1) @ self.mu_l.unsqueeze(0) + self.Sigma_l 
        hat_lml02 = torch.diagonal(hat_llt) - 2*self.mu_l*self.l0 + self.l0**2
    
        #update of psi
        self.eta = self.eta0 + 1/2 * hat_lml02
        self.hat_psi = self.zeta / self.eta

        #update of omega if it should be optimized
        if self.omega_opt:
            self.rho = self.rho0 + 1/2 * torch.trace(self.hat_xxt @ MtM) - y.unsqueeze(0) @ M @ self.hat_x.unsqueeze(1) + 1/2 * y.unsqueeze(0) @ y.unsqueeze(1)
            self.hat_omega = self.nu / self.rho