import torch

class TripleGP_prior():
    """
    Class encompassing all three GP priors needed for BiasCorrGP
    
    Attributes:
        dist_mat: precomputed distance matrix between coordinates of training data, used for faster computation of covariance matrices
        outputscale: vector of outputscales (horizontal, vertical, time) for the three GPs, default is 1
        lengthscale: vector of lengthscales (horizontal, vertical, time) for the three GPs, default is 1
    """
    def __init__(self, train_X: torch.Tensor, lengthscale: torch.Tensor | None = None, outputscale: float | None = None) -> None:
        """
        :param train_X: Training data coordinates.
        :param lengthscale: Lengthscales for the three GPs, if not provided they are initialized as 1.
        :param outputscale: Outputscale for the three GPs, if not provided it is initialized as 1.
        """
        self.dist_mat = distance_matrix(train_X, train_X)
        if lengthscale is not None:
            self.lengthscale = lengthscale.detach().clone()
        else:
            self.lengthscale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        if outputscale is not None:
            self.outputscale = torch.tensor([outputscale, outputscale, outputscale], dtype=torch.float64)
        else:
            self.outputscale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    def covar_module_h(self, x1: torch.Tensor | None = None, x2: torch.Tensor | None = None) -> torch.Tensor:
        """
        Computes prior covariance for horizontal shifts based on RBF kernel for coordintes x1 and x2. If they are not given, it is created for training data.
        
        :param x1: coordinates of first set of points, shape (number of points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
        :param x2: coordinates of second set of points, shape (number of points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
        """
        if x1 is None and x2 is None:
            return self.outputscale[0]*torch.exp(-0.5*self.dist_mat/self.lengthscale[0]**2)
        else:
            D = distance_matrix(x1, x2)
            return self.outputscale[0]*torch.exp(-0.5*D/self.lengthscale[0]**2)
    
    def covar_module_v(self, x1: torch.Tensor | None = None, x2: torch.Tensor | None = None) -> torch.Tensor:
        """
        Computes prior covariance for vertical shifts based on RBF kernel for coordintes x1 and x2. If they are not given, it is created for training data.
        
        :param x1: coordinates of first set of points, shape (number of points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
        :param x2: coordinates of second set of points, shape (number of points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
        """
        if x1 is None and x2 is None:
            return self.outputscale[1]*torch.exp(-0.5*self.dist_mat/self.lengthscale[1]**2)
        else:
            D = distance_matrix(x1, x2)
            return self.outputscale[1]*torch.exp(-0.5*D/self.lengthscale[1]**2)

    def covar_module_t(self, x1: torch.Tensor | None = None, x2: torch.Tensor | None = None) -> torch.Tensor:
        """
        Computes prior covariance for time shifts based on RBF kernel for coordintes x1 and x2. If they are not given, it is created for training data.
        
        :param x1: coordinates of first set of points, shape (number of points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
        :param x2: coordinates of second set of points, shape (number of points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
        """
        if x1 is None and x2 is None:
            return self.outputscale[2]*torch.exp(-0.5*self.dist_mat/self.lengthscale[2]**2)
        else:
            D = distance_matrix(x1, x2)
            return self.outputscale[2]*torch.exp(-0.5*D/self.lengthscale[2]**2)


    def set_lengthscale(self, lh: float | None = None, lv: float | None = None, lt: float | None = None) -> None:
        """
        Sets prior lengthscales for the three GPs. If a lengthscale is not given, it is not changed.
 
        :param lh: Horizontal lengthscale
        :param lv: Vertical lengthscale
        :param lt: Time lengthscale
        """
        if (lh is None) & (lv is None) & (lt is None):
            raise Exception("No lengthscale was given!")
        else:
            if lh is not None:
                self.lengthscale[0] = lh
            if lv is not None:
                self.lengthscale[1] = lv
            if lt is not None:
                self.lengthscale[2] = lt

    def set_outputscale(self, sh: float | None = None, sv: float | None = None, st: float | None = None) -> None:
        """
        Sets prior output scales for the three GPs. If an outputscale is not given, it is not changed.
 
        :param sh: Horizontal outputscale
        :param sv: Vertical outputscale
        :param st: Time outputscale
        """
        if (sh is None) & (sv is None) & (st is None):
            raise Exception("No outputscale was given!")
        else:
            if sh is not None:
                self.outputscale[0] = sh
            if sv is not None:
                self.outputscale[1] = sv
            if st is not None:
                self.outputscale[2] = st

def distance_matrix(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Computes the  squared distance matrix between points in X1 and X2.
    
    :param X1: tensor of coordinates of first set of points, shape (number of points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
    :param X2: tensor of coordinates of second set of points, shape (number of points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
    """
    int_dist = interval_distance_matrix(X1[:, 2:4], X2[:, 2:4])**2 #distance between time intervals
    point_dist = torch.cdist(X1[:, 0:2], X2[:, 0:2], p=2)**2 #distance between longitudes and latitudes

    return int_dist + point_dist

def interval_distance_matrix(intervals1: torch.Tensor, intervals2: torch.Tensor) -> torch.Tensor:
    """
    Compute RBF kernel between intervals using custom L2 indicator distance.

    intervals: (N, 2) torch tensor
    intervals2: (M, 2) torch tensor
    """
    #N1 = intervals1.shape[0]
    #N2 = intervals2.shape[0]
    #dist_matrix = torch.zeros((N1, N2))
    #for i in range(N1):
    #    dist_matrix[i, :] = int_distance5_broad(intervals1[i, 0], intervals1[i, 1], intervals2[:, 0], intervals2[:, 1])
    N1 = intervals1.shape[0]
    N2 = intervals2.shape[0]
    dist_matrix = torch.zeros((N1, N2))
    for i in range(N1):
        #for j in range(N2):
        #    dist_matrix[i, j] = int_distance5(intervals1[i, 0], intervals1[i, 1], intervals2[j, 0], intervals2[j, 1])
        dist_matrix[i, :] = int_distance_broad(intervals1[i, 0], intervals1[i, 1], intervals2[:, 0], intervals2[:, 1])

    return dist_matrix

def int_distance_broad(a1: torch.Tensor, b1: torch.Tensor, a2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """
    Distance between inatervals <a1, b1> and <a2, b2>. For definition see Appendinx A of the papper.
    
    :param a1: Vector of interval starts for first set of intervals, shape (number of intervals)x(1)
    :param b1: Vector of interval ends for first set of intervals, shape (number of intervals)x(1)
    :param a2: Vector of interval starts for second set of intervals, shape (1)x(number of intervals)
    :param b2: Vector of interval ends for second set of intervals, shape (1)x(number of intervals)
    """
    a1 = torch.as_tensor(a1, dtype=torch.float64)
    b1 = torch.as_tensor(b1, dtype=torch.float64)

    # Case: a1 < a2
    case1 = a1 < a2
    case1_no_overlap = case1 & (b1 <= a2)
    case1_overlap    = case1 & ~(b1 <= a2)

    # Case: a2 <= a1
    case2 = ~case1
    case2_no_overlap = case2 & (b2 <= a1)
    case2_overlap    = case2 & ~(b2 <= a1)

    # Initialize outputs
    iou_dist = torch.zeros_like(a2, dtype=torch.float64)
    euc_dist = torch.zeros_like(a2, dtype=torch.float64)

    # Fill per case
    iou_dist[case1_overlap] = (torch.minimum(b1, b2[case1_overlap]) - a2[case1_overlap]) / \
                              (torch.maximum(b1, b2[case1_overlap]) - a1)
    euc_dist[case1_no_overlap] = a2[case1_no_overlap] - b1

    iou_dist[case2_overlap] = (torch.minimum(b1, b2[case2_overlap]) - a1) / \
                              (torch.maximum(b1, b2[case2_overlap]) - a2[case2_overlap])
    euc_dist[case2_no_overlap] = a1 - b2[case2_no_overlap]

    return ((1 - iou_dist) + euc_dist) / 2

class TripleGP_posterior():
    """
    Class encompassing all three GP posteriors needed for BiasCorrGP
    
    Attributes:
        Dh: matrix of differences in the SRS matrix for horizontal shifts
        Dv: matrix of differences in the SRS matrix for vertical shifts
        Dt: matrix of differences in the SRS matrix for time shifts 
        M: SRS matrix
        mu_h: vector of posterior mean values for horizontal shifts, initialized as zero vector     
        mu_v: vector of posterior mean values for vertical shifts, initialized as zero vector
        mu_t: vector of posterior mean values for time shifts, initialized as zero vector
        Sigma_h: posterior covariance matrix for horizontal shifts
        Sigma_v: posterior covariance matrix for vertical shifts
        Sigma_t: posterior covariance matrix for time shifts
        mask: identity matrix used for masking in optimization step
    """
    def __init__(self, Dh: torch.Tensor, Dv: torch.Tensor, Dt: torch.Tensor, M: torch.Tensor, 
                 sh: float = 1.0, sv: float = 1.0, st: float = 1.0) -> None:
        """
        :param Dh: matrix of differences in the SRS matrix for horizontal shifts
        :param Dv: matrix of differences in the SRS matrix for vertical shifts
        :param Dt: matrix of differences in the SRS matrix for time shifts
        :param M: SRS matrix
        :param sh: real stepsize s_h in longitude
        :param sv: real stepsize s_v in latitude
        :param st: real stepsize s_t in time
        """
        self.Dh = Dh.double()
        self.Dv = Dv.double()
        self.Dt = Dt.double()
        self.M = M.double()
        N, _ = self.M.shape #N is the number of measurements

        # initialization of posterior mean values
        self.mu_h = torch.zeros(N, dtype = torch.float64)
        self.mu_v = torch.zeros(N, dtype = torch.float64)
        self.mu_t = torch.zeros(N, dtype = torch.float64)
        # initialization of posterior covariance values
        self.Sigma_h = 0*torch.eye(N, dtype = torch.float64)
        self.Sigma_v = 0*torch.eye(N, dtype = torch.float64)
        self.Sigma_t = 0*torch.eye(N, dtype = torch.float64)

        #step sizes s_h, s_v, s_t
        self.shvt = torch.tensor([sh, sv, st], dtype = torch.float64)

        self.mask = torch.eye(N, dtype = torch.float64)


    def optimization_step(self, hat_z: torch.Tensor, hat_zz: torch.Tensor, y: torch.Tensor, hat_omega: float, prior: TripleGP_prior, 
                          Kh_xx: torch.Tensor | None = None, Kv_xx: torch.Tensor | None = None, Kt_xx: torch.Tensor | None = None) -> None: 
        """
        Bias correction optimization step as described in Algotihm 1 of the paper.

        :param hat_z: first moment of the source term
        :param hat_zz: second moment of the source term
        :param y: measurements
        :param hat_omega: expected value of precision of measurements
        :param prior: TripleGP_prior object containing the prior covariance functions and their hyperparameters
        :param Kh_xx: prior covariance matrix for horizontal shifts for training data, if not given it is computed using the prior object
        :param Kv_xx: prior covariance matrix for vertical shifts for training data, if not given it is computed using the prior object
        :param Kt_xx: prior covariance matrix for time shifts for training data, if not given it is computed using the prior object
        """

        if Kh_xx is None:
            Kh_xx = prior.covar_module_h()
        if Kv_xx is None:
            Kv_xx = prior.covar_module_v()
        if Kt_xx is None:
            Kt_xx = prior.covar_module_t()

        #Sigma_h
        quad_term = (self.Dh @ (hat_zz) @ self.Dh.T)*self.mask 
        quad_Kxx = hat_omega*quad_term @ Kh_xx
        self.Sigma_h = Kh_xx - Kh_xx @ torch.linalg.inv(self.mask + quad_Kxx) @ quad_Kxx

        #mu_h
        fo_term = (self.Dh @ hat_z) * y
        L = self.M + torch.diag(self.mu_v) @ self.Dv + torch.diag(self.mu_t) @ self.Dt
        quad_terms = self.Dh @ hat_zz @ L.T
        self.mu_h = self.Sigma_h @ (fo_term - torch.diag(quad_terms)) * hat_omega

        #Sigma_v
        quad_term = (self.Dv @ (hat_zz) @ self.Dv.T)*self.mask 
        quad_Kxx = hat_omega*quad_term @ Kv_xx
        self.Sigma_v = Kv_xx - Kv_xx @ torch.linalg.inv(self.mask + quad_Kxx) @ quad_Kxx

        #mu_v
        fo_term = (self.Dv @ hat_z) * y
        L = self.M + torch.diag(self.mu_h) @ self.Dh + torch.diag(self.mu_t) @ self.Dt
        quad_terms = self.Dv @ hat_zz @ L.T
        self.mu_v = self.Sigma_v @ (fo_term - torch.diag(quad_terms)) * hat_omega

        #Sigma_t
        quad_term = (self.Dt @ (hat_zz) @ self.Dt.T)*self.mask 
        quad_Kxx = hat_omega*quad_term @ Kt_xx
        self.Sigma_t = Kt_xx - Kt_xx @ torch.linalg.inv(self.mask + quad_Kxx) @ quad_Kxx #using woodburry identity so that no inversion of Kxx is needed

        #mu_t
        fo_term = (self.Dt @ hat_z) * y
        L = self.M + torch.diag(self.mu_h) @ self.Dh + torch.diag(self.mu_v) @ self.Dv
        quad_terms = self.Dt @ hat_zz @ L.T
        self.mu_t = self.Sigma_t @ (fo_term - torch.diag(quad_terms)) * hat_omega

    def get_tildeM(self) -> torch.Tensor:
        """
        Returns corrected SRS matrix (its first moment).
        """
        return self.M + self.mu_h.unsqueeze(1) * self.Dh + self.mu_v.unsqueeze(1) * self.Dv + self.mu_t.unsqueeze(1) * self.Dt
    
    def get_tildeMttildeM(self, tildeM: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns second moment of corrected SRS matrix.
        :param tildeM: first moment of corrected SRS matrix, if not given it is computed using get_tildeM function
        """
        if tildeM is None:
            tildeM = self.get_tildeM()

        diag_term = self.Dh.T @ (self.mask*self.Sigma_h) @ self.Dh + self.Dv.T @ (self.mask*self.Sigma_v) @ self.Dv + self.Dt.T @ (self.mask*self.Sigma_t) @ self.Dt
        return tildeM.T @ tildeM + diag_term
    
    def predict_y(self, z: torch.Tensor, sc: float = 1.0) -> torch.Tensor:
        """
        Predicts the measurements for a given source term z based on the optimized posterior.
        :param z: source term, shape (number of source term components)x(1)
        :param sc: scaling coefficient of data, if used
        """
        tildeM = self.get_tildeM()
        y_corr = tildeM @ z.reshape(-1,1)
        return y_corr*sc
    
    def estimated_shifts(self) -> torch.Tensor:
        """
        Returns estimated shifts in real units. 
        """
        return (torch.stack((self.mu_h, self.mu_v, self.mu_t)).squeeze())*(self.shvt.view(3,1))