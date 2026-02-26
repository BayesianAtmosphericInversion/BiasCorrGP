import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from src.biascorrgp.models import TripleGP_prior, TripleGP_posterior

def prediction_data(start_lon: float, start_lat: float, end_lon: float, end_lat: float, 
                    time_int: np.ndarray, xy_scaler: MinMaxScaler, ab_scaler:MinMaxScaler, stepsize: float=0.5) -> tuple[torch.tensor, torch.tensor]:
    """
    Creates the real and normalized coordinates of the new points.
    :param start_lon: start of the prediction grid in longitude. 
    :param start_lat: start of the prediction grid in latitude.
    :param end_lon: end of the prediction grid in longitude.
    :param end_lat: end of the prediction grid in latitude.
    :param time_int: time intervals for the grid.
    :param xy_scaler: scaler used for longitude an latitude, returned by function create_training_data.
    :param ab_scaler: scaler used for time intervals, returned by function create_training_data.
    :param stepsize: stepsize on the spatial grid in degrees.

    Returns:
    x_star: original coordinates.
    test_X: normalized coordinates.  
    """
    len_times = time_int.shape[0]
    span_lat = end_lat - start_lat
    span_lon = end_lon - start_lon
    steps_lon = int(round((span_lon+stepsize)/stepsize))
    steps_lat = int(round((span_lat+stepsize)/stepsize))
    print(f"{steps_lon} steps in logintude")
    print(f"{steps_lat} steps in latitude")
    grid_lons = np.linspace(start_lon, end_lon, steps_lon)
    grid_lats = np.linspace(start_lat, end_lat, steps_lat)
    ln, lt, tm = np.meshgrid(grid_lons, grid_lats, np.linspace(0, len_times-1, len_times), indexing='ij')
    x_star = torch.tensor(np.stack([ln, lt, time_int[tm.astype(int), 0], time_int[tm.astype(int), 1]], axis=-1).reshape(-1, 4))

    #scale the inputs
    X_star_np = x_star[:, :2].numpy()
    X_star_np = xy_scaler.transform(X_star_np)
    T_star_np = x_star[:, 2:].numpy().reshape(-1, 1)
    T_star_np = ab_scaler.transform(T_star_np)
    test_X = torch.tensor(np.concatenate((X_star_np, T_star_np.reshape(-1,2)), axis =1), dtype=torch.float64)

    return x_star, test_X

def predict_shifts(prior: TripleGP_prior, posterior: TripleGP_posterior, train_X: torch.Tensor, test_X: torch.Tensor,
                    same_lengthscale: bool = True, same_outputscale: bool = True)-> torch.Tensor:
    """
    Predicts the horizontal, vertical and time shifts for test data points with coordinates test_X based on the optimized posterior and the prior.
    Returns a tensor of shape (number of new points)x(3) where the second dimension is (predicted horizontal shift, predicted vertical shift, predicted time shift).
    :param train_X: coordinates of training data points, shape (number of training points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
    :param test_X: new coordinates, shape (number of new points)x(4) where the second dimension is (longitude, latitude, start of time interval, end of time interval)
    :param same_lengthscale: boolean indicating whether the lengthscales for the three GPs are the same, default True
    :param same_outputscale: boolean indicating whether the outputscales for the three GPs are the same, default True
    :param shvt: torch.Tensor of (sh, sv, st), default is (1.0,1.0,1.0)
    """

    if same_lengthscale and same_outputscale: #if all lengthscales and outputscales are the same, only one covariance matrix needs to be computed
        K_xxs = prior.covar_module_h(train_X, test_X)
        jitter = 1e-6
        try: 
            L = torch.linalg.cholesky(prior.covar_module_h() + prior.outputscale[0]*jitter*posterior.mask)
            hh = K_xxs.T @ torch.cholesky_solve(posterior.mu_h.unsqueeze(-1) if posterior.mu_h.ndim==1 else posterior.mu_h, L)
            hv = K_xxs.T @ torch.cholesky_solve(posterior.mu_v.unsqueeze(-1) if posterior.mu_v.ndim==1 else posterior.mu_v, L)
            ht = K_xxs.T @ torch.cholesky_solve(posterior.mu_t.unsqueeze(-1) if posterior.mu_t.ndim==1 else posterior.mu_t, L)
        except RuntimeError:
            print("Cholesky failed")
            hh = K_xxs.T @ torch.linalg.solve(prior.covar_module_h() + prior.outputscale[0]*jitter*posterior.mask, posterior.mu_h.unsqueeze(-1) if posterior.mu_h.ndim==1 else posterior.mu_h)
            hv = K_xxs.T @ torch.linalg.solve(prior.covar_module_v() + prior.outputscale[1]*jitter*posterior.mask, posterior.mu_v.unsqueeze(-1) if posterior.mu_v.ndim==1 else posterior.mu_v)
            ht = K_xxs.T @ torch.linalg.solve(prior.covar_module_t() + prior.outputscale[2]*jitter*posterior.mask, posterior.mu_t.unsqueeze(-1) if posterior.mu_t.ndim==1 else posterior.mu_t)
    else:
        Kh_xxs = prior.covar_module_h(train_X, test_X)
        Kv_xxs = prior.covar_module_v(train_X, test_X)
        Kt_xxs = prior.covar_module_t(train_X, test_X)
        jitter = 1e-6
        try: 
            Lh = torch.linalg.cholesky(prior.covar_module_h() + prior.outputscale[0]*jitter*posterior.mask)
            Lv = torch.linalg.cholesky(prior.covar_module_v() + prior.outputscale[1]*jitter*posterior.mask)
            Lt = torch.linalg.cholesky(prior.covar_module_t() + prior.outputscale[2]*jitter*posterior.mask)
            hh = Kh_xxs.T @ torch.cholesky_solve(posterior.mu_h.unsqueeze(-1) if posterior.mu_h.ndim==1 else posterior.mu_h, Lh)
            hv = Kv_xxs.T @ torch.cholesky_solve(posterior.mu_v.unsqueeze(-1) if posterior.mu_v.ndim==1 else posterior.mu_v, Lv)
            ht = Kt_xxs.T @ torch.cholesky_solve(posterior.mu_t.unsqueeze(-1) if posterior.mu_t.ndim==1 else posterior.mu_t, Lt)
        except RuntimeError:
            print("Cholesky failed")
            hh = Kh_xxs.T @ torch.linalg.solve(prior.covar_module_h() + prior.outputscale[0]*jitter*posterior.mask, posterior.mu_h.unsqueeze(-1) if posterior.mu_h.ndim==1 else posterior.mu_h)
            hv = Kv_xxs.T @ torch.linalg.solve(prior.covar_module_v() + prior.outputscale[1]*jitter*posterior.mask, posterior.mu_v.unsqueeze(-1) if posterior.mu_v.ndim==1 else posterior.mu_v)
            ht = Kt_xxs.T @ torch.linalg.solve(prior.covar_module_t() + prior.outputscale[2]*jitter*posterior.mask, posterior.mu_t.unsqueeze(-1) if posterior.mu_t.ndim==1 else posterior.mu_t)

    return (torch.stack((hh, hv, ht)).squeeze())*(posterior.shvt.view(3,1))