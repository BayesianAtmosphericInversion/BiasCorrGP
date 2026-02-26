import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.biascorrgp.data_handling.load_data import InputData 


def create_training_data(data: InputData) -> tuple[torch.Tensor, MinMaxScaler, MinMaxScaler]:
    """
    Creates training data - normalized coordinates of measurements.
    :param data: InputData instance with all input data.
    Returns:
    train_X: torch.Tensor of shape (N, 4) with normalized coordinates of measurements. First two columns are normalized longitudes and latitudes, third and fourth are normalized time intervals.
    xy_scaler: MinMaxScaler for longitude and latitude.
    ab_scaler: MinMaxScaler for time intervals.
    """
    X_xy = torch.tensor(np.stack((data.longitudes, data.latitudes))).T.double() 
    X_ab = torch.tensor(data.times).reshape(-1).double()

    # Normalize input to [0, 1]
    xy_scaler = MinMaxScaler() #scalar of longitude and latitude
    xy_np = X_xy.numpy()
    xy_np = xy_scaler.fit_transform(xy_np) #scale longitude and latitude separately
    ab_scaler = MinMaxScaler() #scalar of time intervals
    ab_np = X_ab.numpy()
    ab_np = ab_scaler.fit_transform(ab_np.reshape(-1, 1)) 
    t_np = ab_np.reshape(-1,2) #back to interval shape

    train_X = torch.tensor(np.concatenate((xy_np, t_np), axis=1), dtype=torch.float64) 
    return train_X, xy_scaler, ab_scaler

def normalized_lengthscales(lh: float, lv: float, lt:float, lon_lat_scaler: MinMaxScaler, time_scaler: MinMaxScaler) -> torch.Tensor:
    """
    Computes normalized lengthscales for given lengthscales in longitude, latitude and time directions and scalers for longitude/latitude and time.
    :param lh: lengthscale in longitude direction.
    :param lv: lengthscale in latitude direction.
    :param lt: lengthscale in time direction.
    :param lon_lat_scaler: MinMaxScaler for longitude and latitude.
    :param time_scaler: MinMaxScaler for time intervals.
    Returns:
    torch.Tensor of shape (3,) with normalized lengthscales in longitude, latitude and time directions.
    """
    #lengthscales are normalized by the range of the corresponding input dimension, which is given by the scaler scale_ attribute
    norm_lh = lh / lon_lat_scaler.data_range_[0]
    norm_lv = lv / lon_lat_scaler.data_range_[1]
    norm_lt = lt / time_scaler.data_range_[0]
    return torch.tensor([norm_lh, norm_lv, norm_lt], dtype=torch.float64)

def normalized_lengthscales_same(l: float, lon_lat_scaler: MinMaxScaler, time_scaler: MinMaxScaler) -> torch.Tensor:
    """
    Computes normalized lengthscales for given lengthscale in all directions and scalers for longitude/latitude and time.
    Used in case when the longitude and latitude lengthscales are the same and time is double of them.
    :param l: lengthscale in all directions.
    :param lon_lat_scaler: MinMaxScaler for longitude and latitude.
    :param time_scaler: MinMaxScaler for time intervals.
    Returns:
    torch.Tensor of shape (3,) with normalized lengthscales in longitude, latitude and time directions.
    """
    return normalized_lengthscales(l, l, l*2, lon_lat_scaler, time_scaler)