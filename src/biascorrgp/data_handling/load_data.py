import numpy as np
import h5py
from dataclasses import dataclass
import torch

@dataclass
class InputData:
    """
    Stores all input data as np.ndarrays. 

    Attributes:
    M: SRS matrix for the coordinates of the measurements.
    Ms: SRS matrix for coordinates of measurements shifted by one step to south.
    Mn: SRS matrix for coordinates of measurements shifted by one step to north.
    Mw: SRS matrix for coordinates of measurements shifted by one step to west.
    Me: SRS matrix for coordinates of measurements shifted by one step to east.
    Mtm: SRS matrix for coordinates of measurements shifted by one step back in time.
    Mtp: SRS matrix for coordinates of measurements shifted by one step forward in time.
    x_true: true source term, it it is available, otherwise None.
    y: all measurements.
    longitudes: longitudes of the measurement stations.
    latitudes: latitudes of the measurement stations.
    times: time intervals of the measurements. First column is tsrat, second end of the interval.
    """
    M: np.ndarray
    Ms: np.ndarray
    Mn: np.ndarray
    Mw: np.ndarray
    Me: np.ndarray
    Mtm: np.ndarray
    Mtp: np.ndarray
    x_true: np.ndarray | None
    y: np.ndarray
    longitudes: np.ndarray
    latitudes: np.ndarray
    times: np.ndarray

def load_input_data(filename, xtrue: bool = False) -> InputData:
    """
    Creates InputData instance from input data file.
    :param filename: path to the .h5 file with input data.
    :param xtrue: whether the file contains true source term, default is False.
    """
    with h5py.File(filename, "r") as f:
        M = f["M"][:]
        Ms = f["Ms"][:]
        Mn = f["Mn"][:]
        Mw = f["Mw"][:]
        Me = f["Me"][:]
        Mtm = f["Mtm"][:]
        Mtp = f["Mtp"][:]
        x_true = f["x_true"][:] if xtrue else None
        y = f["y"][:]
        longitudes = f["longitudes"][:]
        latitudes = f["latitudes"][:]
        times = f["times"][:]

    return InputData(M, Ms, Mn, Mw, Me, Mtm, Mtp, x_true, y, longitudes, latitudes, times)

@dataclass
class InversionDataTorch:
    """
    Contains difference matrices Dh, Dv, Dt and data necessary for inversion as torch.Tensors.
    Attributes:
    y: measurement vector.
    M: SRS matrix for the coordinates of the measurements.
    Dh: horizontal difference of SRS matrices.
    Dv: vertical difference of SRS matrices.
    Dt: temporal difference of SRS matrices.
    coef: normalization coefficient
    """
    y: torch.Tensor
    M: torch.Tensor
    Dh: torch.Tensor
    Dv: torch.Tensor
    Dt: torch.Tensor
    coef: float


def prepare_inversion_data(data: InputData) -> InversionDataTorch:
    """
    Creates difference matrices Dh, Dv, Dt, normalizes and transfers M and y to torch.Tensors. Returns InversionDataTorch instance with all these data.
    :param data: InputData instance with all input data.
    """
    Dh = (data.Me - data.Mw)/2
    Dv = (data.Ms - data.Mn)/2
    Dt = (data.Mtm - data.Mtp)/2

    coef = np.max(data.y)

    return InversionDataTorch(torch.from_numpy(data.y/coef), torch.from_numpy(data.M/coef), torch.from_numpy(Dh/coef), torch.from_numpy(Dv/coef), torch.from_numpy(Dt/coef), coef)