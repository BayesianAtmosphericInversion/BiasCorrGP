import matplotlib.pyplot as plt
import torch
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import cartopy.mpl.ticker as cticker
import matplotlib.cm as cm

from typing import Literal
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

from src.biascorrgp.data_handling.load_data import InputData 


Dataset = Literal["etex", "chernobyl", "ruthenium"]

def plot_source(source_term: torch.Tensor, data: Dataset, gt: np.ndarray | None = None) -> tuple[Figure, Axes]:
    """
    Plots the source term estimate.
    :param source_term: the estimate of source term, torch.Tensor.
    :param data: dataset for specific setting, either "etex", "chernobyl", or "ruthenium".
    :param gt: the ground-truth source term, if known. If not, it is None.

    Returns the figure and axes object.
    """
    st_np = source_term.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(st_np, label = "Estimate")
    if gt is not None:
        ax.plot(gt, label = "Ground truth")

    ax.set_title("Source term", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    if data == "etex":
        ax.set_xticks(np.arange(11, 238, 24))
        ax.set_xticklabels([f"{y}" for y in np.arange(2, 31, 3)] )
        ax.set_xlabel("Hour", fontsize=13)
        ax.set_ylabel("Emission (ng)", fontsize=13)
        ax.set_ylim(top = 8e13)
        ax.set_xlim((3*8, st_np.reshape(-1).shape[0]-3*8))
    elif data == "chernobyl":
        ax.set_xticks(np.arange(11, 238, 24))
        ax.set_xticklabels([f"{y}" for y in np.arange(2, 31, 3)] )
        ax.set_xlabel("April", fontsize=13)
        ax.set_ylabel("Emission (GBq)", fontsize=13)
        ax.set_ylim((0, 90))
        ax.set_xlim((3*8, st_np.reshape(-1).shape[0]-3*8))
    elif data == "ruthenium":
        ax.set_xticks(np.arange(3, 50, 8))
        ax.set_xticklabels([f"{21+y}" for y in np.arange(0, 6, 1)] )
        ax.set_xlabel("September", fontsize=13)
        ax.set_ylabel("Emission (TBq)", fontsize=13)
        ax.set_ylim((0, 350))

    ax.legend()
    return fig, ax


def plot_scatter_linear(y_meas: np.ndarray, y_corr: torch.tensor, data: Dataset) -> tuple[Figure, Axes]:
    """
    Plots the linear scatterplot of measurements vs prediction.
    :param y_meas: measurements vector, np.ndarray.
    :param y_corr: predicted concentrations, torch.tensor.
    :param data: dataset for specific setting, either "etex", "chernobyl", or "ruthenium".

    Returns the figure and axes object.
    """
    y_corr_np = y_corr.detach().cpu().numpy().reshape(-1)
    y_meas_np = y_meas.copy().reshape(-1)
    fig, ax = plt.subplots(figsize=(4, 4))

    ss_res = np.sum((y_corr_np - y_meas_np) ** 2).item()
    ss_tot = np.sum((y_meas_np - np.mean(y_meas_np)) ** 2).item()
    r2=1-ss_res/ss_tot
    fmse = np.mean((y_meas_np - y_corr_np)**2)

    y_corr_np_nz = y_corr_np.copy()
    y_corr_np_nz[y_corr_np_nz < 0] = 0

    x_min, x_max = np.min(np.concatenate((y_corr_np_nz, y_meas_np))), np.max(np.concatenate((y_corr_np_nz, y_meas_np)))
    line_x = np.array([x_min/2, x_max+x_max])
    line_y = line_x
    ax.plot(line_x, line_y, color='black', linewidth = 1)

    ax.scatter(y_meas_np, y_corr_np_nz, marker = 'o', s = 70, color = 'none', edgecolor = "C0")
    ax.tick_params(axis='both', labelsize=12)
    ax.set_aspect('equal')
    if data == "etex":
        ax.set_xlabel(r"$\text{Measurements }(\text{ng}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_ylabel(r"$\text{Predictions }(\text{ng}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_xlim(right=14, left = 0)
        ax.set_ylim(bottom = 0, top = 14)
        ax.set_xticks(np.arange(0,14,2))
        ax.set_yticks(np.arange(0,14,2))
    elif data == "chernobyl":
        ax.set_xlabel(r"$\text{Measurements }(\mu\text{Bq}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_ylabel(r"$\text{Predictions }(\mu\text{Bq}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_xlim(left = 1e-2, right = 700)
        ax.set_ylim(bottom = 1e-2, top = 700)
        ax.set_xticks(np.arange(0, 700, 300))
        ax.set_yticks(np.arange(0, 700, 300))
    elif data == "ruthenium":
        ax.set_xlabel(r"$\text{Measurements }(\text{mBq}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_ylabel(r"$\text{Predictions }(\text{mBq}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_xlim(left = 1e-2, right = 200)
        ax.set_ylim(bottom = 1e-2, top = 200)
        ax.set_xticks(np.arange(0,200,50))
        ax.set_yticks(np.arange(0,200,50))


    ax.set_title(f"Final mse {fmse:.3f}, R2={r2:.3f}")
    return fig, ax

def plot_scatter_log(y_meas: np.ndarray, y_corr: torch.tensor, data: Dataset) -> tuple[Figure, Axes]:
    """
    Plots the logarithmic-scale scatterplot of measurements vs prediction.
    :param y_meas: measurements vector, np.ndarray.
    :param y_corr: predicted concentrations, torch.tensor.
    :param data: dataset for specific setting, either "etex", "chernobyl", or "ruthenium".

    Returns the figure and axes object.
    """
    y_corr_np = y_corr.detach().cpu().numpy().reshape(-1)
    y_meas_np = y_meas.copy().reshape(-1)

    if data == 'etex':
        tresh = 1e-8
    elif data == "chernobyl":
        tresh = 1e-11
    elif data == "ruthenium":
        tresh = 1e-8

    nonz_meas = y_meas_np.copy()
    nonz_meas[nonz_meas <= tresh] = tresh
    nonz_corr = y_corr_np.copy()
    nonz_corr[nonz_corr <= tresh] = tresh

    fig, ax = plt.subplots(figsize=(4, 4))

    x_min, x_max = np.min(np.concatenate((nonz_corr, nonz_meas))), np.max(np.concatenate((nonz_corr, nonz_meas)))
    line_x = np.array([x_min/2, x_max+x_max])
    line_y = line_x
    ax.plot(line_x, line_y, color='black', linewidth = 1)

    ax.scatter(nonz_meas, nonz_corr, marker = 'o', s = 70, color = 'none', edgecolor = 'C0')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_aspect('equal')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if data == "etex":
        ax.set_xlabel(r"$\text{Measurements }(\text{ng}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_ylabel(r"$\text{Predictions }(\text{ng}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_xlim((1e-8-3e-9, 14))
        ax.set_ylim((1e-8-3e-9, 14))
        ax.set_xticks([1e-8, 1e-5, 1e-2, 1e1])
        ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
    elif data == "chernobyl":
        ax.set_xlabel(r"$\text{Measurements }(\mu\text{Bq}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_ylabel(r"$\text{Predictions }(\mu\text{Bq}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_xlim((1e-11-3e-12, 1e3))
        ax.set_ylim((1e-11-3e-12, 1e3))
        ax.set_yticks([1e-11, 1e-7, 1e-3, 1e0, 1e3])
        ax.set_xticks([1e-11, 1e-7, 1e-3, 1e0,1e3])
    elif data == "ruthenium":
        ax.set_xlabel(r"$\text{Measurements }(\text{mBq}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_ylabel(r"$\text{Predictions }(\text{mBq}\cdot\text{m}^{-3})$", fontsize=13)
        ax.set_xlim((1e-8-3e-9, 1e2*3))
        ax.set_ylim((1e-8-3e-9, 1e2*3))
        ax.set_yticks([1e-8, 1e-3, 1e2])
        ax.set_xticks([1e-8, 1e-3, 1e2])

    ax.set_title(f"Log-scale")
    return fig, ax

def multi_L_curve(ls: np.ndarray, ss: np.ndarray, mses: np.ndarray, mxhs: np.ndarray, data: Dataset) -> tuple[Figure, Axes]:
    """
    Creates multi-L-curve plot por a grid of lengthscale and outputscale valus.
    :param ls: vector of lengthscales, it is expected to be sorted in increasing order.
    :param ss: vector of outputscales, it is expected to be sorted in increasing order.
    :param mses: matrix of MSE values for the grid, lengthscale is fixed for rows, outputscale is fixed for columns, the order is the same as in ls and ss.
    :param mxhs: matrix of maximal absolute shift values for the grid, lengthscale is fixed for rows, outputscale is fixed for columns, the order is the same as in ls and ss.

    Returns the figure and axes object.
    """

    fig, ax = plt.subplots(1,1,figsize=(12,5))
    for (il, l) in enumerate(ls):
        line, = ax.plot(mses[il, :], mxhs[il,:], linestyle='--', alpha=0.5, linewidth=1.0)
        ax.plot(mses[il, :], mxhs[il,:], linestyle='None', marker='x', color=line.get_color(), alpha=1.0, label = f"l={l}°/{l*6:.2f}hr", markersize = 9)

    ax.set_ylabel("Maximal shift", fontsize=15)
    ax.set_xlabel("MSE", fontsize=15)
    ax.legend(fontsize=14)
    ax.set_title("Multi-L-curve plot", fontsize=17)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5)
    if data == "etex":
        ytick_positions = ax.get_yticks()
        ax.set_yticks(ytick_positions)
    if data == "chernobyl":
        ax.set_ylim(0, 8.1)
        ax.set_xlim(right=155)
        ax.set_xticks([135, 140, 145, 150])
        ytick_positions = [1, 2, 3, 4, 5, 6, 7, 8]
        ax.set_yticks(ytick_positions)
    elif data == "ruthenium":
        ax.set_ylim((0.0001, 7))
        ytick_positions = ax.get_yticks()
        ax.set_yticks(ytick_positions)

    custom_labels = [f"{y*0.5:.1f}°/{y*3:.0f}hr" for y in ytick_positions]
    ax.set_yticklabels(custom_labels)
    ax.tick_params(axis='both', labelsize=13)

    return fig, ax

def plot_predicted_shifts(inp_data: InputData, tm_int: np.ndarray, lon_min: float, lon_max: float, lat_min: float, lat_max: float, 
                          xstar: np.ndarray, hs: np.ndarray, ht: np.ndarray, scale: float = 1.0) -> tuple[Figure, Axes]:
    """
    Create a plot of predicted shifts over the domain of interest.
    :param inp_data: InputData for ETEX experiment.
    :param tm_int: time interval of interest.
    :param lat_min: smallest latitude of the domain.
    :param lat_max: largest latitude of the domain.
    :param lon_min: smallest longitude of the domain.
    :param lon_max: largest longitude of the domain.
    :param xstar: spatio-temporal coordinates of the grid.
    :param hs: predicted shifts for coordinates in xstar. First column is horizontal shift, second vertical, third and fourth start and end of the interval.
    :param ht: estimated time shift in the locations of stations.
    :param scale: scaling coefficient that scales the shifts to make them visible.

    Returns the Figure and Axes objects. 
    """
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(1,1, figsize = (10,10), subplot_kw={'projection': projection})

    #idices of predictions of the desired timestep
    tmind = np.where((xstar[:, 2] == tm_int[0]) & (xstar[:, 3] == tm_int[1]))[0]

    #find the active stations
    alllonlats = np.stack((inp_data.longitudes, inp_data.latitudes), axis=1)
    valid_lonlats = alllonlats[(inp_data.y > 0) & (inp_data.times[:, 0] == tm_int[0]) & (inp_data.times[:, 1] == tm_int[1])]
    stations = np.unique(valid_lonlats, axis=0)
    tm_max = np.max([np.abs(hs[:, 2:3]).max(), np.abs(ht).max()])

    #plot the shifts, modifies ax
    map_plot(ax, tmind, hs*scale, tm_max, lon_min, lon_max, lat_min, lat_max, xstar[:,0], xstar[:,1], stations)

    #create the legend
    N = 256
    alphas = np.linspace(0.2, 1.0, N)
    blues = np.zeros((N, 4))
    blues[:, 1] = 6/225  # Blue channel
    blues[:, 2] = 117/225  # Blue channel
    blues[:, 3] = alphas  # Alpha channel
    blue_cmap = ListedColormap(blues)

    greens = np.zeros((N, 4))
    greens[:, 1] = 105/255  # Green channel
    greens[:, 2] = 19/255  # Green channel
    greens[:, 3] = alphas
    green_cmap = ListedColormap(greens)

    norm = Normalize(vmin=0, vmax=1)
    sm_blue = cm.ScalarMappable(norm=norm, cmap=blue_cmap)
    sm_green = cm.ScalarMappable(norm=norm, cmap=green_cmap)
    sm_blue.set_array([])
    sm_green.set_array([])

    cax_blue = fig.add_axes([0.3, 0.1, 0.15, 0.01])   # [x0, y0, width, height]
    cax_green = fig.add_axes([0.55, 0.1, 0.15, 0.01])

    cbar1 = fig.colorbar(sm_blue, cax = cax_blue, orientation='horizontal', label='Positive |time shift|')
    cbar2 = fig.colorbar(sm_green, cax = cax_green, orientation='horizontal', label='Negative |time shift|')

    cbar1.ax.xaxis.label.set_size(13) 
    cbar2.ax.xaxis.label.set_size(13)
    cbar1.ax.xaxis.label.set_size(12)
    cbar2.ax.xaxis.label.set_size(12)

    fig.subplots_adjust(bottom=0.2)

    return fig, ax


def map_plot(ax: Axes, tmind: np.ndarray, hs:np.ndarray, time_max:float, lon_min:float, lon_max:float, lat_min:float, lat_max:float, 
             longitudes: np.ndarray, latitudes: np.ndarray, stations: np.ndarray):
    """
    Plots the shifts into provided Axes object.
    :param ax: the axes object where shifts should be plotted.
    :param tmind: vector of indiced of nonzero measurements in the correct time interval.
    :param hs: predicted shifts for coordinates in xstar. First column is horizontal shift, second vertical, third and fourth start and end of the interval.
    :param time_max: largest absolute shift in time axis.
    :param lat_min: smallest latitude of the domain.
    :param lat_max: largest latitude of the domain.
    :param lon_min: smallest longitude of the domain.
    :param lon_max: largest longitude of the domain.
    :param longitudes: longitudes of the grid.
    :param latitudes: latitudes of the grid.
    :param stations: locations of the active stations, first column is longitude, second latitude.
    """
    hs_i = hs[:, tmind].T
    
    colors = []
    alphas = []
    timeshifts = hs_i[:, 2]/time_max
    for val in timeshifts.flatten():
        # Blue for positive, green for negative
        if val >= 0:
            base_color = np.array([0, 6/255, 177/255])  # Blue RGB
        else:
            base_color = np.array([0, 105/255, 19/255])  # Green RGB

        # Alpha: stronger if further from zero
        alpha = np.min((0.2 + abs(val/5*4), 1.0))  # abs in [0, 1]

        rgba = np.concatenate([base_color, [alpha]])
        colors.append(rgba)
        alphas.append(alpha)

    colors = np.array(colors)

    # Add map features
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
    ax.set_extent([lon_min - 0.5, lon_max + 0.5,
                        lat_min - 0.5, lat_max + 0.5])
    ax.set_aspect('equal')
    # Set ticks
    ax.set_xticks(np.arange(lon_min, lon_max+1, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max+1, 5), crs=ccrs.PlateCarree())

    # Format ticks with degree symbols
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # Quivers
    quiv = ax.quiver(longitudes[tmind], latitudes[tmind], hs_i[:, 0], hs_i[:, 1], 
                angles='xy', scale_units='xy', scale=1, width=0.005, color=colors, transform=ccrs.PlateCarree())
    
    quiv.set_path_effects([pe.Stroke(linewidth=0.1, foreground='black'), pe.Normal()])
    
    ax.plot(stations[:, 0], stations[:, 1], 'o', color="black", linestyle="None", markersize=3, label="Station", transform=ccrs.PlateCarree())

def concentration_plot_etex(inp_data: InputData, y_pred_grid: np.ndarray, tm_int: np.ndarray, y_pred_stations: np.ndarray, 
                            lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> tuple[Figure, Axes]:
    """
    Create a plot of concentration over a domain of interest with ETEX specific settings.
    :param inp_data: InputData for ETEX experiment.
    :param y_pred_grid: predicted concetrations on the grid (matrix).
    :param tm_int: time interval of interest.
    :param y_pred_stations: concentrations predicted at the locations of the stations.
    :param lat_min: smallest latitude of the domain.
    :param lat_max: largest latitude of the domain.
    :param lon_min: smallest longitude of the domain.
    :param lon_max: largest longitude of the domain.

    Returns the Figure and Axes objects. 
    """
    fig, ax = plt.subplots(1,1,figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})

    lat_center = np.arange(lat_min, lat_max+0.5, 0.5)
    lon_center = np.arange(lon_min, lon_max+0.5, 0.5)
    
    #colormap borders
    vmin = y_pred_grid.min()
    vmax = y_pred_grid.max()

    #alpha colormap
    base = cm.get_cmap("cool")
    colors = base(np.linspace(0, 1, 256))
    x = np.linspace(0.0, 1.0, 256)
    gamma = 0.5                
    colors[:, -1] = x**gamma
    alpha_cmap = ListedColormap(colors)

    #active stations
    alllonlats = np.stack((inp_data.longitudes, inp_data.latitudes), axis=1)
    inds_valid =  (inp_data.y > 0) & (inp_data.times[:, 0] == tm_int[0]) & (inp_data.times[:, 1] == tm_int[1])
    valid_lonlats = alllonlats[inds_valid]

    #errors
    err_res = np.abs(y_pred_stations.reshape(-1) - inp_data.y.reshape(-1))
    bounds = [-np.inf, 0.5, 2.1, 4.0, np.inf]
    cmap = ListedColormap(["black", "green", "orange", "red"])
    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=False)

    #create the maps
    ax.coastlines()
    ax.set_global()

    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)

    ax.set_extent([lon_min+0.5, lon_max-0.5, lat_min+0.5, lat_max-0.5])
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(lon_min, lon_max, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max, 5), crs=ccrs.PlateCarree())

    # Format ticks with degree symbols
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    #plot the concentration colormesh
    yy = np.flipud(y_pred_grid) #y_pred_grid needs to be flipped
    yy[yy<0] = 0
    mesh = ax.pcolormesh(
        lon_center[:-1]+0.25,
        lat_center[:-1]+0.25,
        yy,
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=alpha_cmap
    )

    #plot stations with errors
    sizes = np.where(err_res[inds_valid] < 0.5, 5, 80) #sizes of errors
    ax.scatter(valid_lonlats[:, 0], valid_lonlats[:, 1], s=sizes, c=err_res[inds_valid], cmap=cmap, norm=norm, 
        edgecolor = 'black', transform=ccrs.PlateCarree(), zorder=3)

    #colorbar for colormesh
    cax = fig.add_axes([0.9, 0.42, 0.02, 0.3])
    cbar = fig.colorbar(
        mesh,
        #ax=axes,
        cax=cax,
        orientation="vertical",
        pad=0.02,
        shrink=0.7
    )
    cbar.set_label(r"$\text{Concentration }\text{ng} \,\cdot \text{m}^{-3}$", fontsize=14)
    cbar.ax.xaxis.label.set_size(14)

    #legend for errors
    handles = [
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='red',    markeredgecolor='black', label=r'$> 4\text{ng} \,\cdot \text{m}^{-3}$', markersize=10),
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='orange', markeredgecolor='black', label=r'$> 2.1\text{ng} \,\cdot \text{m}^{-3}$', markersize=10),
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='green',  markeredgecolor='black', label=r'$> 0.5\text{ng} \,\cdot \text{m}^{-3}$', markersize=10),
        Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='black',  markeredgecolor='black', label=r'$\leq 0.5\text{ng} \,\cdot \text{m}^{-3}$', markersize=5),
    ]
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.1), frameon=True, title = "Absolute error", fontsize=14)
    leg.get_title().set_fontsize(14)

    fig.subplots_adjust(top=0.9, bottom=0.1, hspace = 0.4, wspace = 0.5, right = 0.8)

    return fig, ax
