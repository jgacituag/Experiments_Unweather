#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
from wrf import (getvar, interplevel, vinterp)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
from netCDF4 import Dataset


cdict = {'blue': [(0.00000000, 0.92549019607843140, 0.92549019607843140),
                  (0.07142857, 0.96470588235294120, 0.96470588235294120),
                  (0.14285714, 0.96470588235294120, 0.96470588235294120),
                  (0.21428571, 0.00000000000000000, 0.00000000000000000),
                  (0.28571429, 0.00000000000000000, 0.00000000000000000),
                  (0.35714286, 0.00000000000000000, 0.00000000000000000),
                  (0.42857143, 0.00000000000000000, 0.00000000000000000),
                  (0.50000000, 0.00000000000000000, 0.00000000000000000),
                  (0.57142857, 0.00000000000000000, 0.00000000000000000),
                  (0.64285714, 0.00000000000000000, 0.00000000000000000),
                  (0.71428571, 0.00000000000000000, 0.00000000000000000),
                  (0.78571429, 0.00000000000000000, 0.00000000000000000),
                  (0.85714286, 1.00000000000000000, 1.00000000000000000),
                  (0.92857143, 0.78823529411764700, 0.78823529411764700),
                  (1.00000000, 0.00000000000000000, 0.00000000000000000)],
         'green':[(0.00000000, 0.92549019607843140, 0.92549019607843140),
                  (0.07142857, 0.62745098039215690, 0.62745098039215690),
                  (0.14285714, 0.00000000000000000, 0.00000000000000000),
                  (0.21428571, 1.00000000000000000, 1.00000000000000000),
                  (0.28571429, 0.78431372549019610, 0.78431372549019610),
                  (0.35714286, 0.56470588235294120, 0.56470588235294120),
                  (0.42857143, 1.00000000000000000, 1.00000000000000000),
                  (0.50000000, 0.75294117647058820, 0.75294117647058820),
                  (0.57142857, 0.56470588235294120, 0.56470588235294120),
                  (0.64285714, 0.00000000000000000, 0.00000000000000000),
                  (0.71428571, 0.00000000000000000, 0.00000000000000000),
                  (0.78571429, 0.00000000000000000, 0.00000000000000000),
                  (0.85714286, 0.00000000000000000, 0.00000000000000000),
                  (0.92857143, 0.33333333333333330, 0.33333333333333330),
                  (1.00000000, 0.00000000000000000, 0.00000000000000000)],
         'red':  [(0.00000000, 0.00000000000000000, 0.00000000000000000),
                  (0.07142857, 0.00392156862745098, 0.00392156862745098),
                  (0.14285714, 0.00000000000000000, 0.00000000000000000),
                  (0.21428571, 0.00000000000000000, 0.00000000000000000),
                  (0.28571429, 0.00000000000000000, 0.00000000000000000),
                  (0.35714286, 0.00000000000000000, 0.00000000000000000),
                  (0.42857143, 1.00000000000000000, 1.00000000000000000),
                  (0.50000000, 0.90588235294117650, 0.90588235294117650),
                  (0.57142857, 1.00000000000000000, 1.00000000000000000),
                  (0.64285714, 1.00000000000000000, 1.00000000000000000),
                  (0.71428571, 0.83921568627450980, 0.83921568627450980),
                  (0.78571429, 0.75294117647058820, 0.75294117647058820),
                  (0.85714286, 1.00000000000000000, 1.00000000000000000),
                  (0.92857143, 0.60000000000000000, 0.60000000000000000),
                  (1.00000000, 0.00000000000000000, 0.00000000000000000)]}

NWSReflectivity = LinearSegmentedColormap('NWSReflectivity', segmentdata=cdict, N=16) # this is just to make a colormap fo the reflectivity using one from pyart, but without installing the library

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',edgecolor='gray')

countries = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='10m',
        facecolor='none',edgecolor='black')

def plot_max_reflectivity(file_in,lonwest=None,loneast=None,latsouth=None,latnorth=None):
    '''
    plot the infered reflectivity from a WRF simulation.

    Parameters:
        file_in (): file dataset previolsuy loader or readed whith NETCDF of xarray packages.
        lonwest  (float): longitude in degrees of west boundary to plot, example -68.5
        loneast  (float): longitude in degrees of east boundary to plot, example -56.8
        latsouth (float): longitude in degrees of south boundary to plot, example -41.2
        latnorth (float): longitude in degrees of north boundary to plot, example -23.0
    '''
    dbz  = getvar(file_in, "mdbz").to_numpy()
    time = getvar(file_in, 'times').to_numpy()
    lats  = getvar(file_in, 'lat').to_numpy()
    lons  = getvar(file_in, 'lon').to_numpy()
    umet, vmet = getvar(file_in, 'uvmet10',units='kt').to_numpy()

    if any(coord is None for coord in [lonwest,loneast,latsouth,latnorth]):
        lonwest  = np.nanmin(lons)
        loneast  = np.nanmax(lons)
        latsouth = np.nanmin(lats)
        latnorth = np.nanmax(lats)
    #ccrs.PlateCarree()
    
    projection = ccrs.PlateCarree()#data_in['proj']

    axes_class = (GeoAxes,dict(projection=projection))
    fig = plt.figure(figsize=(12,12),dpi=200)
    axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(1, 1), axes_pad=1.5,
                  cbar_location='right', cbar_mode='each',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='keep',share_all=True)
    ax=axgr[0]
    vlim_max = 77.5
    vlim_min = -2.5
    
    
    data = np.ma.masked_less(dbz,vlim_min)
    
    ax.set_extent([lonwest, loneast, latsouth, latnorth], crs=projection)
    cm = ax.pcolormesh(lons,lats,data,cmap=NWSReflectivity,vmin=vlim_min,vmax=vlim_max,linewidths=0.5,edgecolors='none',zorder=2,alpha=0.8,transform=projection)
    delta_barbs = 30
    ax.barbs(lons[::delta_barbs,::delta_barbs],
             lats[::delta_barbs,::delta_barbs], 
             umet[::delta_barbs,::delta_barbs], 
             vmet[::delta_barbs,::delta_barbs],
             pivot= 'middle',length=6,transform=projection)
    
    ax.coastlines(resolution='10m',linewidth=0.6)
    ax.add_feature(countries,linewidth=0.4)
    ax.add_feature(states_provinces,linewidth=0.4)
    
    possible_steps = np.arange(0.25,2.1,0.25)
    for step in possible_steps:
        arr_length = len(np.arange(np.ceil(latsouth), np.ceil(latnorth) + step, step))
        if 4 <= arr_length <= 10:
            delta_grad= step
            
    ax.set_yticks(np.arange(np.ceil(latsouth),np.ceil(latnorth),delta_grad), crs=projection)
    ax.set_xticks(np.arange(np.ceil(lonwest), np.ceil(loneast), delta_grad), crs=projection)

    # Le damos formato a las etiquetas de los ticks
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xticklabels(ax.get_xticklabels(),fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=14)
    
    ax.set_xlabel('Longitude [degrees]',fontsize=16)
    ax.set_ylabel('Latitude [degrees]',fontsize=16)
    

    cbar=axgr.cbar_axes[0].colorbar(cm)
    cbar.set_label('Maximum reflectivity [dBZ]',fontsize=14)

    plt.tight_layout()
    date_str = np.datetime_as_string(time)[0:19]
    axgr[0].set_title(date_str,fontsize=16)
    plt.savefig(f'Figuras/MAX_Reflectivity_{date_str}.png',format='png',dpi=200, bbox_inches='tight')
    plt.show()
    plt.close('all')     
    
def plot_reflectivity_level(file_in,lonwest=None,loneast=None,latsouth=None,latnorth=None):
    '''
    plot the infered reflectivity from a WRF simulation.

    Parameters:
        file_in (): file dataset previolsuy loader or readed whith NETCDF of xarray packages.
        lonwest  (float): longitude in degrees of west boundary to plot, example -68.5
        loneast  (float): longitude in degrees of east boundary to plot, example -56.8
        latsouth (float): longitude in degrees of south boundary to plot, example -41.2
        latnorth (float): longitude in degrees of north boundary to plot, example -23.0
    '''
    dbz_init  = getvar(file_in, "dbz").to_numpy()
    time  = getvar(file_in, 'times').to_numpy()
    lats  = getvar(file_in, 'lat').to_numpy()
    lons  = getvar(file_in, 'lon').to_numpy()
    u_init, v_init = getvar(file_in, 'uvmet',units='kt').to_numpy()
    interp_levels = [2,4,6]#np.arange(1,6.1)
    umet_int = vinterp(file_in,   u_init, 'ght_agl', interp_levels)
    vmet_int = vinterp(file_in,   v_init, 'ght_agl', interp_levels)
    dbz  = vinterp(file_in, dbz_init, 'ght_agl', interp_levels)

    if any(coord is None for coord in [lonwest,loneast,latsouth,latnorth]):
        lonwest  = np.nanmin(lons)
        loneast  = np.nanmax(lons)
        latsouth = np.nanmin(lats)
        latnorth = np.nanmax(lats)
    #ccrs.PlateCarree()
    
    projection = ccrs.PlateCarree()#data_in['proj']

    for i in np.arange(np.shape(umet_int)[0]):
        axes_class = (GeoAxes,dict(projection=projection))
        fig = plt.figure(figsize=(12,12),dpi=200)
        axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(1, 1), axes_pad=1.5,
                      cbar_location='right', cbar_mode='each',
                      cbar_pad=0.2,
                      cbar_size='3%',
                      label_mode='keep',share_all=True)
        ax=axgr[0]
        vlim_max = 77.5
        vlim_min = -2.5
        
        umet = umet_int[i]
        vmet = vmet_int[i]
        data = np.ma.masked_less(dbz[i],vlim_min)
        
        ax.set_extent([lonwest, loneast, latsouth, latnorth], crs=projection)
        cm = ax.pcolormesh(lons,lats,data,cmap=NWSReflectivity,vmin=vlim_min,vmax=vlim_max,linewidths=0.5,edgecolors='none',zorder=2,alpha=0.8,transform=projection)
        delta_barbs = 30
        ax.barbs(lons[::delta_barbs,::delta_barbs],
                 lats[::delta_barbs,::delta_barbs], 
                 umet[::delta_barbs,::delta_barbs], 
                 vmet[::delta_barbs,::delta_barbs],
                 pivot= 'middle',length=6,transform=projection)
        
        ax.coastlines(resolution='10m',linewidth=0.6)
        ax.add_feature(countries,linewidth=0.4)
        ax.add_feature(states_provinces,linewidth=0.4)
        
        possible_steps = np.arange(0.25,2.1,0.25)
        for step in possible_steps:
            arr_length = len(np.arange(np.ceil(latsouth), np.ceil(latnorth) + step, step))
            if 4 <= arr_length <= 10:
                delta_grad= step
                
        ax.set_yticks(np.arange(np.ceil(latsouth),np.ceil(latnorth),delta_grad), crs=projection)
        ax.set_xticks(np.arange(np.ceil(lonwest), np.ceil(loneast), delta_grad), crs=projection)
    
        # Le damos formato a las etiquetas de los ticks
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    
        ax.set_xticklabels(ax.get_xticklabels(),fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=14)
        
        ax.set_xlabel('Longitude [degrees]',fontsize=16)
        ax.set_ylabel('Latitude [degrees]',fontsize=16)
        
    
        cbar=axgr.cbar_axes[0].colorbar(cm)
        cbar.set_label('Maximum reflectivity [dBZ]',fontsize=14)
    
        plt.tight_layout()
        date_str = np.datetime_as_string(time)[0:19]
        axgr[0].set_title(date_str+' level ' + str(int(interp_levels[i])) + 'km',fontsize=16)
        plt.savefig(f'Figuras/Reflectivity_{date_str}_{interp_levels[i]}m.png',format='png',dpi=200, bbox_inches='tight')
        plt.show()
        plt.close('all')


def plot_vorticity(file_in,lonwest=None,loneast=None,latsouth=None,latnorth=None):
    '''
    plot the infered reflectivity from a WRF simulation.

    Parameters:
        file_in (): file dataset previolsuy loader or readed whith NETCDF of xarray packages.
        lonwest  (float): longitude in degrees of west boundary to plot, example -68.5
        loneast  (float): longitude in degrees of east boundary to plot, example -56.8
        latsouth (float): longitude in degrees of south boundary to plot, example -41.2
        latnorth (float): longitude in degrees of north boundary to plot, example -23.0
    '''
    avo  = getvar(file_in, "avo").to_numpy()/100
    time = getvar(file_in, 'times').to_numpy()
    lats  = getvar(file_in, 'lat').to_numpy()
    lons  = getvar(file_in, 'lon').to_numpy()
    w  = getvar(file_in, "wa").to_numpy()

    if any(coord is None for coord in [lonwest,loneast,latsouth,latnorth]):
        lonwest  = np.nanmin(lons)
        loneast  = np.nanmax(lons)
        latsouth = np.nanmin(lats)
        latnorth = np.nanmax(lats)
    #ccrs.PlateCarree()
    idx_avo = np.argmax(np.abs(avo),axis=0)
    avo_abs = np.nan*np.ones(np.shape(avo)[1:3])
    for i in np.arange(np.shape(avo)[1]):
        for j in np.arange(np.shape(avo)[2]):
            avo_abs[i,j]=avo[idx_avo[i,j],i,j]

    idx_w = np.argmax(np.abs(w),axis=0)
    w_abs = np.nan*np.ones(np.shape(w)[1:3])
    for i in np.arange(np.shape(w)[1]):
        for j in np.arange(np.shape(w)[2]):
            w_abs[i,j]=w[idx_w[i,j],i,j]
            
    projection = ccrs.PlateCarree()#data_in['proj']

    axes_class = (GeoAxes,dict(projection=projection))
    fig = plt.figure(figsize=(12,12),dpi=200)

    axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(1, 1), axes_pad=1.5,
                  cbar_location='right', cbar_mode='each',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='keep',share_all=True)
    ax=axgr[0]
    vlim_max = 7
    vlim_min = -7
    
    
    #data = np.ma.masked_less(avo,vlim_min)
    
    ax.set_extent([lonwest, loneast, latsouth, latnorth], crs=projection)
    cm = ax.pcolormesh(lons,lats,avo_abs,cmap='RdBu_r',vmin=vlim_min,vmax=vlim_max,linewidths=0.5,edgecolors='none',zorder=2,alpha=0.8,transform=projection)
    cs = ax.contour(lons,lats,w_abs,levels=[-15,-10,-5,5,10,15],linewidths=0.5,zorder=2,alpha=0.8,transform=projection,colors='k')  # Negative contours default to dashed.
    ax.clabel(cs, fontsize=9, inline=True)
    
    ax.coastlines(resolution='10m',linewidth=0.6)
    ax.add_feature(countries,linewidth=0.4)
    ax.add_feature(states_provinces,linewidth=0.4)
    
    possible_steps = np.arange(0.25,2.1,0.25)
    for step in possible_steps:
        arr_length = len(np.arange(np.ceil(latsouth), np.ceil(latnorth) + step, step))
        if 4 <= arr_length <= 10:
            delta_grad= step
            
    ax.set_yticks(np.arange(np.ceil(latsouth),np.ceil(latnorth),delta_grad), crs=projection)
    ax.set_xticks(np.arange(np.ceil(lonwest), np.ceil(loneast), delta_grad), crs=projection)

    # Le damos formato a las etiquetas de los ticks
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xticklabels(ax.get_xticklabels(),fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=14)
    
    ax.set_xlabel('Longitude [degrees]',fontsize=16)
    ax.set_ylabel('Latitude [degrees]',fontsize=16)
    

    cbar=axgr.cbar_axes[0].colorbar(cm)
    cbar.set_label(r'Maximum Vorticity $[1e^{-3}/s]$',fontsize=14)

    plt.tight_layout()
    date_str = np.datetime_as_string(time)[0:19]
    axgr[0].set_title(date_str,fontsize=16)
    plt.savefig(f'Figuras/MAX_vorticity_{date_str}.png',format='png',dpi=200, bbox_inches='tight')
    plt.show()
    plt.close('all')

def plot_vorticity_level(file_in,lonwest=None,loneast=None,latsouth=None,latnorth=None):
    '''
    plot the infered reflectivity from a WRF simulation.

    Parameters:
        file_in (): file dataset previolsuy loader or readed whith NETCDF of xarray packages.
        lonwest  (float): longitude in degrees of west boundary to plot, example -68.5
        loneast  (float): longitude in degrees of east boundary to plot, example -56.8
        latsouth (float): longitude in degrees of south boundary to plot, example -41.2
        latnorth (float): longitude in degrees of north boundary to plot, example -23.0
    '''
    
    time = getvar(file_in, 'times').to_numpy()
    lats  = getvar(file_in, 'lat').to_numpy()
    lons  = getvar(file_in, 'lon').to_numpy()
    avo_init  = getvar(file_in, "avo").to_numpy()/100
    w_init  = getvar(file_in, "wa").to_numpy()

    interp_levels = [2,4,6]
    avo  = vinterp(file_in, avo_init, 'ght_agl', interp_levels)
    w  = vinterp(file_in, w_init, 'ght_agl', interp_levels)
    
    if any(coord is None for coord in [lonwest,loneast,latsouth,latnorth]):
        lonwest  = np.nanmin(lons)
        loneast  = np.nanmax(lons)
        latsouth = np.nanmin(lats)
        latnorth = np.nanmax(lats)
    #ccrs.PlateCarree()
    
    projection = ccrs.PlateCarree()#data_in['proj']

    for i in np.arange(np.shape(avo)[0]):
        axes_class = (GeoAxes,dict(projection=projection))
        fig = plt.figure(figsize=(12,12),dpi=200)
        axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(1, 1), axes_pad=1.5,
                      cbar_location='right', cbar_mode='each',
                      cbar_pad=0.2,
                      cbar_size='3%',
                      label_mode='keep',share_all=True)
        ax=axgr[0]
        vlim_max = 7
        vlim_min = -7
        
        ax.set_extent([lonwest, loneast, latsouth, latnorth], crs=projection)
        cm = ax.pcolormesh(lons,lats,avo[i],cmap='RdBu_r',vmin=vlim_min,vmax=vlim_max,linewidths=0.5,edgecolors='none',zorder=2,alpha=0.8,transform=projection)
        cs = ax.contour(lons,lats,w[i],levels=[-15,-10,-5,5,10,15],linewidths=0.5,zorder=2,alpha=0.8,transform=projection,colors='k')  # Negative contours default to dashed.
        ax.clabel(cs, fontsize=9, inline=True)
        
        ax.coastlines(resolution='10m',linewidth=0.6)
        ax.add_feature(countries,linewidth=0.4)
        ax.add_feature(states_provinces,linewidth=0.4)
        
        possible_steps = np.arange(0.25,2.1,0.25)
        for step in possible_steps:
            arr_length = len(np.arange(np.ceil(latsouth), np.ceil(latnorth) + step, step))
            if 4 <= arr_length <= 10:
                delta_grad= step
                
        ax.set_yticks(np.arange(np.ceil(latsouth),np.ceil(latnorth),delta_grad), crs=projection)
        ax.set_xticks(np.arange(np.ceil(lonwest), np.ceil(loneast), delta_grad), crs=projection)
    
        # Le damos formato a las etiquetas de los ticks
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    
        ax.set_xticklabels(ax.get_xticklabels(),fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=14)
        
        ax.set_xlabel('Longitude [degrees]',fontsize=16)
        ax.set_ylabel('Latitude [degrees]',fontsize=16)
        
    
        cbar=axgr.cbar_axes[0].colorbar(cm)
        cbar.set_label(r'Maximum Vorticity $[1e^{-3}/s]$',fontsize=14)
    
        plt.tight_layout()
        date_str = np.datetime_as_string(time)[0:19]
        axgr[0].set_title(date_str+' level ' + str(int(interp_levels[i])) + 'km',fontsize=16)
        plt.savefig(f'Figuras/Vorticity_{date_str}_{interp_levels[i]}m.png',format='png',dpi=200, bbox_inches='tight')
        plt.show()
        plt.close('all')

def plot_map_axis():
    return None
def plot_map():
    return None
def plot_data():
    return None
if __name__ == "__main__":
    path = '/mnt/Storage/Experimentos/WRF2k_2019/Orig/01/'
    base_fname_list = sorted(glob.glob(path+'wrfout_d01_2019-10-1*'))
    base_fname_list = base_fname_list[2:50]
    
    n = len(base_fname_list)
    for i, base_fname in enumerate(base_fname_list):
        print('#################################################################')
        print(f'plotting data from file {i+1} of {n}')
        dataset = Dataset(base_fname)
    #    #plot_map(dataset,'WRF 2k','QRAIN', vlim_min=0, vlim_max=0.08,sumar=True ,cmap='BrBG')
    #    #plot_map(dataset,'WRF 2k','QVAPOR',vlim_min=0, vlim_max=0.2 ,sumar=True ,cmap='viridis')
        #plot_vorticity(dataset)
        plot_vorticity(dataset,lonwest=-64,loneast=-61,latsouth=-34,latnorth=-32)
        plot_vorticity_level(dataset,lonwest=-64,loneast=-61,latsouth=-34,latnorth=-32)
        plot_reflectivity_level(dataset,lonwest=-64,loneast=-61,latsouth=-34,latnorth=-32)
        plot_max_reflectivity(dataset,lonwest=-64,loneast=-61,latsouth=-34,latnorth=-32)





