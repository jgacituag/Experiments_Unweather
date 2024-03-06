#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
from wrf import (getvar, to_np, interplevel, extract_times, ALL_TIMES)
import pandas as pd
import datetime as dt

def get_vertical_velocity_series(ncfile):
    data = to_np(getvar(ncfile, "wa", units='ms-1', timeidx=ALL_TIMES, method="cat"))
    dxy = np.nanmin(ncfile.variables['DX2D'][:])/1000
    times = ncfile.variables['XTIME'][:]
    z = to_np(getvar(ncfile, "height", units='km', timeidx=ALL_TIMES, method="cat"))
    
    p90 = np.percentile(data, 90)
    p10 = np.percentile(data, 10)
    pos = np.nanmean(data[np.where(data>0)])
    neg = np.nanmean(data[np.where(data<0)])
    
    idx_90p = np.unravel_index(np.argmin(np.abs(data - p90)), np.abs(data - p90).shape)
    idx_10p = np.unravel_index(np.argmin(np.abs(data - p10)), np.abs(data - p10).shape)
    idx_pos = np.unravel_index(np.argmin(np.abs(data - pos)), np.abs(data - pos).shape)
    idx_neg = np.unravel_index(np.argmin(np.abs(data - neg)), np.abs(data - neg).shape)
    
    data_arr = [["W","P90",     round(p90, 3),int(times[idx_90p[0]]),int(dxy*idx_90p[3]),int(dxy*idx_90p[2]),int(z[idx_90p])],
                ["W","P10",     round(p10, 3),int(times[idx_10p[0]]),int(dxy*idx_10p[3]),int(dxy*idx_10p[2]),int(z[idx_10p])],
                ["W","mean W>0",round(pos, 3),int(times[idx_pos[0]]),int(dxy*idx_pos[3]),int(dxy*idx_pos[2]),int(z[idx_pos])],
                ["W","mean W<0",round(neg, 3),int(times[idx_neg[0]]),int(dxy*idx_neg[3]),int(dxy*idx_neg[2]),int(z[idx_neg])]]
    df = pd.DataFrame(data_arr,
                      columns=["variable","Aggr","value ","Times [min]","x [km]","y [km]","z [km]"])
    
    df_series = pd.DataFrame({"W [max]":np.nanmax(data,axis=(1,2,3))})
    return df,df_series

def get_reflectivity_series(ncfile):
    data = to_np(getvar(ncfile, "dbz", timeidx=ALL_TIMES, method="cat"))
    dxy = np.nanmin(ncfile.variables['DX2D'][:])/1000
    times = ncfile.variables['XTIME'][:]
    z = to_np(getvar(ncfile, "height", units='km', timeidx=ALL_TIMES, method="cat"))
    
    p90 = np.percentile(data, 90)
    pos = np.nanmean(data[np.where(data>0)])
    mxp = np.nanmax(data[np.where(data>0)])

    idx_90p = np.unravel_index(np.argmin(np.abs(data - p90)), np.abs(data - p90).shape)
    idx_pos = np.unravel_index(np.argmin(np.abs(data - pos)), np.abs(data - pos).shape)
    idx_mxp = np.unravel_index(np.argmin(np.abs(data - mxp)), np.abs(data - mxp).shape)
    
    data_arr = [["Reflectivity","P90",     round(p90, 3),int(times[idx_90p[0]]),int(dxy*idx_90p[3]),int(dxy*idx_90p[2]),int(z[idx_90p])],
                ["Reflectivity","mean r>0",round(pos, 3),int(times[idx_pos[0]]),int(dxy*idx_pos[3]),int(dxy*idx_pos[2]),int(z[idx_pos])],
                ["Reflectivity","max r>0", round(mxp, 3),int(times[idx_mxp[0]]),int(dxy*idx_mxp[3]),int(dxy*idx_mxp[2]),int(z[idx_mxp])]]
    df = pd.DataFrame(data_arr,
                      columns=["variable","Aggr","value ","Times [min]","x [km]","y [km]","z [km]"])

    df_series = pd.DataFrame({"Reflectivity [max]":np.nanmax(data,axis=(1,2,3))})
    return df,df_series

def get_precipitation_series(ncfile):
    data = to_np(getvar(ncfile, "RAINNC", timeidx=ALL_TIMES, method="cat"))
    dxy = np.nanmin(ncfile.variables['DX2D'][:])/1000
    times = ncfile.variables['XTIME'][:]
    
    p90 = np.percentile(data, 90)
    pos = np.nanmean(data[np.where(data>0)])
    mxp = np.nanmax(data[np.where(data>0)])

    idx_90p = np.unravel_index(np.argmin(np.abs(data - p90)), np.abs(data - p90).shape)
    idx_pos = np.unravel_index(np.argmin(np.abs(data - pos)), np.abs(data - pos).shape)
    idx_mxp = np.unravel_index(np.argmin(np.abs(data - mxp)), np.abs(data - mxp).shape)
    
    data_arr = [["Precipitation","P90",     round(p90, 3),int(times[idx_90p[0]]),int(dxy*idx_90p[2]),int(dxy*idx_90p[1])],
                ["Precipitation","mean r>0",round(pos, 3),int(times[idx_pos[0]]),int(dxy*idx_pos[2]),int(dxy*idx_pos[1])],
                ["Precipitation","max r>0", round(mxp, 3),int(times[idx_mxp[0]]),int(dxy*idx_mxp[2]),int(dxy*idx_mxp[1])]]
    df = pd.DataFrame(data_arr,
                      columns=["variable","Aggr","value ","Times [min]","x [km]","y [km]"])
    
    df_series = pd.DataFrame({"P [max]":np.nanmax(data,axis=(1,2))})
    return df,df_series


#def calc_w_stats(w):
#
#    return stats
#check updraft helicity
path = "/mnt/Storage/EXPERIMENTS_UNWEATHER/DATA/TEST_Multi_4_vars/000/"
# Load WRF data
ncfile = Dataset(path+'wrfout_000.nc')
w_df,w_df_series = get_vertical_velocity_series(ncfile)
r_df,r_df_series = get_reflectivity_series(ncfile)
p_df,p_df_series = get_precipitation_series(ncfile)

df = pd.concat([w_df,r_df,p_df])
df.to_csv('wrf_results.csv', index=False)

df_series = pd.concat([w_df_series,r_df_series,p_df_series])
df_series.to_csv('wrf_results_series.csv', index=True)