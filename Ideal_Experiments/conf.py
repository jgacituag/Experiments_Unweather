#This script run the wrf model. 
import numpy as np
import os 

#############################################################################################
# CONFIGURATION PARAMETERS (in a future version this can be an input to a function)
#############################################################################################

conf = dict()
conf['expname']   = 'TEST'
conf['run_num']   = 2
conf['modelpath'] = '/home/jruiz/share/EXPERIMENTS/EXPERIMENTS_UNWEATHER/comp_model/em_quarter_ss'
conf['datapath']  = '/home/jruiz/share/EXPERIMENTS/EXPERIMENTS_UNWEATHER/DATA/'
conf['nthreads']  = 3 #Number of threads that will be used for running WRF

#Parameters controling the shape of the wind profile.
conf['modify_wind_profile'] = False  #Are we going to modify the original wind profile?
conf['remove_mean_wind']    = False  #True for convection experiments, false for mountain wave experiments.
conf['shear_depth_u']    = 8000.0   #Heigth of no shear (for the u component a linear type shear is assumed)
conf['shear_strength_u'] = 5.0e-3   #Shear in m/s^2 (for the u component)
conf['shear_depth_v']    = 8000.0   #Heigth of no shear (for the v component)
conf['shear_freq_v']     = 0.5*np.pi    #This controls the wind gire, 0.5pi means quarter circle pi means half a circle, 2pi means full circle)
conf['shear_amp_v']      = 0.0      #Maximum V-wind in m/s (for the v component a sine type shear is assumed)

conf['llj_amp']         = 0.0      #Low level jet local maximum (for the u component)
conf['llj_h']           = 1500.0   #Low level jet heigth (for the u component)
conf['llj_width']       = 500.0    #Low level jet width  (for the u component)
conf['llj_dir']         = 360.0    #Wind direction for the LLJ. 

conf['surf_u'] = 0.0   #u-wind component at the surface (only meaningul is some type of surface drag is active)
conf['surf_v'] = 0.0   #v-wind component at the surface (only meaningul is some type of surface drag is active)

#Parameters controling the stability (temperature profile)
conf['modify_stability'] = False          #Are we going to modify the original stability?
conf['stability_factor'] = -1.0           #Factor controling the stability change (tipical range [-1.5 , 1.5] )
conf['stability_factor_height'] = 10000.0 #Height of maximum warming / cooling. (tipical range [2500 , 10000 ] )

#Parameters controling the shape of the potential temperature profile.
conf['modify_theta_profile'] = False  #Are we going to modify the original temperature?
conf['surf_theta']           = 288.0 #Potential temperature at the lowest model level.
#Temperature profile will be computed as costant d(theta)/dz in user defined layers. 
conf['theta_layer_limits']  = np.array([0.0,3000.0,15000.0])  #Layer limits.
conf['dthetadz_layer']      = np.array([4.0,4.0,4.0])/1000.0 #dthetadz corresponding to each layer.

#Parameters controling the shape of the moisture profile.
conf['modify_moisture_profile'] = False       #Are we going to modify the original moisture profile?
conf['dry_run']                 = False       #Assume 0.0 moisture content at all levels?
conf['low_level_moisture_height'] = 2000.0    #Low level moisture modification will take effect below this level.
conf['low_level_moisture_mult_factor'] = 0.0  #Moisture modification for low levels (tipical range [-15.0 , 15.0] )   
conf['mid_level_moisture_height'] = 2000.0    #Mid level moisture modification will take effect above this level.
conf['mid_level_moisture_mult_factor'] = 0.0  #Moisture modification factor for mid levels (tipical range [-10.0 , 10.0 ] )

#Parameters controling the model parameters (parameters controlling the model integration)
conf['model_xdomain'] = 2000.0 * 42  #Do not change (total extension of the domain in the x direction, meters)
conf['model_nx'] = 42               #Number of grid points in the X direction (integer) 
conf['model_dx'] = conf['model_xdomain'] / conf['model_nx']  #Model resolution is computed from the number of grid points to keep the model domain unchanged (meters) 
conf['model_dt_fract_num'] = 0       #Numerator for fractional time step (from 1 to 10)
conf['model_dt_fract_den'] = 1       #Denominator for fractional time step (for the moment we kept this equal to 10)
#From the WRF documentation> Example, if you want to use 60.3 sec as your time step,
#                            set model_dt = 60, 
#                            model_dt_fract_num = 3, and 
#                            model_dt_fract_den = 10
conf['model_dt'] = 12                #Integration time step (seconds, integer)            
conf['model_nz'] = 41                #Number of vertical levels (integer)
conf['model_dy'] = conf['model_dx']  #Model resolution in the y-direction (for the moment we kept this equal to the resolution in the x-direction). 
conf['model_ny'] = conf['model_nx']  #Number of grid points in the y direction (for the moment we keep this constant). 
#Other possible parameters (difussion, etc)



#Parameters controlling the sensitivity function to be used
conf['sf_type'] = 'mean_precipitation' 
conf['sf_percentile'] = 99.0 
conf['sf_inilev'] = -1
conf['sf_endlev'] = -1
conf['sf_initime'] = -1
conf['sf_endtime'] = -1

conf['run_model'] = True  #Are we going to run the model?
conf['plot_exp']  = True  #Are we going to do detailed plots of the model solution?

conf['plot_types'] = ['ref_vertvel','init_sounding']  #Plot type. 








