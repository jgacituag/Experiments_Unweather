import numpy as np

import wrf_module as wrf
import conf as conf 


#Load the configuration from conf.py
conf = conf.conf 


conf['expname']   = 'TEST_SENS'




conf['run_num']   = 2
conf['surf_u']    = 10.0 


#Call run wrf with the given configuration. 
wrf.run_wrf( conf ) 





