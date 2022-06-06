#!/usr/bin/env python

import numpy as np
# others
import local
import tools_cmb

# define parameters
snmax = 1000
kwargs_ov = {'overwrite':False,'verbose':False}

# Read CMB survey masks
pobj = local.analysis()
cobj = tools_cmb.cmb_map()

# generate noise alms
#tools_cmb.compute_cmb_noise(cobj,snmax,**kwargs_ov)

# generate tensor alms
#tools_cmb.compute_cmb_tensor(pobj,cobj,snmax,**kwargs_ov)

#////////// Wiener-filtered E-mode for lensing template //////////#
#kwargs_ov = {'overwrite':True,'verbose':True}
#tools_cmb.compute_wiener_highl(pobj,cobj,snmax,**kwargs_ov)

#////////// Wiener-filtered B-mode on large scale //////////#
tools_cmb.compute_wiener_lowl(pobj,cobj,snmax,**kwargs_ov)

