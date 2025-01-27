#!/usr/bin/env python

import numpy as np
# others
import local
import tools_cmb

# define parameters
snmin, snmax = 1, 500
kwargs_ov = {'overwrite':False,'verbose':False}

# Read CMB survey masks
pobj = local.analysis()
cobj = tools_cmb.cmb_map()

# generate tensor alms
#tools_cmb.compute_cmb_tensor(pobj,cobj,snmax,**kwargs_ov)

#////////// Wiener-filtered E-mode for lensing template //////////#
tools_cmb.compute_wiener_highl(pobj,cobj,snmin,snmax,**kwargs_ov)

#////////// Wiener-filtered B-mode on large scale //////////#
#tools_cmb.compute_wiener_lowl(pobj,cobj,snmax,**kwargs_ov)

