#!/usr/bin/env python

import numpy as np
# others
import local
import tools_cmb

# define parameters
snmax = 1000
kwargs_ov = {'overwrite':True,'verbose':False}

# Read CMB survey masks
pobj = local.analysis()
cobj = tools_cmb.cmb_map()

# generate noise alms
tools_cmb.compute_cmb_noise(pobj,cobj,snmax,**kwargs_ov)

