#!/usr/bin/env python
# coding: utf-8

import numpy as np
# others
import tools_multitracer as mass
import warnings
warnings.filterwarnings("ignore")

# define parameters
nside = 512        # CMB map resolution

#zbn  = {'euc':5,'lss':5}
snmin, snmax = 251, 500
kwargs_ov = {'overwrite':False,'verbose':True}

# cinv options
kwargs_cinv = {
    'chn':  1, \
    'eps':  [1e-4], \
    'itns': [1000], \
    'ro':   10, \
    'stat': 'status_mass_cinv-2.txt' \
}

#for ctype in ['klb']:
for ctype in ['klb','cib','gal','ext','all']:
    mass.interface(ctype,snmin,snmax,nside=nside,kwargs_ov=kwargs_ov,kwargs_cinv=kwargs_cinv)


