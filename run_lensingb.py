#!/usr/bin/env python

import numpy as np
import healpy as hp
import pickle
import tqdm
# from cmblensplus/wrap/
import curvedsky as cs
# from cmblensplus/utils/
import cmb as CMB
import misctools
import constant as c
# others
import local
import tools_cmb
import tools_multitracer as mass
import tools_delens
import warnings
warnings.filterwarnings("ignore")

# define parameters
snmax = 1000
kwargs_ov = {'overwrite':False,'verbose':True}
#kwargs_ov = {'overwrite':True,'verbose':True}

#method = 'bonly'
method = 'cinv'

# fixed/derived parameters
lmax = 2*512
pobj = local.analysis()
cobj = tools_cmb.cmb_map()
mobj  = mass.mass_tracer(lmin=2,lmax=lmax,gal_zbn={'euc':5,'lss':5})
dobj = tools_delens.filename(method=method)

tools_delens.compute_clbb(cobj,mobj,dobj,snmax)

