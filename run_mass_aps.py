#!/usr/bin/env python
# coding: utf-8

import numpy as np, basic, cosmology, local, tools_multitracer as mass, camb, os
from camb import model, initialpower
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

zmin, zmax = 0.0001, 50.
zn  = 10000
zi  = np.linspace(zmin,zmax,zn)
dz  = zi[1]-zi[0]
Hzi = basic.cosmofuncs.hubble(zi,divc=True,**local.cps)
rzi = basic.cosmofuncs.dist_comoving(zi,**local.cps)


# ### Setup survey parameters
# CIB
#nu = 535.
nu = 353.

zbn = {'euc':5,'lss':5}
bz  = {'euc':np.sqrt(1.+zi),'lss':1+.84*zi}
zbin, dndzi, pz, frac = mass.galaxy_distribution(zi,zbn=zbn)


#### Compute CAMB weight ####
w = {}

# CIB
w['W1'] = cosmology.window_cib(rzi,zi,nu)/Hzi

# galaxy
N = 2
for s in ['euc','lss']:
    for zid in range(zbn[s]):
        w['W'+str(N)] = dndzi[s]*pz[s][zid]
        N += 1

# ### Compute Cl
lmax = 3000

pars = camb.CAMBparams()
pars.set_cosmology(H0=local.H0, ombh2=local.ombh2, omch2=local.omch2)
pars.InitPower.set_params(As=local.As, ns=local.ns)
pars.set_for_lmax(lmax, lens_potential_accuracy=5)
#set Want_CMB to true if you also want CMB spectra or correlations
pars.Want_CMB = False
#NonLinear_both or NonLinear_lens will use non-linear corrections
pars.NonLinear = model.NonLinear_both
pars.Accuracy.AccuracyBoost = 3
#pars.Accuracy.AccuracyBoost = 1
pars.Accuracy.lSampleBoost = 2
#pars.Accuracy.lSampleBoost = 1
#pars.Accuracy.lAccuracyBoost = 3
#Set up W(z) window functions
tracers = [ SplinedSourceWindow( z=zi, W=w['W1'], dlog10Ndm=.4, bias=np.sum(w['W1']*dz) ) ]
for I, m in enumerate(list(w)): # add galaxies
    if I==0:  continue
    if I >= 1 and I < zbn['euc']+1: s = 'euc'
    if I >= zbn['euc']+1 and I < zbn['euc']+zbn['lss']+1: s = 'lss'
    tracers += [ SplinedSourceWindow( z=zi, W=w[m], dlog10Ndm=0, bias_z=bz[s] ) ]
pars.SourceWindows = tracers

print(pars.Accuracy)
#print(pars.SourceWindows)

# turning off GR corrections as they impact on large-scale (ell<20) for galaxy and should not present in CIB
pars.SourceTerms.counts_redshift = False 
pars.SourceTerms.counts_velocity = False
pars.SourceTerms.counts_timedelay = False
pars.SourceTerms.counts_ISW = False
pars.SourceTerms.counts_potential = False

results = camb.get_results(pars)
cls = results.get_source_cls_dict()

klist = mass.tracer_list(add_euc=zbn['euc'], add_lss=zbn['lss'])

camb_list = np.concatenate((np.array(['P']),np.array(['P']),np.array(list(w))))

# factor corrections
l = np.linspace(0,lmax,lmax+1)
camb_cls = {}
for I, m0 in enumerate(camb_list):
    for J, m1 in enumerate(camb_list):
        if J<I: continue
        if m0 == 'P' and m1 == 'P':
            fac   = 2*np.pi/4.
        elif m0 == 'P' and m1 != 'P':
            fac   = 2*np.pi/np.sqrt((l+1e-30)*(l+1))/2.
        else:
            fac   = 2*np.pi/(l+1e-30)/(l+1)
        camb_cls[m0+m1] = cls[m0+'x'+m1][:lmax+1]*fac


# ### Save to files
for I, m0 in enumerate(camb_list):
    for J, m1 in enumerate(camb_list):
        if J<I: continue
        fspec = mass.tracer_filename(klist[I],klist[J])
        np.savetxt(fspec,np.array((l,camb_cls[m0+m1])))

