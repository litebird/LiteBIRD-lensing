#!/usr/bin/env python
# coding: utf-8

import numpy as np
import healpy as hp
import pickle
import tqdm
# from cmblensplus/wrap/
import basic
import curvedsky as cs
# from cmblensplus/utils/
import misctools
# others
import local
import tools_multitracer as mass
import warnings
warnings.filterwarnings("ignore")

import time

# define parameters
nside = 512        # CMB map resolution
lmax  = 2*nside     # maximum multipole of alm to be generated
npix = hp.nside2npix(nside)
zbn  = {'euc':5,'lss':5}
snmin, snmax = 701, 1000
kwargs_ov = {'overwrite':False,'verbose':True}

glob  = local.analysis()
mobj  = mass.mass_tracer(lmin=2,lmax=lmax,gal_zbn=zbn)
klist = mobj.klist
nkap  = len(klist.keys())

# ### Read galaxy survey mask

W = {}
W['litebird'] = hp.read_map(glob.wind['litebird'])
for survey in ['euclid','lsst','cib','cmbs4']:
    W[survey] = W['litebird']*hp.read_map(glob.wind[survey])
mask = {}
for m in klist.values():
    if m == 'klb':  mask[m] = W['litebird']
    if m == 'ks4':  mask[m] = W['cmbs4']
    if m == 'cib':  mask[m] = W['cib']
    if 'euc' in m:  mask[m] = W['euclid']
    if 'lss' in m:  mask[m] = W['lsst']


# load cl and nl of mass tracers
Cov  = mobj.cov_signal()
Ncov = mobj.cov_noise()

InvN = np.reshape( np.array( [ mask[m] for m in klist.values() ] ),(nkap,npix) )
INls = np.array( [ 1./Ncov[:,:,l].diagonal() for l in range(lmax+1) ] ).T

# cinv options
kwargs_cinv = {
    'chn':  1, \
    'eps':  [1e-4], \
    'itns': [1000], \
    'ro':   10, \
    'stat': 'status_mass_cinv-2.txt' \
}


for rlz in tqdm.tqdm(local.rlz(snmin,snmax),ncols=100,desc='each rlz'):
    
    if misctools.check_path(mobj.fklm['klb'][rlz],**kwargs_ov): 
        
        oklm = { m: pickle.load(open(mobj.fklm[m][rlz],"rb")) for I, m in klist.items() }
    
    else:
    
        # read true CMB lensing kappa
        iklm = glob.load_input_kappa(rlz,lmax)
    
        # Gaussian signal alms are generated here
        sklm = {}
        glm = cs.utils.gaussalm(Cov[1:,1:,:],ilm=iklm)
        for I, m in klist.items():
            if m in ['klb','ks4']: 
                sklm[m] = glm[0]
            else:
                sklm[m] = glm[I-1]

        # Gaussian noise alms are generated here
        glm  = cs.utils.gaussalm(Ncov)
                
        # observed kappa alms
        oklm = { m: sklm[m]+glm[I] for I, m in klist.items() }

        for I, m in klist.items():
            pickle.dump((oklm[m]),open(mobj.fklm[m][rlz],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    

    if misctools.check_path(mobj.fwklm[rlz],**kwargs_ov): continue
    
    # observed mass-tracer maps
    kmaps = np.zeros((nkap,npix))
    for I, m in klist.items():
        kmaps[I,:] = mask[m] * cs.utils.hp_alm2map(nside,lmax,lmax,oklm[m])

    # Computing filtered-alms
    clm = {}
    
    # LiteBIRD
    print('klb')
    xlm = cs.cninv.cnfilter_kappa(1,nside,lmax,Cov[:1,:1,:],InvN[:1,:],kmaps[:1,:],inl=INls[:1,:],**kwargs_cinv)
    clm['klb'] = np.array( [ np.dot(Cov[0,:1,l],xlm[:1,l,:]) for l in range(lmax+1) ] )

    # LiteBIRD + CIB
    print('cib')
    data = np.delete(kmaps[:3,:],1,0)
    scov = np.delete(np.delete(Cov[:3,:3,:],1,0),0,1)
    invn = np.delete(InvN[:3,:],1,0)
    inls = np.delete(INls[:3,:],1,0)
    xlm = cs.cninv.cnfilter_kappa(2,nside,lmax,scov,invn,data,inl=inls,**kwargs_cinv)
    clm['cib'] = np.array( [ np.dot(scov[0,:,l],xlm[:,l,:]) for l in range(lmax+1) ] )
    
    # LiteBIRD + galaxies
    print('gal')
    data = np.delete( np.delete(kmaps,1,0), 1,0 )
    scov = np.delete(np.delete( np.delete(np.delete(Cov,1,0),0,1), 1,0),0,1)
    invn = np.delete( np.delete(InvN,1,0), 1,0 )
    inls = np.delete( np.delete(INls,1,0), 1,0 )
    xlm = cs.cninv.cnfilter_kappa(nkap-2,nside,lmax,scov,invn,data,inl=inls,**kwargs_cinv)
    clm['gal'] = np.array( [ np.dot(scov[0,:,l],xlm[:,l,:]) for l in range(lmax+1) ] )

    # LiteBIRD + CIB + galaxies
    print('ext')
    data = np.delete(kmaps,1,0)
    scov = np.delete(np.delete(Cov,1,0),0,1)
    invn = np.delete(InvN,1,0)
    inls = np.delete(INls,1,0)
    xlm = cs.cninv.cnfilter_kappa(nkap-1,nside,lmax,scov,invn,data,inl=inls,**kwargs_cinv)
    clm['ext'] = np.array( [ np.dot(scov[0,:,l],xlm[:,l,:]) for l in range(lmax+1) ] )

    # LiteBIRD + CIB + galaxies + S4
    print('all')
    xlm = cs.cninv.cnfilter_kappa(nkap,nside,lmax,Cov,InvN,kmaps,inl=INls,**kwargs_cinv)
    clm['all'] = np.array( [ np.dot(Cov[0,:,l],xlm[:,l,:]) for l in range(lmax+1) ] )

    pickle.dump( (clm['klb'], clm['cib'],clm['gal'],clm['ext'],clm['all']), open(mobj.fwklm[rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )

