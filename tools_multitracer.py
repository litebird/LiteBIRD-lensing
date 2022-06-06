# Module for Multitracers
import numpy as np
import healpy as hp
import pickle
import tqdm
from astropy import units as u

# from cmblensplus/wrap
import basic
import curvedsky as cs

# from cmblensplus/utils
import constant as c
import misctools
import cmb
import quad_func
import delens_func

# from local module
import local


# //// Fixed values //// #

# galaxy survey parameters
def galaxy_distribution( zi, survey=['euc','lss'], zbn={'euc':5,'lss':5}, z0={'euc':.9/np.sqrt(2.),'lss':.311}, nz_b={'euc':1.5,'lss':1.}, sig={'euc':.05,'lss':.05}):
    
    zbin, dndzi, pz = {}, {}, {}

    if zbn['euc']==5:
        zbin['euc'] = np.array([0.,.8,1.5,2.,2.5,6.])
    if zbn['lss']==5:
        zbin['lss'] = np.array([0.,.5,1.,2.,3.,6.])
    if zbn['lss']==6:
        zbin['lss'] = np.array([0.,.5,1.,2.,3.,4.,7.])

    for s in survey:
        dndzi[s] = basic.galaxy.dndz_sf(zi,2.,nz_b[s],z0=z0[s])
        if s=='euc' and zbn['euc']!=5:  zbin[s]  = basic.galaxy.zbin(zbn[s],2.,nz_b[s],z0=z0[s])
        pz[s]    = {zid: basic.galaxy.photoz_error(zi,[zbin[s][zid],zbin[s][zid+1]],sigma=sig[s],zbias=0.) for zid in range(zbn[s])}

    # fractional number density
    frac = {}
    for s in survey:
        frac[s] = {zid: np.sum(dndzi[s]*pz[s][zid])/np.sum(dndzi[s]) for zid in range(zbn[s]) }
    
    return zbin, dndzi, pz, frac



def tracer_list(add_cmb=['klb','ks4'], add_euc=5, add_lss=5, add_cib=True):
    
    # construct list of mass tracers to be combined
    klist = {}

    # store id for cmb lensing maps
    kid = 0
    for k in add_cmb:
        klist[kid] = k
        kid += 1

    # store id for cib maps
    if add_cib: 
        klist[kid] = 'cib'
        kid += 1
        
    # store id for Euclid galaxy maps
    for z in range(add_euc):
        klist[kid] = 'euc'+str(z+1)+'n'+str(add_euc)
        kid += 1

    # store id for Euclid galaxy maps
    for z in range(add_lss):
        klist[kid] = 'lss'+str(z+1)+'n'+str(add_lss)
        kid += 1

    return klist

        
#//// Load analytic spectra and covariance ////#

def tracer_filename(m0,m1):

    return local.data_directory()['mas'] + 'spec/cl'+m0+m1+'.dat'


def read_camb_cls(lmax=2048,lminI=100,return_klist=False,**kwargs):

    klist = tracer_list(**kwargs)
    
    # load cl of mass tracers
    cl = {}    
    for I, m0 in klist.items():
        for J, m1 in klist.items():
            if J<I: continue
            l, cl[m0+m1] = np.loadtxt( tracer_filename(m0,m1) )[:,:lmax+1]

            # remove low-ell CIB
            if m0=='cib' or m1=='cib':
                cl[m0+m1][:lminI] = 1e-20

    if return_klist:
        return l, cl, klist
    else:
        return l, cl
        

def get_covariance_signal(lmax,lmin=1,lminI=100,**kwargs): 
        # signal covariance matrix

        # read camb cls
        l, camb_cls, klist = read_camb_cls(lminI=lminI,return_klist=True,**kwargs)
        nkap = len(klist.keys())

        # form covariance
        Cov = np.zeros((nkap,nkap,lmax+1))
        
        for I, m0 in klist.items():
            for J, m1 in klist.items():
                if J<I: continue
                Cov[I,J,lmin:] = camb_cls[m0+m1][lmin:lmax+1]
                
        # symmetrize
        Cov = np.array( [ Cov[:,:,l] + Cov[:,:,l].T - np.diag(Cov[:,:,l].diagonal()) for l in range(lmax+1) ] ).T
        
        return Cov


def get_spectrum_noise(lmax,lminI=100,nu=353.,return_klist=False,frac=None,**kwargs):
    
    klist = tracer_list(**kwargs)

    l  = np.linspace(0,lmax,lmax+1)    
    nl = {}
    
    #//// prepare reconstruction noise of LB and S4 ////#
    #for experiment in ['litebird','s4']:
    #    obj = local.forecast(experiment)
    #    obj.compute_nlkk()

    if 'klb' in klist.values():
        obj = local.forecast('litebird')
        nl['klb'] = obj.load_nlkk(Lmax=lmax)
    
    if 'ks4' in klist.values():
        obj = local.forecast('s4')
        nl['ks4'] = obj.load_nlkk(Lmax=lmax)

    if 'cib' in klist.values():
        Jysr = c.MJysr2uK(nu)/c.Tcmb
        nI = 2.256e-10
        nl['cib'] = ( nI + .00029989393 * (1./(l[:lmax+1]+1e-30))**(2.17) ) * Jysr**2
        nl['cib'][:lminI] = nl['cib'][lminI]

    for m in klist.values():
        if 'euc' in m:
            if frac is None:
                f = 1./kwargs['add_euc']
            else:
                f = frac['euc'][int(m[3])-1]
            nl[m] = np.ones(lmax+1)*c.ac2rad**2/(30.*f)
        if 'lss' in m:
            if frac is None:
                f = 1./kwargs['add_lss']
            else:
                f = frac['lss'][int(m[3])-1]
            nl[m] = np.ones(lmax+1)*c.ac2rad**2/(40.*f)

    for m in nl.keys():
        nl[m][0] = 0.
    
    if return_klist:
        return nl, klist
    else:
        return nl


def get_covariance_noise(lmax,lminI=100,frac=None,**kwargs):
    
    nl, klist = get_spectrum_noise(lmax,lminI=lminI,return_klist=True,frac=frac,**kwargs)
    nkap = len(klist.keys())

    Ncov = np.zeros((nkap,nkap,lmax+1))

    for I, m in enumerate(nl.keys()):
        Ncov[I,I,:] = nl[m]
 
    return Ncov


class mass_tracer():
    # define object which has parameters and filenames for multitracer analysis
    
    def __init__( self, lmin, lmax, add_cmb=['klb','ks4'], gal_zbn={'euc':5,'lss':5}, add_cib=True ):

        # multipole range of the mass tracer
        self.lmin = lmin
        self.lmax = lmax

        # list of mass tracers
        self.add_cmb = add_cmb
        self.add_euc = gal_zbn['euc']
        self.add_lss = gal_zbn['lss']
        self.add_cib = add_cib
        self.gal_zbn = gal_zbn
        self.klist   = tracer_list(add_cmb=self.add_cmb, add_euc=self.add_euc, add_lss=self.add_lss, add_cib=self.add_cib)
        
        # total number of mass tracer maps
        self.nkap = len(self.klist)
        
        #set directory
        d = local.data_directory()
 
        # kappa alm of each mass tracer
        self.fklm = {}
        for m in self.klist.values():
            self.fklm[m] = [ d['mas'] + 'alm/' + m + '_' + str(rlz) + '.pkl' for rlz in local.ids ]
        
        # kappa alm of combined mass tracer
        self.fwklm = [ d['mas'] + 'alm/wklm_' + str(rlz) + '.pkl' for rlz in local.ids ]
        

    def cov_signal(self):
        
        return get_covariance_signal(self.lmax,lmin=self.lmin,add_euc=self.add_euc,add_lss=self.add_lss)

    def gal_frac(self):
        
        return galaxy_distribution(np.linspace(0,50,1000),zbn=self.gal_zbn)[3]
    
    def cov_noise(self,frac=None):
        
        if frac is None: frac = self.gal_frac()
        
        return get_covariance_noise(self.lmax,frac=frac,add_euc=self.add_euc,add_lss=self.add_lss)


