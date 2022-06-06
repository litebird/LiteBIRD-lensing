#///////////////////////////////////////////////////////////////////////////////////////////////////#
# This file is intended to provide parametes, functions, etc, affecting the delensing code globally #
# Set up analysis parameters, filenames, arrays, functions                                          #
#///////////////////////////////////////////////////////////////////////////////////////////////////#

import numpy as np
import healpy as hp
import sys
import pickle

# from cmblensplus/utils/
import curvedsky as cs
import constant as c
import cmb


#////////// Define fixed values //////////#
# Index for realizations e.g. 0001
ids = [str(i).zfill(4) for i in range(-1,1000)]
ids[0] = 'real'

# cosmological parameters
H0    = 67.36
ombh2 = 0.02237
omch2 = 0.12
As    = 2.099e-9
ns    = 0.965
Om    = (omch2+ombh2)/(H0*.01)**2
cps   = {'H0':H0,'Om':Om,'Ov':1-Om,'w0':-1,'wa':0.}

# data directory
def data_directory(root_fg='../data/PTEP_FG/',root_lens='../data/lensing/',root_mass='../data/lensing/multi-tracer/'):
    direct = {}
    direct['fgs']  = root_fg + '/'
    direct['inp']  = '../data/'
    direct['loc']  = root_mass
    direct['cmb']  = root_mass + 'cmb/'
    direct['del']  = root_mass + 'delens/'
    direct['mas']  = root_mass + 'mass/'
    direct['msk']  = root_lens + 'Masks/'
    direct['LOC']  = root_lens + '/'
    return direct

#////////// Input data files and products //////////#

class analysis:
    '''
    Input products to start lensing/delensing analysis
    '''

    def __init__(self,doreal=False,ilmax=5100):

        #//// set parameters ////#

        # input cmb maximum multipole
        self.ilmax  = ilmax

        #//// theory angular power spectra ////#
        # set directory
        d = data_directory()
        
        # input Anto's CMB map
        self.ficmb = [ d['LOC'] + 'S4BIRD/CMB_Lensed_Maps/CMB/cmb_sims_'+x+'.fits' for x in ids ]
        
        # input Anto's phi alm
        self.fiplm = [ d['LOC'] + 'S4BIRD/CMB_Lensed_Maps/MASS/phi_sims_'+x+'.fits' for x in ids ]

        # input cmb cls
        self.fucl = d['LOC'] + 'S4BIRD/CAMB/BBSims_scal_dls.dat'
        self.flcl = d['LOC'] + 'S4BIRD/CAMB/BBSims_lensed_dls.dat'
        self.ftcl = d['LOC'] + 'S4BIRD/CAMB/BBSims_tens_dls.dat'

        # loading theoretical cl
        self.ucl = cmb.read_camb_cls(self.fucl,ftype='scal',output='array')[:,:ilmax+1]
        self.lcl = cmb.read_camb_cls(self.flcl,ftype='lens',output='array')[:,:ilmax+1]
        self.tcl = cmb.read_camb_cls(self.ftcl,ftype='lens',output='array')[:,:ilmax+1]

        #multipole
        self.l  = np.linspace(0,ilmax,ilmax+1)

        #conversion factor from phi to kappa
        self.kL = self.l*(self.l+1)*.5

        #rename cls
        self.uTT = self.ucl[0]
        self.uEE = self.ucl[1]
        self.uTE = self.ucl[2]
        self.lTT = self.lcl[0]
        self.lEE = self.lcl[1]
        self.lBB = self.lcl[2]
        self.lTE = self.lcl[3]

        #kappa cl
        self.pp = self.ucl[3]
        self.kk = self.ucl[3]*self.kL**2

        #//// survey window ////#
        self.wind = {}
        self.wind['FG']       = d['LOC'] + 'FG_mask.fits'
        self.wind['PR2']      = d['msk'] + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
        self.wind['litebird'] = d['LOC'] + 'FG_mask.fits'
        self.wind['euclid']   = d['msk'] + 'euclid.fits'
        self.wind['lsst']     = d['msk'] + 'lsst.fits'
        self.wind['cib']      = d['msk'] + 'cib.fits'
        self.wind['cmbs4']    = d['msk'] + 'cmbs4.fits'
        self.wind['cmbs4_o']  = d['msk'] + 'cmbs4_noFG.fits'
        self.wind['plklens']  = d['msk'] + 'plk_lensing.fits'

        #//// residual FG ////#
        self.ffgs = [d['fgs']+'/output_component_separation_PTEP_v18022021_noise_'+x+'.fits' for x in ids]
        self.clfg = bl = np.loadtxt(d['fgs']+'Cl.txt',unpack=True)[1]/c.Tcmb**2

    
    def load_input_kappa(self,rlz_index,lmax):
        '''
        Read input phi alm and then convert it to kappa alm
        '''
        iplm = hp.read_alm( self.fiplm[rlz_index] )

        # convert to healpix alm convention:
        LMAX = cs.utils.lmpy2lmax(len(iplm))
        iplm = cs.utils.lm_healpy2healpix( iplm, LMAX ) [:lmax+1,:lmax+1]

        # convert to kappa and output
        return  iplm * self.kL[:lmax+1,None]


#////////// Utility functions //////////#

def rlz(snmin,snmax):
    '''
    Array of realization index
    '''
    return np.linspace(snmin,snmax,snmax-snmin+1,dtype=np.int)


class forecast:
    '''
    Simple forecast tools
    '''
    
    def __init__(self,experiment):
        
        # pol noise and beam
        if experiment=='litebird':
            self.sigma = 3.
            self.theta = 30.
            self.lTmax = 3000
            self.rlmin = 100
            self.rlmax = 1024
        if experiment=='s4':
            self.sigma = 1.
            self.theta = 3.
            self.lTmax = 3000
            self.rlmin = 100
            self.rlmax = 4096
        
        self.fnlkk = data_directory()['mas']+'nlkk/'+experiment+'.dat'


    def set_noise_spectrum(self):

        # noise spectrum
        self.nl = cmb.nl_cmb_all(self.rlmax,self.sigma/np.sqrt(2.),self.theta,lTmax=self.lTmax)
        
        # observed cl
        obj = analysis(ilmax=self.rlmax)
        self.lcl = obj.lcl[:4,:]
        self.ocl = self.lcl + self.nl

        
    def compute_nlkk(self,Lmax=2048):
        self.set_noise_spectrum()
        self.nlkk = cs.norm_quad.qall('lens',[True,True,True,True,True,False],Lmax,self.rlmin,self.rlmax,self.lcl,self.ocl,lfac='k')[0]
        np.savetxt(self.fnlkk,self.nlkk.T)

        
    def load_nlkk(self,Lmax=2048):
        return np.loadtxt( self.fnlkk, unpack=True )[5,:Lmax+1]




