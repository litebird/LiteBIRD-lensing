# Module for Multitracers
import numpy as np
import healpy as hp
import pickle
import tqdm
from astropy import units as u

# from cmblensplus/wrap
import cmblensplus.basic as basic
import cmblensplus.curvedsky as cs

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


def mass_tracer_mask(klist):

    glob = local.analysis()
    
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
        
    return mask


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
    obj = local.analysis() # used for reading kappa noise curve

    if 'klb' in klist.values():
        #nlpp = pickle.load(open(obj.nlkk['klb'],"rb"))
        nl['klb'] = np.zeros(lmax+1)
        nl['klb'][2:] = np.loadtxt(obj.nlkk['klb'],unpack=True)[1][:lmax-1] # Dl^phiphi
        nl['klb'] = (np.pi/2.) * nl['klb'][:lmax+1]
    
    if 'ks4' in klist.values():
        #obj = local.forecast('s4')
        #nl['ks4'] = obj.load_nlkk(Lmax=lmax)
        nl['ks4'] = np.zeros(lmax+1)
        nl['ks4'][2:] = np.loadtxt(obj.nlkk['ks4'],unpack=True)[7][:lmax-1]

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
    
    def __init__( self, lmin, lmax, add_cmb=['klb','ks4'], gal_zbn={'euc':5,'lss':5}, add_cib=True, ktag='' ):

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
        self.fwklm = [ d['mas'] + 'alm/' + '_'.join(filter(None,['wklm',ktag,str(rlz)])) +'.pkl' for rlz in local.ids ]
        
    def cov_signal(self):
        
        return get_covariance_signal(self.lmax,lmin=self.lmin,add_cmb=self.add_cmb,add_euc=self.add_euc,add_lss=self.add_lss,add_cib=self.add_cib)

    def gal_frac(self):
        
        return galaxy_distribution(np.linspace(0,50,1000),zbn=self.gal_zbn)[3]
    
    def cov_noise(self,frac=None):
        
        if frac is None: frac = self.gal_frac()
        
        return get_covariance_noise(self.lmax,frac=frac,add_cmb=self.add_cmb,add_euc=self.add_euc,add_lss=self.add_lss,add_cib=self.add_cib)

    def generate_klm(self,rlz):
        
        Cov  = self.cov_signal()
        Ncov = self.cov_noise()
    
        # read true CMB lensing kappa
        iklm = glob.load_input_kappa(rlz,self.lmax)
    
        # Gaussian signal alms are generated here
        sklm = {}
        glm  = cs.utils.gaussalm(Cov[1:,1:,:],ilm=iklm)
        for I, m in self.klist.items():
            if m in ['klb','ks4']: 
                sklm[m] = glm[0]
            else:
                sklm[m] = glm[I-1]

        # Gaussian noise alms are generated here
        glm  = cs.utils.gaussalm(Ncov)
                
        # observed kappa alms
        oklm = { m: sklm[m]+glm[I] for I, m in self.klist.items() }

        for I, m in self.klist.items():
            pickle.dump((oklm[m]),open(self.fklm[m][rlz],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


    def comb_tracers(self,kmaps,mask,nside=512,**kwargs_cinv):

        print(self.klist)
        
        Cov  = self.cov_signal()
        Ncov = self.cov_noise()
        npix = 12*nside**2

        InvN = np.reshape( np.array( [ mask[m] for m in self.klist.values() ] ),(self.nkap,npix) )
        INls = np.array( [ 1./Ncov[:,:,l].diagonal() for l in range(self.lmax+1) ] ).T
    
        xlm = cs.cninv.cnfilter_kappa(self.nkap,nside,self.lmax,Cov,InvN,kmaps,inl=INls,**kwargs_cinv)
        clm = np.array( [ np.dot(Cov[0,:,l],xlm[:,l,:]) for l in range(self.lmax+1) ] )
        
        return clm
    

def comb_mobj(ctype,lmax):
    
    if ctype=='klb': # LiteBIRD alne
        add_cmb = ['klb']
        gal_zbn = {'euc':0,'lss':0}
        add_cib = False
    
    if ctype=='cib': # LiteBIRD + CIB
        add_cmb = ['klb']
        gal_zbn = {'euc':0,'lss':0}
        add_cib = True
    
    if ctype=='gal': # LiteBIRD + galaxies
        add_cmb = ['klb']
        gal_zbn = {'euc':5,'lss':5}
        add_cib = False
    
    if ctype=='ext': # LiteBIRD + CIB + galaxies
        add_cmb = ['klb']
        gal_zbn = {'euc':5,'lss':5}
        add_cib = True
    
    if ctype=='all': # LiteBIRD + CIB + galaxies + S4
        add_cmb = ['klb','ks4']
        gal_zbn = {'euc':5,'lss':5}
        add_cib = True

    return mass_tracer(lmin=2, lmax=lmax, add_cmb=add_cmb, gal_zbn=gal_zbn, add_cib=add_cib, ktag=ctype)


def interface(ctype,snmin,snmax,nside=512,kwargs_ov={},kwargs_cinv={}):

    # define objects
    glob = local.analysis()
    mobj = comb_mobj(ctype,lmax=2*nside)
    
    # read maximum number of tracers
    nkap = len(mobj.klist.keys())
    
    # npix for mass maps
    npix = hp.nside2npix(nside)

    # load signal and noise covariance of mass tracers
    Cov  = mobj.cov_signal()
    Ncov = mobj.cov_noise()

    # load mask
    mask = mass_tracer_mask(mobj.klist)
    
    # Inverse noise covariance and spectra
    InvN = np.reshape( np.array( [ mask[m] for m in mobj.klist.values() ] ),(nkap,npix) )
    INls = np.array( [ 1./Ncov[:,:,l].diagonal() for l in range(mobj.lmax+1) ] ).T

    # loop over realization
    for rlz in tqdm.tqdm(local.rlz(snmin,snmax),ncols=100,desc='each rlz'):
    
        if not misctools.check_path(mobj.fklm['klb'][rlz],**kwargs_ov): 
            mobj.generate_klm(rlz)
        
        oklm = { m: pickle.load(open(mobj.fklm[m][rlz],"rb")) for I, m in mobj.klist.items() }

        if misctools.check_path(mobj.fwklm[rlz],**kwargs_ov): continue
    
        # observed mass-tracer maps
        kmaps = np.zeros((nkap,npix))
        for I, m in mobj.klist.items():
            kmaps[I,:] = mask[m] * cs.utils.hp_alm2map(nside,mobj.lmax,mobj.lmax,oklm[m])

        # Computing filtered-alms
        #print(np.shape(Cov),np.shape(InvN),nside,np.shape(kmaps),nkap)
        xlm = cs.cninv.cnfilter_kappa(nkap,nside,mobj.lmax,Cov,InvN,kmaps,inl=INls,**kwargs_cinv)
        clm = np.array( [ np.dot(Cov[0,:,l],xlm[:,l,:]) for l in range(mobj.lmax+1) ] )

        pickle.dump( (clm), open(mobj.fwklm[rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )


        
