# Module for Multitracers
import numpy as np
import healpy as hp
import pickle
import tqdm
import warnings
warnings.filterwarnings("ignore")

import cmblensplus.curvedsky as cs
import constant as c
import misctools
import cmb

# from local module
import local

# //// Fixed values //// #

masks = {'lbs4','lbonly','lbfull','cmball'}

#theta = {'lb':30.,'s4':2.}
#sigma = {'lb':2.,'s4':1.} # uK-arcmin in polarization


# //// Derived products //// #

class cmb_map():
    '''
    Derived products for CMB
    '''
    
    def __init__( self ):

        #set directory
        d = local.data_directory()
 
        # noise alms
        self.fnalm = [ d['cmb'] + 'alm/nalm_' + str(rlz) + '.pkl' for rlz in local.ids ]
        
        # tensor alms
        self.fralm = [ d['cmb'] + 'alm/ralm_' + str(rlz) + '.pkl' for rlz in local.ids ]
        
        # Wiener-filtered CMB E and B modes (used for lensing template)
        self.fwalm = { m: [ d['cmb'] + 'alm/walm_' + m + '_' + str(rlz) + '.pkl' for rlz in local.ids ] for m in masks }

        # Wiener-filtered CMB B-modes on large scale (to be delensed)
        self.foblm = { m: [ d['cmb'] + 'alm/oblm_' + m + '_' + str(rlz) + '.pkl' for rlz in local.ids ] for m in masks }


# //// Utilities //// #

def prepare_masks(nside=None):
    
    params = local.analysis()

    W_LB = hp.read_map(params.wind['litebird'])
    W_S4 = W_LB * hp.read_map(params.wind['cmbs4'])

    mask = {}
    mask['lbs4']   = W_S4
    mask['lbonly'] = W_LB*(1.-W_S4)
    mask['lbfull'] = W_LB
    mask['cmball'] = W_LB
    
    if nside is not None:

        for m in masks:
            mask[m] = hp.ud_grade(mask[m],nside)
            mask[m][mask[m]<1.] = 0.
    
    return mask


def qumap_smoothing(iQ,iU,lmax,nside,bl=None):

    alm = hp.sphtfunc.map2alm(np.array((0*iQ,iQ,iU)), lmax=lmax, pol=True)
    
    # beam smearing
    if bl is not None:
        alm[1] = hp.sphtfunc.almxfl(alm[1,:], bl)
        alm[2] = hp.sphtfunc.almxfl(alm[2,:], bl)
        
    __, Q, U = hp.sphtfunc.alm2map(alm, nside, lmax=lmax, pixwin=True, pol=True)
    
    return Q, U


def prepare_obs_Bmap(pobj,cobj,rlz,maskname,lmax=190,nside=128,method='bonly'):

    #//// compute large-scale observed B-mode////#
    bl = cmb.beam(80.,lmax,inv=False) # 80 arcmin beam to match the PTEP FG sims
    Wl = hp.sphtfunc.pixwin(64,lmax=lmax)
    wl = hp.sphtfunc.pixwin(nside,lmax=lmax)

    # load mask
    mask = prepare_masks(nside=nside)[maskname]

    # residual FG noise
    Qn, Un = hp.read_map(pobj.ffgs[rlz],field=(1,2))/c.Tcmb
    # nan to zero
    Qn[np.isnan(Qn)] = 0
    Un[np.isnan(Un)] = 0
    Qn = hp.ud_grade( Qn, nside )
    Un = hp.ud_grade( Un, nside )
    nBlm = cs.utils.hp_map2alm_spin(lmax,lmax,2,mask*Qn,mask*Un)[1]/(bl[:,None]*Wl[:,None])

    # lensing 
    Qs, Us = hp.read_map(pobj.ficmb[rlz],field=(1,2))/c.Tcmb
    nsides = hp.get_nside(Qs)
    if method == 'bonly': # ignore E-to-B leakage of lensing
        sBlm = cs.utils.hp_map2alm_spin(lmax,lmax,2,Qs,Us)[1]
        Qs, Us = cs.utils.hp_alm2map_spin(nsides,2,0*sBlm,sBlm)
    Qs, Us = qumap_smoothing(Qs,Us,lmax,nside,bl)
    sBlm = cs.utils.hp_map2alm_spin(lmax,lmax,2,mask*Qs,mask*Us)[1]/(bl[:,None]*wl[:,None])

    # tensor
    rBlm = pickle.load(open(cobj.fralm[rlz],"rb"))
    lrmax = len(rBlm[:,0]) - 1
    Qr, Ur = cs.utils.hp_alm2map_spin(nside,2,0*rBlm,rBlm)
    Qr, Ur = qumap_smoothing(Qr,Ur,lmax,nside,bl)
    rElm, rBlm = cs.utils.hp_map2alm_spin(lmax,lmax,2,mask*Qr,mask*Ur)/(bl[:,None]*wl[:,None])

    return sBlm, rBlm, nBlm
    

def compute_cmb_noise(pobj,cobj,snmax,lmax=1024,**kwargs_ov):
    '''
    Generate CMBS4 noise alms
    '''
    
    # S4 noise spectrum
    nls4 = np.zeros((2,lmax+1))
    nls4[:,10:] = np.loadtxt(pobj.nls4,unpack=True,usecols=(1,2))[:,:lmax-9]/cmb.Tcmb**2
    
    # noise covariance
    #Ncov = prepare_cmb_Ncov(lmax)
    Ncov = np.zeros((2,2,lmax+1))
    Ncov[0,0] = nls4[0]
    Ncov[1,1] = nls4[1]

    for rlz in tqdm.tqdm(local.rlz(1,snmax),ncols=100,desc='rlz (cmb noise)'):

        if misctools.check_path(cobj.fnalm[rlz],**kwargs_ov): continue
            
        nlm = cs.utils.gaussalm(Ncov)
        pickle.dump( (nlm), open(cobj.fnalm[rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )


def compute_cmb_tensor(pobj,cobj,snmax,ltmax=200,**kwargs_ov):
    '''
    Generate tensor alms
    '''

    for rlz in tqdm.tqdm(local.rlz(1,snmax),ncols=100,desc='rlz (cmb tensor)'):

        if misctools.check_path(cobj.fralm[rlz],**kwargs_ov): continue
            
        rlm = cs.utils.gauss1alm(ltmax,pobj.tcl[2,:ltmax+1])
        pickle.dump( (rlm), open(cobj.fralm[rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )
        

def inv_aps(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret


def compute_wiener_highl(pobj,cobj,snmin,snmax,nside=512,lmax=1024,**kwargs_ov):
    '''
    Combine LiteBIRD and S4 E-modes
    '''

    # set parameters
    npix = hp.nside2npix(nside)

    # get mask
    Mask = prepare_masks(nside)
    
    # beam
    bl = np.zeros((2,lmax+1))
    bl[0] = cmb.beam(15.,lmax,inv=False) # artificial beam to suppress high-ell LB noise
    bl[1] = cmb.beam(1.,lmax,inv=False)

    # inverse noise spectra
    pobj.load_nl_LB(lmax)
    pobj.load_nl_S4(lmax)

    iNls = np.zeros((2,2,lmax+1))
    
    iNls[0,0,:] = inv_aps(pobj.nEE[:lmax+1]*bl[0]**2)
    iNls[1,0,:] = inv_aps(pobj.nBB[:lmax+1]*bl[0]**2)    
    iNls[0,1,:] = inv_aps(pobj.nls4[0,:lmax+1])
    iNls[1,1,:] = inv_aps(pobj.nls4[1,:lmax+1])
    
    # kwargs for cinv
    kwargs_cinv = {'chn':1,'itns':[1000],'eps':[1e-4],'ro':10,'stat':'status_wiener_highl.txt'}
    
    # loop over realizations
    for rlz in tqdm.tqdm(local.rlz(snmin,snmax),ncols=100,desc='rlz (cmb wiener high-l)'):
        
        # S4 noise
        nlms4  = pickle.load(open(cobj.fnalm[rlz],"rb"))
        nQs4, nUs4 = cs.utils.hp_alm2map_spin(nside,lmax,lmax,2,nlms4[0],nlms4[1])
        
        omap = None

        for m in masks:
            # lbfull: use only E-mode for LB-entire region
            # lbonly: use LB E-mode for LB-only region
            # lbs4:   use LB+S4 E-mode for LB-S4 overlap region
            # cmball: try to combine LB and S4 E-mode for LB region
            
            if misctools.check_path(cobj.fwalm[m][rlz],**kwargs_ov): continue

            if omap is None: # only one time calculation
            
                omap = np.zeros((2,2,npix))
        
                # LiteBIRD HILC map
                hElm, hBlm = hp.read_alm(pobj.fhilc[rlz],(2,3))
                bElm = hp.almxfl(hElm,bl[0],inplace=True)/cmb.Tcmb
                bBlm = hp.almxfl(hBlm,bl[0],inplace=True)/cmb.Tcmb
                __, omap[0,0,:], omap[1,0,:] = hp.alm2map([bElm*0.,bElm,bBlm],nside=nside)
                
                # S4
                Q, U = hp.read_map(pobj.ficmb[rlz],field=(1,2))/c.Tcmb
                Elm, Blm = cs.utils.hp_map2alm_spin(hp.get_nside(Q),lmax,lmax,2,Q,U)
                sQs4, sUs4 = cs.utils.hp_alm2map_spin(nside,lmax,lmax,2,Elm*bl[1],Blm*bl[1])
                omap[0,1,:] = sQs4 + nQs4
                omap[1,1,:] = sUs4 + nUs4

            
            data = omap * Mask[m]
            invN = np.zeros((2,2,npix))
            invN[:,0,:] = Mask['lbfull'] # LB mask
            invN[:,1,:] = Mask['lbs4'] # S4 mask
            
            if m == 'lbfull':  # observed cmb maps with only LB
                wElm, wBlm = cs.cninv.cnfilter_freq(2,1,nside,lmax,pobj.lcl[1:3,:lmax+1],bl[:1,:],invN[:,:1,:],data[:,:1,:],inl=iNls[:,:1,:],**kwargs_cinv)

            else:  # observed cmb maps by combining LB and S4
                wElm, wBlm = cs.cninv.cnfilter_freq(2,2,nside,lmax,pobj.lcl[1:3,:lmax+1],bl,invN,data,inl=iNls,**kwargs_cinv)

            pickle.dump( (wElm,wBlm), open(cobj.fwalm[m][rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )
    

def compute_wiener_lowl(pobj,cobj,snmax,nside=128,lmax=190,**kwargs_ov):
    '''
    Compute wiener-filtered B-mode at large scale using LiteBIRD B-modes
    '''

    # get mask
    Mask = prepare_masks(nside)

    # get beam and pixel window function for large scale B-modes
    bl = cmb.beam(80.,lmax,inv=False) # 80 arcmin beam to match the PTEP FG sims
    wl = hp.sphtfunc.pixwin(nside,lmax=lmax)
    
    # loop over realizations
    for rlz in tqdm.tqdm(local.rlz(1,snmax),ncols=100,desc='rlz (cmb wiener low-l)'):

        Qs = None

        for m in masks: # lbfull and cmball look the same

            if misctools.check_path(cobj.foblm[m][rlz],**kwargs_ov): continue

            if Qs is None: # only one time calculation
            
                # lensed Q/U map
                Q, U = hp.read_map(pobj.ficmb[rlz],field=(1,2))/c.Tcmb
                Qs, Us = qumap_smoothing(Q,U,lmax,nside,bl)
            
                # tensor Q/U map
                rlm = pickle.load(open(cobj.fralm[rlz],"rb"))[:lmax+1,:lmax+1]
                Qr, Ur = cs.utils.hp_alm2map_spin(nside,lmax,lmax,2,0*rlm,rlm)
                Qr, Ur = qumap_smoothing(Qr,Ur,lmax,nside,bl)

                # FG noise map
                Qn = hp.ud_grade(hp.read_map(pobj.ffgs[rlz],field=1)/c.Tcmb,nside)
                Un = hp.ud_grade(hp.read_map(pobj.ffgs[rlz],field=2)/c.Tcmb,nside)
                Qn[np.isnan(Qn)] = 0
                Un[np.isnan(Un)] = 0

                inls = np.array((1./pobj.clfg[:lmax+1],1./pobj.clfg[:lmax+1])).reshape(2,1,lmax+1)

            wBlm = qumap_filter(Qs+Qn,Us+Un,Mask[m],pobj.lcl,inls,bl*wl)[1]
            rBlm = qumap_filter(Qr,Ur,Mask[m],pobj.lcl,inls,bl*wl)[1]
            pickle.dump( (wBlm,rBlm), open(cobj.foblm[m][rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )

        
def qumap_filter(Q,U,M,lcl,inls,beam): 
    # Wiener filter for large-scale B-modes
    
    NSIDE = hp.get_nside(Q)
    NPIX  = hp.nside2npix(NSIDE)

    data = np.array((Q*M,U*M)).reshape((2,1,NPIX))
    invN = np.array((M,M)).reshape((2,1,NPIX))
    
    lmax = len(beam) - 1
    bls = np.array((beam)).reshape((1,lmax+1))

    kwargs_cinv = {
        'chn':  1, \
        'eps':  [1e-4], \
        'itns': [1000], \
        'ro':   10, \
        'inl':  inls, \
        'stat': 'status_wiener_lowl.txt' \
    }
    
    return cs.cninv.cnfilter_freq(2,1,NSIDE,lmax,lcl[1:3,:lmax+1],bls,invN,data,**kwargs_cinv)



