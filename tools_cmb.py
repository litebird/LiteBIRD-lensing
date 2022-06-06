# Module for Multitracers
import numpy as np
import healpy as hp
import pickle
import tqdm
import warnings
warnings.filterwarnings("ignore")

# from cmblensplus/wrap
import curvedsky as cs

# from cmblensplus/utils
import constant as c
import misctools
import cmb

# from local module
import local

# //// Fixed values //// #

masks = {'lbs4','lbonly','lbfull'}

theta = {'lb':30.,'s4':2.}
sigma = {'lb':2.,'s4':1.} # uK-arcmin in polarization


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

def prepare_cmb_Ncov(lmax):
    Ncov = np.zeros((4,4,lmax+1))
    Ncov[0,0,:] = Ncov[1,1,:] = (sigma['lb']*c.ac2rad/c.Tcmb)**2
    Ncov[2,2,:] = Ncov[3,3,:] = (sigma['s4']*c.ac2rad/c.Tcmb)**2
    return Ncov

def prepare_beam(lmax):
    bl = np.zeros((2,lmax+1))
    bl[0] = cmb.beam(theta['lb'],lmax,inv=False)
    bl[1] = cmb.beam(theta['s4'],lmax,inv=False)
    return bl


def prepare_masks(nside=None):
    
    params = local.analysis()

    W_LB = hp.read_map(params.wind['litebird'])
    W_S4 = W_LB * hp.read_map(params.wind['cmbs4'])

    mask = {}
    mask['lbs4']   = W_S4
    mask['lbonly'] = W_LB*(1.-W_S4)
    mask['lbfull'] = W_LB
    
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
    #mask = hp.ud_grade(hp.read_map('../data/lensing/FG_mask.fits'),nside)
    #W = cs.utils.apodize(nside, mask, 1.)

    # residual FG noise
    Qn, Un = hp.read_map(pobj.ffgs[rlz],field=(1,2))/c.Tcmb
    # nan to zero
    Qn[np.isnan(Qn)] = 0
    Un[np.isnan(Un)] = 0
    Qn = hp.ud_grade( Qn, nside )
    Un = hp.ud_grade( Un, nside )
    nBlm = cs.utils.hp_map2alm_spin(nside,lmax,lmax,2,mask*Qn,mask*Un)[1]/(bl[:,None]*Wl[:,None])

    # lensing 
    Qs, Us = hp.read_map(pobj.ficmb[rlz],field=(1,2))/c.Tcmb
    nsides = hp.get_nside(Qs)
    if method == 'bonly': # ignore E-to-B leakage of lensing
        sBlm = cs.utils.hp_map2alm_spin(nsides,lmax,lmax,2,Qs,Us)[1]
        Qs, Us = cs.utils.hp_alm2map_spin(nsides,lmax,lmax,2,0*sBlm,sBlm)
    Qs, Us = qumap_smoothing(Qs,Us,lmax,nside,bl)
    sBlm = cs.utils.hp_map2alm_spin(nside,lmax,lmax,2,mask*Qs,mask*Us)[1]/(bl[:,None]*wl[:,None])

    # tensor
    rBlm = pickle.load(open(cobj.fralm[rlz],"rb"))
    lrmax = len(rBlm[:,0]) - 1
    Qr, Ur = cs.utils.hp_alm2map_spin(nside,lrmax,lrmax,2,0*rBlm,rBlm)
    Qr, Ur = qumap_smoothing(Qr,Ur,lmax,nside,bl)
    rElm, rBlm = cs.utils.hp_map2alm_spin(nside,lmax,lmax,2,mask*Qr,mask*Ur)/(bl[:,None]*wl[:,None])

    return sBlm, rBlm, nBlm
    

def compute_cmb_noise(cobj,snmax,lmax=1024,**kwargs_ov):
    '''
    Generate noise alms
    '''
    
    # noise covariance
    Ncov = prepare_cmb_Ncov(lmax)

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
        

    
def compute_wiener_highl(pobj,cobj,snmax,nside=512,lmax=1024,**kwargs_ov):

    # set parameters
    npix = hp.nside2npix(nside)

    # get mask
    Mask = prepare_masks(nside)
    
    # noise covariance
    Ncov = prepare_cmb_Ncov(lmax)
    
    # beam
    bl = prepare_beam(lmax)

    # kwargs for cinv
    kwargs_cinv = {'chn':1,'itns':[1000],'eps':[1e-4],'ro':10,'stat':'status_wiener_highl.txt'}
    
    # loop over realizations
    for rlz in tqdm.tqdm(local.rlz(1,snmax),ncols=100,desc='rlz (cmb wiener high-l)'):
        
        nlm = pickle.load(open(cobj.fnalm[rlz],"rb"))

        smap = None

        for m in masks:

            if misctools.check_path(cobj.fwalm[m][rlz],**kwargs_ov): continue

            if smap is None: # only one time calculation
            
                nmap = np.zeros((2,2,npix))
                nmap[0,0,:], nmap[1,0,:] = cs.utils.hp_alm2map_spin(nside,lmax,lmax,2,nlm[0],nlm[1])
                nmap[0,1,:], nmap[1,1,:] = cs.utils.hp_alm2map_spin(nside,lmax,lmax,2,nlm[2],nlm[3])

                Q, U = hp.read_map(pobj.ficmb[rlz],field=(1,2))/c.Tcmb
                Elm, Blm = cs.utils.hp_map2alm_spin(hp.get_nside(Q),lmax,lmax,2,Q,U)

                smap = np.zeros((2,2,npix))
                smap[0,0,:], smap[1,0,:] = cs.utils.hp_alm2map_spin(nside,lmax,lmax,2,Elm*bl[0],Blm*bl[0])
                smap[0,1,:], smap[1,1,:] = cs.utils.hp_alm2map_spin(nside,lmax,lmax,2,Elm*bl[1],Blm*bl[1])

            
            data = (smap+nmap) * Mask[m]
            invN = np.zeros((2,2,npix))
            invN[:,0,:] = Mask[m]/Ncov[0,0,0]
            invN[:,1,:] = Mask[m]/Ncov[2,2,0]
            
            if m == 'lbfull':  # observed cmb maps with only LB
                wElm, wBlm = cs.cninv.cnfilter_freq(2,1,nside,lmax,pobj.lcl[1:3,:lmax+1],bl[:1,:],invN[:,:1,:],data[:,:1,:],**kwargs_cinv)

            else:  # observed cmb maps by combining LB and S4
                wElm, wBlm = cs.cninv.cnfilter_freq(2,2,nside,lmax,pobj.lcl[1:3,:lmax+1],bl,invN,data,**kwargs_cinv)

            pickle.dump( (wElm,wBlm), open(cobj.fwalm[m][rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )
    

def compute_wiener_lowl(pobj,cobj,snmax,nside=128,lmax=190,**kwargs_ov):

    # get mask
    Mask = prepare_masks(nside)

    # get beam and pixel window function for large scale B-modes
    bl = cmb.beam(80.,lmax,inv=False) # 80 arcmin beam to match the PTEP FG sims
    wl = hp.sphtfunc.pixwin(nside,lmax=lmax)
    
    # loop over realizations
    for rlz in tqdm.tqdm(local.rlz(1,snmax),ncols=100,desc='rlz (cmb wiener low-l)'):

        Qs = None

        for m in masks:

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



