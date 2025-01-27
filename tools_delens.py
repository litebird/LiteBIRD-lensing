# Module for Multitracers
import numpy as np
import pickle
import tqdm
#import warnings
#warnings.filterwarnings("ignore")

import cmblensplus.curvedsky as cs
import analysis as ana
import misctools
import binning as bn

# from local module
import local
import tools_cmb
import tools_multitracer

# //// Fixed values //// #
masks = tools_cmb.masks

class filename():
    # define object which has parameters and filenames for multitracer analysis
    
    def __init__( self, method='cinv'):

        #set directory
        d = local.data_directory()
        
        # large scale B-mode calculation method
        if method not in ['cinv','bonly','apod']:
            sys.exit('method is not specified')

        self.method = method

        # type of kappa map
        self.ctypes = ['klb','cib','gal','ext','all']
 
        # Lensing B-mode template
        self.fLTlm = {m: [ d['del'] + 'alm/LT_' + m + '_' + method + '_' + str(rlz) + '.pkl' for rlz in local.ids ] for m in masks }

        # BB spectra
        self.cl = {m: {ctype: [ d['del'] + 'aps/rlz/LT_' + m + '_' + method + '_' + ctype + '_' + str(rlz) + '.pkl' for rlz in local.ids ] for ctype in self.ctypes } for m in masks }


def compute_mean(cl):
    simn = len(cl[:,0])
    return np.array( [ np.mean(np.delete(cl,i,0),axis=0) for i in range(simn) ] )


def compute_HL_r(oBB,BB,rBB,rs):

    mBB  = np.mean(BB,axis=0)
    icov = np.linalg.inv(np.cov(BB,rowvar=0))
    return np.array( [ ana.lnLHL(oBB/(mBB+r*rBB),mBB,icov) for r in rs ] )


def simple_r(BB,rBB,r,cross=0.):
    '''
    Computing hat r
    BB: obs BB with r=0
    rBB: tensor BB with r=1
    cross: 2 B^obs x B^tens ( - 2 B^temp x B^tens )
    r: desired r
    '''

    # obtain averaged cls over rlz
    mrBB = np.mean(rBB,axis=0)
    
    simn = len(BB[:,0])
    mBB  = compute_mean(BB) # r=0 BB
    oBB  = BB + cross + r*rBB # observed BB with nonzero r
    
    # hat r at each bin
    amp  = (oBB-mBB)/mrBB

    # combine har r
    wbi = np.array( [ np.sum( np.linalg.inv( np.cov(np.delete(amp,i,0),rowvar=0) ),axis=0 ) for i in range(simn) ] )
    wti = np.array( [ np.sum(wbi[i,:]) for i in range(simn)] )
    return np.array( [np.sum(wbi[i,:]*amp[i,:])/wti[i] for i in range(simn)] )


def simple_r_zero(BB,rBB):
    # assuming r=0 and compute hat r
    
    # obtain averaged cls over rlz
    mrBB = np.mean(rBB,axis=0)
    
    simn = len(BB[:,0])
    mBB  = compute_mean(BB)
    amp  = (BB-mBB)/mrBB
    
    wbi = np.array( [ np.sum( np.linalg.inv( np.cov(np.delete(amp,i,0),rowvar=0) ),axis=0 ) for i in range(simn) ] )
    wti = np.array( [ np.sum(wbi[i,:]) for i in range(simn)] )
    return np.array( [np.sum(wbi[i,:]*amp[i,:])/wti[i] for i in range(simn)] )


def compute_clbb(cobj,dobj,snmax,snmin=1,elmin=150,klmin=2,lmax=2*512,lbmax=190,nbside=128,
                 kwargs_ov_blm={'overwrite':False,'verbose':True},kwargs_ov_aps={'overwrite':False,'verbose':True}
                ):
    
    pobj = local.analysis()
    
    for rlz in tqdm.tqdm(local.rlz(snmin,snmax),ncols=100,desc='each rlz'):

        # read coadd kappa map
        klm = {}
        for ctype in dobj.ctypes:
            mobj = tools_multitracer.comb_mobj(ctype,lmax=lmax)
            klm[ctype] = pickle.load(open(mobj.fwklm[rlz],"rb"))
            #klm[0], klm[1], klm[2], klm[3], klm[4] = pickle.load(open(mobj.fwklm[rlz],"rb"))


        for m in tools_cmb.masks:
            
            if m != 'cmball': continue # only compute for LB+S4 combined case
        
            if misctools.check_path(dobj.fLTlm[m][rlz],**kwargs_ov_blm):
            
                blm = pickle.load(open(dobj.fLTlm[m][rlz],"rb"))
    
            else:

                # read Wiener-filtered polarization
                wElm = pickle.load(open(cobj.fwalm[m][rlz],"rb"))[0]

                # compute lensing B-mode template
                blm = {}
                for ctype in dobj.ctypes:
                    blm[ctype] = cs.delens.lensingb( lmax, elmin, lmax, klmin, lmax, wElm, klm[ctype], gtype='k')

                pickle.dump( (blm), open(dobj.fLTlm[m][rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )

            if misctools.check_path(dobj.cl[m][dobj.ctypes[0]][rlz],**kwargs_ov_aps): continue
        
            if dobj.method == 'cinv':
                # wiener-filtered observed B-mode map
                if m != 'cmball':
                    wBlm, rBlm = pickle.load(open(cobj.foblm[m][rlz],"rb"))
                else:
                    wBlm, rBlm = pickle.load(open(cobj.foblm['lbfull'][rlz],"rb")) # LB-only CMB data
                    
                sBlm = wBlm.copy() # dummy
                
            if dobj.method == 'bonly': # not used for paper
                # sBlm contains only lensing-B-mode
                sBlm, rBlm, nBlm = tools_cmb.prepare_obs_Bmap(pobj,cobj,rlz,m,nside=nbside,lmax=lbmax,method='bonly')
                wBlm = sBlm + nBlm

            # aps
            oBB  = cs.utils.alm2cl(lbmax,wBlm)
            sBB  = cs.utils.alm2cl(lbmax,sBlm)
            rBB  = cs.utils.alm2cl(lbmax,rBlm)
            orBB = cs.utils.alm2cl(lbmax,wBlm,rBlm) # this is need to compute non-zero r case
        
            for ctype in dobj.ctypes:

                lBB  = cs.utils.alm2cl(lbmax,blm[ctype][:lbmax+1,:lbmax+1])
                xBB  = cs.utils.alm2cl(lbmax,blm[ctype][:lbmax+1,:lbmax+1],wBlm)
                xrBB = cs.utils.alm2cl(lbmax,blm[ctype][:lbmax+1,:lbmax+1],rBlm) # this is need to compute non-zero r case

                print(dobj.cl[m][ctype][rlz])
                np.savetxt(dobj.cl[m][ctype][rlz],np.array((oBB,rBB,lBB,xBB,sBB,orBB,xrBB)).T)
                

def load_clbbs(fcl,mb,totN,fsky):
    bN = mb.n
    lbmin = mb.lmin
    lbmax = mb.lmax
    Ob = np.zeros((totN,bN))
    Rb = np.zeros((totN,bN))
    Lb = np.zeros((totN,bN))
    Xb = np.zeros((totN,bN))
    orb = np.zeros((totN,bN))
    xrb = np.zeros((totN,bN))
    for rlz in range(1,totN+1):
        #obb, rbb, lbb, xbb, orbb, xrbb = np.loadtxt(fcl[rlz],usecols=(0,1,2,3,5,6),unpack=True)[:,lbmin:lbmax+1]
        obb, rbb, lbb, xbb = np.loadtxt(fcl[rlz],usecols=(0,1,2,3),unpack=True)[:,lbmin:lbmax+1]
        Ob[rlz-1,:] = bn.binning(obb,mb)/fsky
        Rb[rlz-1,:] = bn.binning(rbb,mb)/fsky
        Lb[rlz-1,:] = bn.binning(lbb,mb)/fsky
        Xb[rlz-1,:] = bn.binning(xbb,mb)/fsky
        #orb[rlz-1,:] = bn.binning(orbb,mb)/fsky
        #xrb[rlz-1,:] = bn.binning(xrbb,mb)/fsky
    # compute delensed cls
    mAb = compute_mean(Xb)/compute_mean(Lb)
    Db = Ob - 2*mAb*Xb + mAb**2*Lb
    return Ob, Db, Rb#, 2*orb, -2*mAb*xrb

