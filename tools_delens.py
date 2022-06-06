# Module for Multitracers
import numpy as np
import pickle
import tqdm
#import warnings
#warnings.filterwarnings("ignore")

# from cmblensplus/wrap
import curvedsky as cs

# from cmblensplus/utils
import analysis as ana
import misctools

# from local module
import local
import tools_cmb

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
 
        # Lensing B-mode template
        self.fLTlm = {m: [ d['del'] + 'alm/LT_' + m + '_' + method + '_' + str(rlz) + '.pkl' for rlz in local.ids ] for m in masks }

        # BB spectra
        self.cl = {m: [ [ d['del'] + 'aps/rlz/LT_' + m + '_' + method + '_' + str(mi) + '_' + str(rlz) + '.pkl' for rlz in local.ids ] for mi in range(5) ] for m in masks }


def compute_HL_r(oBB,BB,rBB,rs):

    mBB  = np.mean(BB,axis=0)
    icov = np.linalg.inv(np.cov(BB,rowvar=0))
    return np.array( [ ana.lnLHL(oBB/(mBB+r*rBB),mBB,icov) for r in rs ] )
        

def simple_r(oBB,BB,rBB):

    mBB  = np.mean(BB,axis=0)
    icov = np.linalg.inv(np.cov(BB-mBB,rowvar=0))
    
    wb = np.dot(icov,rBB)
    
    return np.sum(wb*(oBB-mBB))/np.sum(rBB*wb)
    

def compute_clbb(cobj,mobj,dobj,snmax,snmin=1,elmin=150,klmin=2,lmax=2*512,lbmax=190,nbside=128,**kwargs_ov):
    
    pobj = local.analysis()
    
    for rlz in tqdm.tqdm(local.rlz(snmin,snmax),ncols=100,desc='each rlz'):

        # read coadd kappa map
        klm = {}
        klm[0], klm[1], klm[2], klm[3], klm[4] = pickle.load(open(mobj.fwklm[rlz],"rb"))


        for m in tools_cmb.masks:
        
            if misctools.check_path(dobj.fLTlm[m][rlz],**kwargs_ov):
            
                blm = pickle.load(open(dobj.fLTlm[m][rlz],"rb"))
    
            else:

                # read Wiener-filtered polarization
                wElm = pickle.load(open(cobj.fwalm[m][rlz],"rb"))[0]

                # compute lensing B-mode template
                blm = {}
                for i in range(5):
                    blm[i] = cs.delens.lensingb( lmax, elmin, lmax, klmin, lmax, wElm, klm[i], gtype='k')

                pickle.dump( (blm), open(dobj.fLTlm[m][rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )

            if misctools.check_path(dobj.cl[m][0][rlz],**kwargs_ov): continue
        
            if dobj.method == 'cinv':
                # wiener-filtered observed B-mode map
                wBlm, rBlm = pickle.load(open(cobj.foblm[m][rlz],"rb"))
            if dobj.method == 'bonly':
                # sBlm contains only lensing-B-mode
                sBlm, rBlm, nBlm = tools_cmb.prepare_obs_Bmap(pobj,cobj,rlz,m,nside=nbside,lmax=lbmax,method='bonly')
                wBlm = sBlm + nBlm

            # aps
            obb = cs.utils.alm2cl(lbmax,wBlm)
            rbb = cs.utils.alm2cl(lbmax,rBlm)
        
            for i in range(5):

                lbb = cs.utils.alm2cl(lbmax,blm[i][:lbmax+1,:lbmax+1])
                xbb = cs.utils.alm2cl(lbmax,blm[i][:lbmax+1,:lbmax+1],wBlm)

                np.savetxt(dobj.cl[m][i][rlz],np.array((obb,rbb,lbb,xbb)).T)
                
                
