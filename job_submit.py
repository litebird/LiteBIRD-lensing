
import numpy as np, os
from quicksub import *

def jobfile(f_run):

    # set job file
    f_sub = 'tmp_job_'+f_run.replace('.py','')+'.sh'
    set_sbatch_params(f_sub,'',mem='64G',t='0-12:00',email=True)
    add('source ~/.bashrc.ext',f_sub)
    add('py4so',f_sub)
    add('python '+f_run,f_sub)
    
    # submit    
    os.system('sbatch '+f_sub)
    #os.system('sh '+f_sub)
    #os.system('rm -rf '+f_run+' '+f_sub)


f_run = 'run_mass_cinv.py'
#f_run = 'run_cmb.py'
jobfile(f_run)

