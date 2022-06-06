# LiteBIRD-lensing
This repository contains tools for litebird lensing analysis aiming at analyzing real LiteBIRD data for measuring gravitational lenisng. 

### Main codes
#### run_cmb.py --- reading or generating CMB maps and computing C-inverse filtered harmonic coefficients.
#### run_lensingb.py --- reading Wiener-filtered E-modes and Wiener-filtered kappa alms and constructing lensing B-mode template.
#### run_mass_aps.py --- computing theoretical Cls for mass tracers (CIB, LSST, kappa). 
#### run_mass_cinv.py --- generating external mass-tracer maps, and combining then with the Wiener-filtering. 
#### run_recons.py (not yet implemented) --- performing lensing reconstruction from the C-inverse filtered harmonic coefficients. 

### Sub codes
#### tools_cmb.py --- tools for run_cmb.py
#### tools_delens.py --- tools for run_lensing.py
#### tools_multitracer.py --- tools for run_mass_aps.py and run_mass_cinv.py
#### tools_recons.py --- tools for run_recons.py


### Dependencies
This code utilizes cmblensplus (https://toshiyan.github.io/clpdoc/html/)

### Contributors
Anto I Lonappan, 
Toshiya Namikawa, 
...

