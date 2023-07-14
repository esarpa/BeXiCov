#==== BeXiCov: Best-fit Xi covariance matrix
python package to construct the semi-analytical Gaussian covariance matrix of the measured 2PCF multipoles (pre and post-reconstruction)

code used for the DR1-Euclid forecast on BAO measurements (Sarpa et al in prep)


#---- Modules
- Models.py:      
   containes the routines used to compute the power-specturm and correlation fucntion models
- Covariance.py
    module generate the theoretical and best fit covariance matrix reproducinf the data

#---- Inputs: 
- measured two-point correlation fucntion multipoles
- fiducial cosmology
- survey volume
- survey mean number density
- rectype: 
    '': for pre-reconstruction
    'rec-iso': for Zel'dovich post-reconstruction, with RSD removal
    'rec-sym': for Zel'dovich post-reconstruction, without RSD removal
-space: 'RealSpace' or 'RedshiftSpace

#---- Outputs: 
- covariance matrix of the two-point correlation function multipoles evaluated at the data separation vector


#---- installation
gitclone ...

pip install .

To build theoretical covariances:
- GaussianCovariance
    git clone https://gitlab.com/veropalumbo.alfonso/gaussiancovariance/
    cd gaussiancovariance
    pip insall.
  

The routine to generate the template for the 2PCF multipoles requires
- camb
    conda install -c conda-forge camb
- hankl
    pip install hankl

#---- Examples:

#---- Code description:
- generates the theoretical covariance matrix based on 1st order Lagrangian perturbation theory (padmanabhan et al 2009, Sarpa et al 2023.)
- models the 2PCF multipoles using the damped correlation function modelm (Scocciamarro ...) + polynomial broad-band
- computes the hankle tranform of the best fit 2PCF to obtsined the best-fit anisotropic power-spectrum
- generate the gaussian covariance matrix from the best-fit anisotropic power-spectrum