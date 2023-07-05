# BeXiCov: Best-fit Xi Covariance Matrix

**BeXiCov** is a Python package that constructs the semi-analytical Gaussian covariance matrix for the measured multipoles of the two-point correlation function (2PCF) pre and post-reconstruction. It was developed for the DR1-Euclid forecast on BAO measurements (Sarpa et al., in prep).

## Modules

- **Models.py**: Contains the routines used to compute the power spectrum and correlation function models.
- **Covariance.py**: Generates the theoretical and best-fit covariance matrix reproducing the data.

## Inputs

- Measured two-point correlation function multipoles.
- Fiducial cosmology.
- Survey volume.
- Survey mean number density.
- `rectype`:
  - `''`: Pre-reconstruction.
  - `'rec-iso'`: Zel'dovich post-reconstruction with RSD removal.
  - `'rec-sym'`: Zel'dovich post-reconstruction without RSD removal.
- `space`: `'RealSpace'` or `'RedshiftSpace'`.

## Outputs

- Covariance matrix of the two-point correlation function multipoles evaluated at the data separation vector.

## Installation

1. Clone the BeXiCov repository
2. Clone the GaussianCovariance repository:
   - git clone https://gitlab.com/veropalumbo.alfonso/gaussiancovariance/
   - cd gaussiancovariance
   - pip insall.
3. Install Camb via: conda install -c conda-forge camb
4. Install Hankl via: pip install hankl


## Examples

Provide examples of how to use the BeXiCov package with code snippets and explanations.

## Code Description

- Generates the theoretical covariance matrix based on 1st order Lagrangian perturbation theory (Padmanabhan et al. 2009, Sarpa et al. 2023).
- Models the two-point correlation function multipoles using the damped correlation function model (Scoccimarro...) and polynomial broad-band.
- Computes the Hankel transform of the best-fit 2PCF to obtain the best-fit anisotropic power spectrum.
- Generates the Gaussian covariance matrix from the best-fit anisotropic power spectrum.

