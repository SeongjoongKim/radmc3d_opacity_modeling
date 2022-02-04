
radmc3d_opacity_modeling

For making DSHARP composition, dsharp_opac package is prerequisit. You can install this package with:

pip install dsharp_opac

After installing the package, please modify the model parameters in create_radmc3d_inputfile.py

parameter setup

logawidth    = None          # Smear out the grain size by 5% in both directions

wfact        = 3.0           # Grid width of na sampling points in units of logawidth

chopforward  = 1.          # Remove forward scattering within an angle of 5 degrees

optconst     = "dsharp"      # The optical constants name

descr        = "DSHARP mixture model"

extrapolate  = True          # Extrapolate optical constants beyond its wavelength grid, if necessary

verbose      = False         # If True, then write out status information

ntheta       = 181           # Number of scattering angle sampling points

amin = 1e-5; amax = 0.15e-1; na = 200  # Set a_min, a_max in cm unit and number of a grids

wave_min = 1.0e-4; wave_max = 1e0; nwave = 200   # Set lam_min, lam_max in cm unit and number of lam grids

errtol       = 0.1           # Tolerance of the relative difference between kscat and the integral over the zscat Z11 element over angle. If this tolerance is exceeded, a warning is given.

name         = 'DSHARP_amax{:5.1f}um'.format(amax*1e4)  # Output file name

