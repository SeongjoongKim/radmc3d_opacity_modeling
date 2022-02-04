import numpy as np
import matplotlib.pyplot as plt
import dsharp_opac as opacity
from scipy.interpolate import interp1d
import math
#from makedustopac import *
from bhmie import *

# ====================================================================================================
#   parameter setup
# ====================================================================================================
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

# ====================================================================================================
#   Calculation start
# ====================================================================================================
# Grid setup for a and lam
a = np.logspace(np.log10(amin),np.log10(amax),na) # grain sizes in cm
lam = np.logspace(np.log10(wave_min), np.log10(wave_max),nwave)  # wavelength grid in cm
theta     = np.linspace(0.,180.,ntheta)  # Set scattering angle grids in degree

assert lam.size > 1, "Error: Optical constants file must have at least two rows with two different wavelengths."
assert lam[1]!=lam[0], "Error: Optical constants file must have at least two rows with two different wavelengths."

# DSHARP composition and optical constants calculation
references = "Birnstiel et al. 2018 ApJL 869 45"
extrapolate_large_grains = False
fm_ice = 0.2
dc,rhomat = opacity.get_dsharp_mix(fm_ice=fm_ice, porosity=0.0, rule='Bruggeman')
matdens = rhomat
#
# Make the complex index of refraction
#
n = np.zeros(len(lam)); k = np.zeros(len(lam))
for i in range(len(n)):
    n[i], k[i] = dc.nk(lam[i])
lammic = lam*1e4
f = interp1d(np.log(lammic),np.log(n))
ncoefi = np.exp(f(np.log(lammic)))
f = interp1d(np.log(lammic),np.log(k))
kcoefi = np.exp(f(np.log(lammic)))
refidx = ncoefi + kcoefi*1j

# Set weightings following size distribution of N(a) ~ a^pla
pla = -3.5
N_agr = a**pla
mgrain = 4.*np.pi/3.*rhomat*a**3
wgt = N_agr * mgrain * a
wgt /= wgt.sum()
#
# Make a grid of angles, if not already given by theta
#
angles = theta
nang = angles.size
#
# Check that the theta array goes from 0 to 180 or
# 180 to 0, and store which is 0 and which is 180
#
if angles[0]==0.0:
    assert angles[nang-1]==180, "Error: Angle grid must extend from 0 to 180 degrees."
else:
    assert angles[0]==180, "Error: Angle grid must extend from 0 to 180 degrees."
    assert angles[nang-1]==0, "Error: Angle grid must extend from 0 to 180 degrees."
#
# Make a size distribution for the grains
# If a single size set and width is given then compute gaussian distribution around agraincm
# If width is not set, then take just one size
#
if (type(a) in [list, np.ndarray]) and (logawidth is None):
    agr = np.array(a)
    assert wgt.all() != None, "Error: Range of grain sizes is given, but the weighting (wgt) is not set."
    wgt = np.array(wgt)/np.array(wgt).sum() # make sure that the distribution is normalised
elif logawidth:
    agr   = np.exp(np.linspace(math.log(a)-wfact*logawidth,
                               math.log(a)+wfact*logawidth,na))
    wgt   = np.exp(-0.5*((np.log(agr/a))/logawidth)**2)
    wgt   = wgt/wgt.sum()
else:
    agr   = np.array([a])
    wgt   = np.array([1.0])
#
# Get the true number of grain sizes
#
nagr = agr.size
#
# Compute the geometric cross sections
#
siggeom = math.pi*agr*agr
#
# Compute the mass of the grain
#
mgrain  = (4*math.pi/3.0)*matdens*agr*agr*agr
#
# Now prepare arrays
#
nlam  = lam.size
kabs  = np.zeros(nlam)
kscat = np.zeros(nlam)
gscat = np.zeros(nlam)
if theta is not None:
    zscat = np.zeros((nlam,nang,6))
    S11   = np.zeros(nang)
    S12   = np.zeros(nang)
    S33   = np.zeros(nang)
    S34   = np.zeros(nang)
    if chopforward>0:
        zscat_nochop = np.zeros((nlam,nang,6))
        kscat_nochop = np.zeros(nlam)
#
# Set error flag to False
#
error  = False
errmax = 0.0
kscat_from_z11 = np.zeros(nlam)
#
# Loop over wavelengths
#
for i in range(nlam):
    #
    # Message
    #
    if verbose:
        print("Doing wavelength %13.6e cm"%lam[i])
    #
    # Now loop over the grain sizes
    #
    for l in range(nagr):
        #
        # Message
        #
        if verbose and nagr>1:
            print("...Doing grain size %13.6e cm"%agr[l])
        #
        # Compute x
        #
        x = 2*math.pi*agr[l]/lam[i]
        #
        # Call the bhmie code
        #
        S1, S2, Qext, Qabs, Qsca, Qback, gsca = bhmie(x,refidx[i],angles)
        #
        # Add results to the averaging over the size distribution
        #
        kabs[i]   += wgt[l] * Qabs*siggeom[l]/mgrain[l]
        kscat[i]  += wgt[l] * Qsca*siggeom[l]/mgrain[l]
        gscat[i]  += wgt[l] * gsca
        #
        # If angles were set, then also compute the Z matrix elements
        #
        if theta is not None:
            #
            # Compute conversion factor from the Sxx matrix elements
            # from the Bohren & Huffman code to the Zxx matrix elements we
            # use (such that 2*pi*int_{-1}^{+1}Z11(mu)dmu=kappa_scat).
            # This includes the factor k^2 (wavenumber squared) to get
            # the actual cross section in units of cm^2 / ster, and there
            # is the mass of the grain to get the cross section per gram.
            #
            factor = (lam[i]/(2*math.pi))**2/mgrain[l]
            #
            # Compute the scattering Mueller matrix elements at each angle
            #
            S11[:]        = 0.5 * ( np.abs(S2[:])**2 + np.abs(S1[:])**2 )
            S12[:]        = 0.5 * ( np.abs(S2[:])**2 - np.abs(S1[:])**2 )
            S33[:]        = np.real(S2[:]*np.conj(S1[:]))
            S34[:]        = np.imag(S2[:]*np.conj(S1[:]))
            zscat[i,:,0] += wgt[l] * S11[:] * factor
            zscat[i,:,1] += wgt[l] * S12[:] * factor
            zscat[i,:,2] += wgt[l] * S11[:] * factor
            zscat[i,:,3] += wgt[l] * S33[:] * factor
            zscat[i,:,4] += wgt[l] * S34[:] * factor
            zscat[i,:,5] += wgt[l] * S33[:] * factor
    #
    # If possible, do a check if the integral over zscat is consistent
    # with kscat
    #
    if theta is not None:
        mu  = np.cos(angles*math.pi/180.)
        dmu = np.abs(mu[1:nang]-mu[0:nang-1])
        zav = 0.5 * ( zscat[i,1:nang,0] + zscat[i,0:nang-1,0] )
        dum = 0.5 * zav*dmu
        sum = dum.sum() * 4 * math.pi
        kscat_from_z11[i] = sum
        err = abs(sum/kscat[i]-1.0)
        if err>errtol:
            error = True
            errmax = max(err,errmax)
    #
    # If the chopforward angle is set >0, then we will remove
    # excessive forward scattering from the opacity. The reasoning
    # is that extreme forward scattering is, in most cases, equivalent
    # to no scattering at all.
    #
    if chopforward>0:
        iang  = np.where(angles<chopforward)
        if angles[0]==0.0:
            iiang = np.max(iang)+1
        else:
            iiang = np.min(iang)-1
        zscat_nochop[i,:,:] = zscat[i,:,:]  # Backup
        kscat_nochop[i]     = kscat[i]      # Backup
        zscat[i,iang,0]     = zscat[i,iiang,0]
        zscat[i,iang,1]     = zscat[i,iiang,1]
        zscat[i,iang,2]     = zscat[i,iiang,2]
        zscat[i,iang,3]     = zscat[i,iiang,3]
        zscat[i,iang,4]     = zscat[i,iiang,4]
        zscat[i,iang,5]     = zscat[i,iiang,5]
        mu  = np.cos(angles*math.pi/180.)
        dmu = np.abs(mu[1:nang]-mu[0:nang-1])
        zav = 0.5 * ( zscat[i,1:nang,0] + zscat[i,0:nang-1,0] )
        dum = 0.5 * zav*dmu
        sum = dum.sum() * 4 * math.pi
        kscat[i] = sum
        muav = 0.5 * (mu[1:]+mu[:-1])
        dumg = 0.5 * zav*muav*dmu
        sumg = dumg.sum() * 4 * math.pi
        gscat[i] = sumg/sum
#
# If error found, then warn
#
if error:
    print("Warning: Angular integral of Z11 is not equal to kscat at all wavelength. ")
    print("Maximum error = %13.6e"%errmax)
    if chopforward>0:
        print("But I am using chopforward to remove strong forward scattering, and then renormalized kapscat.")
#
# Now return what we computed in a dictionary
#
package = {"lamcm":lam, "kabs":kabs, "kscat":kscat,
           "gscat":gscat, "matdens":matdens, "agraincm":a,
           "references":references,}
if theta is not None:
    package["zscat"] = np.copy(zscat)
    package["theta"] = np.copy(angles)
    package["kscat_from_z11"] = np.copy(kscat_from_z11)
if extrapolate:
    package["wavmic"] = np.copy(lammic)
    package["ncoef"] = np.copy(ncoefi)
    package["kcoef"] = np.copy(kcoefi)
if nagr>1:
    package["agr"] = np.copy(agr)
    package["wgt"] = np.copy(wgt)
    package["wfact"] = wfact
    package["logawidth"] = logawidth
if chopforward>0:
    package["zscat_nochop"] = np.copy(zscat_nochop)
    package["kscat_nochop"] = np.copy(kscat_nochop)
#return package

#
# Write radmc input file including zscat matrix
#
"""
The RADMC-3D radiative transfer package
  http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/
can perform dust continuum radiative transfer for diagnostic purposes.
It is designed for astronomical applications. The code
needs the opacities in a particular form. This subroutine
writes the opacities out in that form. It will write it to
the file dustkapscatmat_<name>.inp.
"""
filename = 'dustkapscatmat_'+name+'.inp'
ref = references
if descr is None: descr=name
with open(filename,'w+') as f:
    f.write('# Opacity and scattering matrix file for '+descr+'\n')
    if ref is not None:
        f.write('# Optical constants from '+ref+'\n')
    f.write('# Please do not forget to cite in your publications the original paper of these optical constant measurements\n')
    if(package["references"]!=''):
        refs = package["references"].split('\n')
        for r in refs:
            f.write('# @references = '+r+'\n')
    f.write('# Made with the makedustopac.py code by Cornelis Dullemond\n')
    f.write('# using the bhmie.py Mie code of Bohren and Huffman (python version by Cornelis Dullemond, from original bhmie.f code by Bruce Draine)\n')
    f.write('# Grain size distribution:\n')
    f.write('# agrain min = %13.6e cm\n'%(package['agraincm'][0]))
    f.write('# agrain max = %13.6e cm\n'%(package['agraincm'][-1]))
    f.write('# powerlaw index = %13.6f \n'%(pla))
    f.write('# na = %13d \n'%(na))
    f.write('# Material density:\n')
    f.write('# @density = %13.6f g/cm^3\n'%(package['matdens']))
    f.write('1\n')  # Format number
    f.write('%d\n'%(package['lamcm'].size))
    f.write('%d\n'%(package['theta'].size))
    f.write('\n')
    for i in range(package['lamcm'].size):
        f.write('%13.6e %13.6e %13.6e %13.6e\n'%(package['lamcm'][i]*1e4,
                                                 package['kabs'][i],
                                                 package['kscat'][i],
                                                 package['gscat'][i]))
    f.write('\n')
    for j in range(package['theta'].size):
        f.write('%13.6e\n'%(package['theta'][j]))
    f.write('\n')
    for i in range(package['lamcm'].size):
        for j in range(package['theta'].size):
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n'%
                    (package['zscat'][i,j,0],package['zscat'][i,j,1],
                     package['zscat'][i,j,2],package['zscat'][i,j,3],
                     package['zscat'][i,j,4],package['zscat'][i,j,5]))
        f.write('\n')


'''
wave1 = np.linspace(0.1e-4,1.0e-4,40,endpoint=False)
wave2 = np.linspace(1.0e-4,1.0e-3,30,endpoint=False)
wave3 = np.linspace(1.0e-3,2e0,20,endpoint=True)
wavel = np.concatenate([wave1,wave2,wave3])
n = np.zeros(len(wavel)); k = np.zeros(len(wavel))
for i in range(len(n)):
    n[i], k[i] = dc.nk(wavel[i])

with open('dsharp.lnk', 'w+') as f:
    f.write('# Optical constants for DSHARP mixed dust grains. From\n')
    f.write('# @reference = Birnstiel et al. (2018) ApJL 869 45.\n')
    f.write('# The material density is:\n')
    f.write('# @density = %6.3e g/cm^3\n' % rho_s)
    f.write('# When you use this file, please cite the above papers.\n')
    f.write('# Columns are: lambda [micron], n, k\n')
    for i in range(wavel.size):
        f.write('%13.6e %13.6e %13.6e \n' % (wavel[i]*1e4 , n[i], k[i]))

with open('wavelength_micron.inp', 'w+') as f:
    f.write('%8d \n' % (lam.size))
    for i in range(lam.size):
        f.write('%13.6e \n' % (lam[i]))
'''
