import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from scipy.optimize import curve_fit

#imsubimage(
#    imagename = 'autoIRAS_180891732_spw1.12m.tc_final.fits',
#    chans = '400~448',
#    outfile = 'autoIRAS_180891732_spw1.12m.tc_final.image',
#    )
#exportfits(
#    imagename = 'autoIRAS_180891732_spw1.12m.tc_final.image',
#    fitsimage = 'autoIRAS_180891732_spw1.12m.tc_final_resize.fits',
#    dropstokes = True,
#    dropdeg = True,
#    )

fitsimage = 'autoIRAS_180891732_spw1.12m.tc_final_resize.fits'
rms = 0.003 # Jy/beam, very roughly
snr = 3.0   # signal-to-noise ratio

########
# main #
########

# import data
with fits.open(fitsimage) as hdul:
    cube = hdul[0].data
    header = hdul[0].header
    wcs = WCS(header)

# get grid
nz, ny, nx = cube.shape
cube[np.isnan(cube)] = 0

# init outputs
best_fit_cube = np.zeros((nz, ny, nx))
residual_cube = np.zeros((nz, ny, nx))
red_cube      = np.zeros((nz, ny, nx))
blue_cube     = np.zeros((nz, ny, nx))
best_fit_cube[best_fit_cube==0] = np.nan
residual_cube[residual_cube==0] = np.nan
red_cube[red_cube==0]           = np.nan
blue_cube[blue_cube==0]         = np.nan

# get velocity resolution
restfreq = header['RESTFRQ'] * u.Unit(header['CUNIT3'])
freq_res = abs(header['CDELT3']) * u.Unit(header['CUNIT3'])
freq_to_vel = u.doppler_radio(restfreq)
vel_res = (restfreq-freq_res).to(u.km / u.s, equivalencies=freq_to_vel)

def func(x, a1, b1, c1):
    return a1 * np.exp(-(x - b1) ** 2 / (2 * c1 ** 2))

def numpy2fits(
    infile,
    vel_res,
    out_prefix,
    ):

    mom0 = np.sum(infile, axis=0) * vel_res
    mom1 = np.argmax(infile, axis=0)
    mom1 = np.where(mom1!=0, (mom1 - np.nanmean(mom1)) * vel_res, np.nan)
    mom2 = np.sum(infile, axis=0) * vel_res / np.max(infile, axis=0) / np.sqrt(2*np.pi)
    mom8 = np.max(infile, axis=0)

    os.system('rm -rf ' + out_prefix + '_mom0.fits')
    hdu = fits.PrimaryHDU(mom0, header=header)
    hdu.writeto(out_prefix + '_mom0.fits')

    os.system('rm -rf ' + out_prefix + '_mom1.fits')
    hdu = fits.PrimaryHDU(mom1, header=header)
    hdu.writeto(out_prefix + '_mom1.fits')

    os.system('rm -rf ' + out_prefix + '_mom8.fits')
    hdu = fits.PrimaryHDU(mom8, header=header)
    hdu.writeto(out_prefix + '_mom8.fits')

# loop over the grid and fit
for i in range(ny):
    for j in range(nx):

        # get spectrum
        spectrum = cube[:, i, j]
        spectrum[np.isnan(spectrum)] = 0
        peak = np.nanmax(spectrum)

        # running mean convolution using 3x3 box (still smaller than beam?)
        if i>1 and j>1:
            spectrum = np.average(np.average(cube[:, i-1:i+2, j-1:j+2],axis=1),axis=1)

        # skip when faint line-of-sight
        if peak < rms * snr:
            best_fit_cube[:, i, j] = np.nan
            residual_cube[:, i, j] = np.nan
            red_cube[:, i, j]      = np.nan
            blue_cube[:, i, j]     = np.nan

            continue

        # intial guess (sigma in unts of channel)
        p0 = [peak, np.nanargmax(spectrum), 3]

        # do fitting to spectrum
        try:
            popt1, _ = curve_fit(func, np.arange(nz), spectrum, p0=p0)
            residual_cube[:, i, j] = spectrum - func(np.arange(nz), *popt1)

            # skip when faint resdual 
            if np.max(residual_cube[:, i, j]) < rms * snr:
                best_fit_cube[:, i, j] = np.nan
                residual_cube[:, i, j] = np.nan
                red_cube[:, i, j]      = np.nan
                blue_cube[:, i, j]     = np.nan

                continue

            # intial guess
            p0 = [peak, np.nanargmax(residual_cube[:, i, j]), 3]

            # do fitting when bright residual
            try:
                popt2, _ = curve_fit(func, np.arange(nz), residual_cube[:, i, j], p0=p0)
                best_fit_cube[:, i, j] = func(np.arange(nz), *popt1) + func(np.arange(nz), *popt2)
                residual_cube[:, i, j] = spectrum - func(np.arange(nz), *popt2)

                # identify blue- and red-shfited components
                if popt1[1]>popt2[1]:
                    red_cube[:, i, j]  = func(np.arange(nz), *popt2)
                    blue_cube[:, i, j] = func(np.arange(nz), *popt1)
                else:
                    red_cube[:, i, j]  = func(np.arange(nz), *popt1)
                    blue_cube[:, i, j] = func(np.arange(nz), *popt2)

            except:
                best_fit_cube[:, i, j] = np.nan
                residual_cube[:, i, j] = np.nan
                red_cube[:, i, j]      = np.nan
                blue_cube[:, i, j]     = np.nan

        except:
            best_fit_cube[:, i, j] = np.nan
            residual_cube[:, i, j] = np.nan
            red_cube[:, i, j]      = np.nan
            blue_cube[:, i, j]     = np.nan

# export: red component
os.system('rm -rf red_cube.fits')
hdu = fits.PrimaryHDU(red_cube, header=header)
hdu.writeto('red_cube.fits')
numpy2fits(red_cube,vel_res.value,"red")

# export: blue component
os.system('rm -rf blue_cube.fits')
hdu = fits.PrimaryHDU(blue_cube, header=header)
hdu.writeto('blue_cube.fits')
numpy2fits(blue_cube,vel_res.value,"blue")

# export: red+blue component
os.system('rm -rf best_cube.fits')
hdu = fits.PrimaryHDU(best_fit_cube, header=header)
hdu.writeto('best_cube.fits')
numpy2fits(best_fit_cube,vel_res.value,"best")

# export: obs data
numpy2fits(cube,vel_res.value,"data")

#######
# end #
#######