import os, glob, math
import numpy as np
from reproject import reproject_interp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from radio_beam import Beam
from astropy.nddata import Cutout2D
from astroquery.sdss import SDSS
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import matplotlib.patheffects as PathEffects
from astropy.cosmology import WMAP9 as cosmo
from astroquery.skyview import SkyView
from astropy.visualization import (MinMaxInterval,
                                  LogStretch,
                                  ImageNormalize)

keyfile_dist = 'keys/distance_key.txt'

#
data_dist = np.loadtxt(keyfile_dist, delimiter=',', dtype = 'str')

for i in range(len(data_dist)):
    this_gal    = data_dist[i,0]
    fitsimage   = glob.glob(this_gal+'_12m_cont.image')#'_12m_cont_400pc.fits')

    if not fitsimage:
        continue

    hdu = fits.open(fitsimage[0])[0]
    data = hdu.data[0,0,:,:]

    # plot
    fig = plt.figure(figsize=(14,4))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(data, origin="lower", cmap='gnuplot', alpha=1.0)
    plt.savefig('map_'+this_gal+'.png', bbox_inches='tight', pad_inches=0.05, dpi=150)

#######
# end #
#######