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

keyfile = 'keys/distance_key.txt'

########
# main #
########

# ファイルを読み込む
data  = np.loadtxt(keyfile, delimiter=',', dtype = 'unicode')

for i in range(len(data)):
    this_gal    = data[i,0]
    this_dist   = float(data[i,1]) * u.Mpc
    kpc_per_1as = (this_dist * math.tan(math.radians(1./3600.))).to(u.pc)
    as_per400pc = np.round(400./kpc_per_1as.value, 2)
    print(this_gal, as_per400pc)

#######
# end #
#######