import os, re, sys, glob, math, scipy, datetime, itertools
import numpy as np
import scipy.ndimage
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

### astropy
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord

### matplotlib
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from matplotlib.colors import LogNorm
plt.ioff()

### CASA imports
from taskinit import *
import analysisUtils as aU
from specsmooth import specsmooth
from imsubimage import imsubimage
from imval import imval
from impv import impv
from concat import concat
from immath import immath
from imhead import imhead
from imstat import imstat
from impbcor import impbcor
from imrebin import imrebin
from feather import feather
from imregrid import imregrid
from imsmooth import imsmooth
from makemask import makemask
from simalma import simalma
from immoments import immoments
from simobserve import simobserve
from importfits import importfits
from exportfits import exportfits
mycl = aU.createCasaTool(cltool)
mycs = aU.createCasaTool(cstool)
myia = aU.createCasaTool(iatool)
myrg = aU.createCasaTool(rgtool)
myqa = aU.createCasaTool(qatool)
