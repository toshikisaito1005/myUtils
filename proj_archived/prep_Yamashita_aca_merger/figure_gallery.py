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

csvfile = 'doc/GSWLC_v20230404.csv'
keyfile = 'doc/target_definitions.txt'

########
# main #
########
SkyView.list_surveys()

# ファイルを読み込む
data  = np.loadtxt(keyfile, dtype = 'unicode')
data2 = np.loadtxt(csvfile, delimiter=',', dtype = 'unicode')

os.system('rm -rf png/')
os.system('mkdir png/')

# 天体名を変更する
name = []
for this_name in data[:,0]:
    this_name = this_name.lower()
    name.append(this_name)

# 天体名を変更する
name2 = []
for this_name in data2[:,0]:
    this_name = this_name.lower()
    if 'disk' in this_name:
        this_name = 'ms' + this_name[-4:]
    name2.append(this_name)

redshift = []
for this_name in name:
    index = np.where(np.array(name2)==this_name.replace('vv00','vv').replace('vv0','vv'))[0]
    if len(index)>0:
        z = data2[index[0],13]
        redshift.append(z.astype(float))
    else:
        redshift.append(0.02)

# カタログ再構成
dat = np.c_[
    name,
    np.array(data[:,1]), # ra
    np.array(data[:,2]), # dec
    redshift,
    ]

def get_fits(
    survey,
    radius_min=60.,
    pixel=1000,
    ):

    try:
        hdu = SkyView.get_images(
            position=obj_coords,
            survey=survey,
            radius=radius_min/60.*u.deg,
            pixels=pixel)
    except:
        hdu = SkyView.get_images(
            position=obj_coords,
            survey=['WISE 12'],
            radius=3./60.*u.deg,
            pixels=800)
        hdu[0][0].data = hdu[0][0].data * 0

    return hdu

def get_pix(
    hdu,
    dist,
    ):

    try: 
        pixel_size = hdu[0][0].header['CDELT1']
        image_size = hdu[0][0].data.shape[0]
        arcsec_per_pixel = abs(pixel_size) * 3600
        kpc_per_pix = (dist * math.tan(math.radians(arcsec_per_pixel/3600.))).value * 1000
    except:
        pixel_size = hdu[0].header['CDELT1']
        image_size = hdu[0].data.shape[0]
        arcsec_per_pixel = abs(pixel_size) * 3600
        kpc_per_pix = (dist * math.tan(math.radians(arcsec_per_pixel/3600.))).value * 1000

    return [kpc_per_pix, arcsec_per_pixel]

def cutout_fits(
    hdu,
    pix,
    obj_coords,
    plot_length=25.,
    ):

    npix = plot_length / pix[0] * pix[1]
    size = u.Quantity((npix, npix), u.arcsec)

    try:
        stamp = Cutout2D(hdu[0][0].data, obj_coords, size, wcs=WCS(hdu[0][0].header))
    except:
        stamp = Cutout2D(hdu[0].data, obj_coords, size, wcs=WCS(hdu[0].header))

    return stamp

def cunstruct_beam(
    hdu,
    stamp,
    beam=None,
    ):

    if isinstance(beam, float)==True:
        my_beam = Beam(
            beam*u.arcsec,
            beam*u.arcsec,
            0*u.deg,
            )
        bvalue = beam
    else:
        beam_size = Beam.from_fits_header(hdu[0].header)
        my_beam = Beam(beam_size.major.value*3600*u.arcsec,
                       beam_size.minor.value*3600*u.arcsec,
                       beam_size.pa.value*u.deg,
                       )
        bvalue = np.round(beam_size.major.value*3600,1)

    ycen_pix, xcen_pix = np.shape(stamp.data)[0] * 0.12, np.shape(stamp.data)[1] * 0.12

    try:
        pixscale = hdu[0][0].header['CDELT1'] * 3600 * u.arcsec
    except:
        pixscale = hdu[0].header['CDELT1'] * 3600 * u.arcsec

    ellipse_artist = my_beam.ellipse_to_plot(xcen_pix, ycen_pix, pixscale)

    return ellipse_artist, bvalue

def cookfits(
    survey,
    obj_coords,
    dist,
    fitsimage=None,
    radius_min=60.,
    pixel=1000,
    beam=1.32,
    ):

    if fitsimage==None:
        hdu = get_fits(survey,radius_min,pixel)
    else:
        hdu = fits.open(glob.glob(fitsimage)[0])

    pix = get_pix(hdu,dist)
    stamp = cutout_fits(hdu,pix,obj_coords)
    ell,bval = cunstruct_beam(hdu,stamp,beam=beam)

    return hdu, pix, stamp, ell, bval

def plotax(
    ax,
    stamp,
    ell,
    cmap='Blues',
    survey="GALEX FUV",
    beam=4.3,
    kpc_per_1as=1.0,
    norm=None,
    ref=None,
    ):

    ax.imshow(stamp.data, origin="lower", cmap=cmap, alpha=1.0, norm=norm)
    txt = ax.text(0.95, 0.88, survey, color="black", ha="right", transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

    if np.sum(stamp.data)!=0:
        txt = ax.text(0.95, 0.09, r"$\theta$ = "+str(beam)+"$^{\prime}$$^{\prime}$ ("+str(np.round(kpc_per_1as*beam,1))+" kpc)",
            color="black", transform=ax.transAxes, ha="right", fontsize=8)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        _ = ax.add_artist(ell)

    if np.sum(stamp.data)!=0:
        ax.plot([0,np.shape(stamp.data)[0]-1],[np.shape(stamp.data)[1]/2,np.shape(stamp.data)[1]/2],'--',color='black',lw=0.5)
        ax.plot([np.shape(stamp.data)[0]/2,np.shape(stamp.data)[0]/2],[0,np.shape(stamp.data)[1]-1],'--',color='black',lw=0.5)

    if ref!=None:
        txt = ax.text(0.95, 0.78, ref, color="black", ha="right", transform=ax.transAxes, fontsize=8)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_ticks_visible(False)
    lon.set_ticklabel_visible(False)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)

# plot
for i in range(len(name)):
    #if 'ms9201' not in dat[i,0]:
    #    continue
    this_name   = dat[i,0]
    this_z      = redshift[i]
    dist        = cosmo.luminosity_distance(this_z)
    kpc_per_1as = (dist * math.tan(math.radians(1./3600.))).value * 1000
    print(this_name,dist)

    # Define the sky position you want to map
    obj_coords = SkyCoord(dat[i,1], dat[i,2], frame='icrs', unit=(u.hourangle, u.deg))

    # import sdss
    hdu_sdss,pix_sdss,stamp_sdss,ell_sdss,_ = cookfits(['SDSSr'],obj_coords,dist,None,10.,2000,1.32)
    norm = ImageNormalize(stamp_sdss.data, interval=MinMaxInterval(),stretch=LogStretch())
    wcs = WCS(hdu_sdss[0][0].header)

    # import alma mom0
    fitsimage = 'data_raw/' + this_name + '/' + this_name + '_7m_co21_strict_mom0.fits'
    hdu_mom0,pix_mom0,stamp_mom0,ell_mom0,alma_beam = cookfits(None,obj_coords,dist,fitsimage,None,None,None)

    # import alma mom1
    fitsimage = 'data_raw/' + this_name + '/' + this_name + '_7m_co21_strict_mom1.fits'
    hdu_mom1,pix_mom1,stamp_mom1,ell_mom1,_ = cookfits(None,obj_coords,dist,fitsimage,None,None,None)

    # import alma 230GHz
    fitsimage = 'data_raw/' + this_name + '/' + this_name + '_7m_cont_pbcor.fits'
    hdu_cont,pix_cont,stamp_cont,ell_cont,_ = cookfits(None,obj_coords,dist,fitsimage,None,None,None)

    # import wise
    hdu_wise,pix_wise,stamp_wise,ell_wise,_ = cookfits(['WISE 12'],obj_coords,dist,None,3.,800,6.5)

    # import galex
    hdu_galex,pix_galex,stamp_galex,ell_galex,_ = cookfits(['GALEX Far UV'],obj_coords,dist,None,3.,800,4.3)

    # import vla first
    hdu_vla,pix_vla,stamp_vla,ell_vla,_ = cookfits(['VLA FIRST (1.4 GHz)'],obj_coords,dist,None,10.,1000,5.4)

    # plot
    fig = plt.figure(figsize=(14,4))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(1,7,1,projection=wcs)
    ax2 = fig.add_subplot(1,7,2,projection=wcs)
    ax3 = fig.add_subplot(1,7,3,projection=wcs)
    ax4 = fig.add_subplot(1,7,4,projection=wcs)
    ax5 = fig.add_subplot(1,7,5,projection=wcs)
    ax6 = fig.add_subplot(1,7,6,projection=wcs)
    ax7 = fig.add_subplot(1,7,7,projection=wcs)

    txt = ax1.text(0.05, 0.88, this_name, color="black", transform=ax1.transAxes, weight="bold")
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
    stamp_mom0.data = np.where(stamp_mom0.data>0,stamp_mom0.data,np.nan)

    plotax(ax1,stamp_sdss,ell_sdss,'Greys',"SDSS r",1.32,kpc_per_1as,norm)#,"(Abdurro'uf et al. 2022)")
    plotax(ax2,stamp_wise,ell_wise,'Reds',"WISE 12",6.5,kpc_per_1as,None)
    plotax(ax3,stamp_galex,ell_galex,'Blues',"GALEX FUV",4.3,kpc_per_1as,None)
    plotax(ax4,stamp_vla,ell_vla,'Greens',"VLA FIRST 1.4GHz",5.4,kpc_per_1as,None)#,"(Becker et al. 1995)")
    plotax(ax5,stamp_cont,ell_cont,'rainbow',"ALMA 230GHz",alma_beam,kpc_per_1as,None)#,"(This work)")
    plotax(ax6,stamp_mom0,ell_mom0,'rainbow',"ALMA CO(2-1)",alma_beam,kpc_per_1as,None)#,"(This work)")
    plotax(ax7,stamp_mom1,ell_mom1,'rainbow',"CO Vel. Field",alma_beam,kpc_per_1as,None)#,"(This work)")

    plt.savefig('png/map_'+this_name+'.png', bbox_inches='tight', pad_inches=0.05, dpi=150)

#######
# end #
#######