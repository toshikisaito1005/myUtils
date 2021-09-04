"""
Standalone routines that are used for stacking data cubes using CASA.

contents:
    radial_stacking
    cube_stacking

history:
2021-07-26   created
2021-07-28   refactored
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys
import mycasa_tasks as mytask
reload(mytask)

execfile(os.environ["HOME"] + "/myUtils/stuff_casa.py")

#

###########################
### radial_stacking
###########################

###########################
### cube_stacking
###########################

def cube_stacking(
    cubeimage,
    velimage,
    binimage,
    maskimage,
    ra_cnt,
    dec_cnt,
    masking=True,
    nbins=8,
    ):
    """
    Run shuffling stacking.

    Parameters
    ----------
    cubeimage : str
        3D CASA or FITS datacube used for stacking.
    velimage : str
        2D CASA or FITS map used for velocity prior.
    binimage : str
        2D CASA or FITS map used for stacking axis.
    maskimage : str
        2D CASA or FITS map used for masking cubeimage.

    CASA tasks
    ----------
    many

    References
    ----------
    https://ui.adsabs.harvard.edu/abs/2011AJ....142...37S
    """

    taskname = sys._getframe().f_code.co_name
    mytask.check_first(cubeimage, taskname)
    mytask.check_first(velimage)
    mytask.check_first(binimage)
    mytask.check_first(maskimage)
    
    # regridding to a common grid
    cubeimage = _do_regrid(cubeimage, binimage, axes=[0,1])
    velimage  = _do_regrid(velimage, binimage, axes=[0,1])
    maskimage = _do_regrid(maskimage, cubeimage)
    
    # shuffling against velocity
    l = _do_shuffling(cubeimage,velimage,binimage,maskimage)

    # stacking shuffled cube
    stack_bins, stack_specs = _do_stacking(
        l, ra_cnt, dec_cnt, maskr=8.0, logbin=True, nbins=nbins)
    
    return stack_bins, stack_specs

#

def _do_stacking(
    shuffled_data,
    ra_cnt,
    dec_cnt,
    maskr=0.0,
    vrange=[-300,300],
    bintype="step",
    nbins=8,
    logbin=False,
    output_binimage=None,
    output_spec=None,
    ):
    """
    Parameters
    ----------
    ra_cnt : float (degree)
    dec_cnt : float (degree)
    maskr : float (arcsec)
        radius of the central aperture at (ra_cnt, dec_cnt),
        which will be flagged before stacking.
    vrange : list, float or int
        velocity range for stacking.
    bintype : str
        binnning scheme
        step = even-spaced binning.
        num  = even-numbered binning.
    nbins : int
        number of bins for stacking.
    logbin : boolean
        binning in linear or log scale.
    """
    
    print("# stacking")
    
    # get data
    ra       = shuffled_data[0]
    dec      = shuffled_data[1]
    data     = shuffled_data[2]
    data_vel = shuffled_data[3]
    data_bin = shuffled_data[4]
    
    # get distance mask and apply the mask
    x         = (ra - ra_cnt) * 3600
    y         = (dec - dec_cnt) * 3600
    r         = np.sqrt(x**2 + y**2)
    vbin      = abs(data_vel[0,0,1]-data_vel[0,0,0])

    ra        = np.where(r>=maskr, x, 0)
    dec       = np.where(r>=maskr, y, 0)
    data_cube = np.where(r>=maskr, data, 0)
    data_vel  = np.where(r>=maskr, data_vel, 0)
    data_bin  = np.where(r>=maskr, data_bin, 0)
    vel       = np.arange(vrange[0], vrange[1], vbin)
    
    # get bins
    if bintype=="step":
        bins = np.linspace(np.min(data_bin), np.max(data_bin), nbins+1)
    
    elif bintype=="num":
        bins = []
        for i in range(nbins):
            bins.append( np.percentile(data_bin, 100./float(nbins)) )

        bins.append(np.percentile(data_bin,100))
        bins = np.array(bins)

    # plot the bin image
    if output_binimage!=None:
        _plot_image(ra, dec, data_bin, bins, output_binimage)

    # stacking
    stacked_bins    = []
    stacked_spectra = []
    
    print("# stacking")
    for i in range(len(bins)-1):
        print("# " + str(i+1) + "th stacking")
        
        # extract data within this bin
        bin_l, bin_r  = [bins[i], bins[i+1]]
        bin_m         = (bin_l+bin_r) / 2.0
        cut = np.where((data_bin>bin_l)&(data_bin<=bin_r))
        this_vel  = data_vel[cut]
        this_data = data_cube[cut]
        this_vel  = this_vel.flatten()
        this_data = this_data.flatten()
        
        this_spectrum = []
        
        for j in range(len(vel)-1):
            # get this velocity channel bin
            vel_l, vel_r = vel[j], vel[j+1]
            cut = np.where((this_vel>vel_l) & (this_vel<=vel_r))
            
            this_chan_data = this_data[cut]
            this_value = np.average(this_chan_data) # including 0
            this_spectrum.append(this_value)

        stacked_bins.append(bin_m)
        stacked_spectra.append(this_spectrum)

        # plot the stacked spectum at this bin
        if output_spec!=None:
            print("ToDo plot stacked spectra")
        
        spec = np.c_[np.array(vel[:-1]), np.array(stacked_spectra).transpose()]

    return np.array(stacked_bins), spec

#

def _plot_image(
    ra,
    dec,
    data_bin,
    bins,
    output,
    ):
    
    print("# plot binimage")

    figname = binimage.split("/")[-1].split(".fits")[0].split(".image")[0]
    
    # plt
    fig = plt.figure(figsize=(9,9))
    plt.subplots_adjust(bottom=0.10, left=0.15, right=0.95, top=0.90)
    gs  = gridspec.GridSpec(nrows=3, ncols=3)
    ax  = plt.subplot(gs[0:3,0:3])

    for i in range(len(bins)-1):
        c   = cm.gnuplot(i/float(len(bins)-1))
        
        cut = np.where((data_bin>bins[i]) & (data_bin<=bins[i+1]))
        x   = ra[cut]
        y   = dec[cut]
        
        ax.scatter(x, y, c=c, lw=0, s=4, marker="s")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    lim = np.max([np.max(x),np.max(y)])
    ax.set_xlim([lim,-lim])
    ax.set_ylim([-lim,lim])
    plt.savefig(output, dpi=300)

#

def _do_shuffling(
    imagename,
    velimage,
    binimage,
    maskimage,
    masking=True,
    ):
    
    print("# shuffling " + imagename.split("/")[-1])
    
    # masking if required
    mytask.run_immath_one(maskimage,maskimage+"_tmp1","iif(IM0>0,1,0)")

    if masking==True:
        expr = "IM0*IM1"
    else:
        expr = "IM0"

    mytask.run_immath_two(
        imagename,
        maskimage + "_tmp1",
        imagename + "_tmp1",
        expr,
        )

    # get parameters
    c          = 299792.458 # km/s
    
    data,_     = mytask.imval_all(imagename+"_tmp1")
    data_cube  = data["data"]
    data_cube[np.isnan(data_cube)] = 0

    freq       = data["coords"][:,:,:,2]
    ra         = data["coords"][:,:,:,0] * 180/np.pi
    dec        = data["coords"][:,:,:,1] * 180/np.pi

    data_vel,_ = mytask.imval_all(velimage)
    data_vel   = data_vel["data"]
    data_vel[np.isnan(data_vel)] = 0
    data_vel   = np.tile(data_vel,(np.shape(ra)[2],1,1)).transpose(1,2,0)
    restfreq   = imhead(imagename+"_tmp1",mode="get",hdkey="restfreq")["value"]
    vel        = np.array( (restfreq-freq) / restfreq * c ) # km/s
    relvel     = vel - data_vel

    data_bin,_ = mytask.imval_all(binimage)
    data_bin   = data_bin["data"]
    data_bin   = np.tile(data_bin,(np.shape(ra)[2],1,1)).transpose(1,2,0)
    data_bin[np.isnan(data_bin)] = 0

    ra        = np.where((relvel!=0)&(data_bin>0),ra,0)
    dec       = np.where((relvel!=0)&(data_bin>0),dec,0)
    data_cube = np.where((relvel!=0)&(data_bin>0),data_cube,0)
    #relvel    = np.where((relvel!=0)&(data_bin>0),relvel,0)
    data_bin  = np.where((relvel!=0)&(data_bin>0),data_bin,0)

    os.system("rm -rf " + imagename + "_tmp1")
    os.system("rm -rf " + maskimage + "_tmp1")

    return ra, dec, data_cube, relvel, data_bin

#

def _do_regrid(imagename,template,axes=-1):

    imshape,  impix  = _get_grid(imagename)
    tmpshape, tmppix = _get_grid(template)
    
    if impix!=tmppix or imshape[0]!=tmpshape[0] or imshape[1]!=tmpshape[1]:
        print("# regrid " + imagename.split("/")[-1])
        if template[-5:]==".fits":
            mytask.run_importfits(
                template,
                template + ".image",
                defaultaxes = True,
                )
            template = template + ".image"
            
        mytask.run_imregrid(
            imagename,
            template,
            imagename + ".image",
            axes = axes,
            )
            
        os.system("rm -rf " + template + ".image")
        imagename = imagename + ".image"
    
    else:
        print("# skip regrid" + imagename.split("/")[-1])
    
    return imagename

#

def _get_grid(imagename):

    print("# get grid " + imagename.split("/")[-1])

    head  = imhead(imagename,mode="list")
    shape = head["shape"][0:2]
    pix   = abs(head["cdelt1"]) * 3600 * 180/np.pi
    
    return shape, pix
