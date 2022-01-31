"""
Standalone routines that are used for rotation diagram using CASA.

contents:
    hf_cn10

history:
2021-12-18   created
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, pyfits
import mycasa_tasks as mytask
reload(mytask)
import mycasa_plots as myplot
reload(myplot)

execfile(os.environ["HOME"] + "/myUtils/stuff_casa.py")

###############
# fitting_two #
###############

def fitting_two(
    cubelow,
    cubehigh,
    ecubelow,
    ecubehigh,
    restfreq_low=None,
    restfreq_high=None,
    ra_cnt=40.669625, # deg
    dec_cnt=-0.01331667, # deg
    snr=10.0,
    smooth=0,
    ratio_max=2.0,
    ):
    """
    """

    # preamble
    taskname = sys._getframe().f_code.co_name
    mytask.check_first(cubelow, taskname)

    # constants
    if restfreq_low==None:
        header_low    = imhead(cubelow,mode="list")
        restfreq_low  = header_low["restfreq"][0] / 1e9

    if restfreq_high==None:
        header_high   = imhead(cubehigh,mode="list")
        restfreq_high = header_high["restfreq"][0] / 1e9

    # read cube
    data_low,err_low,freq_low,ra_deg,dec_deg = _get_data(cubelow,ecubelow,ra_cnt,dec_cnt)
    data_high,err_high,freq_high,_,_         = _get_data(cubehigh,ecubehigh,ra_cnt,dec_cnt)

    ra  = ra_deg[:,:,0] * 3600
    dec = dec_deg[:,:,0] * 3600

    # fitting spectra
    bw  = smooth
    lim = np.max([np.max(abs(ra)), np.max(abs(dec))])
    x   = range(np.shape(data_low)[0])
    y   = range(np.shape(data_low)[1])
    xy  = itertools.product(x, y)

    mom0_low  = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    mom0_high = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    mom1      = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    mom2      = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))

    for i in xy:
        # get data of this sightline
        this_x,this_y  = i[0],i[1]

        this_freq_low  = freq_low[this_x, this_y]
        this_freq_high = freq_high[this_x, this_y]

        this_data_low  = data_low[this_x, this_y]
        this_data_high = data_high[this_x, this_y]

        this_err_low   = err_low[this_x, this_y]
        this_err_high  = err_high[this_x, this_y]

        #this_data_low  = np.mean(data_low[max(0,this_x-bw):this_x+1+bw, max(0,this_y-bw):this_y+1+bw],axis=(0,1)) # data_low[this_x, this_y]
        #this_data_high = np.mean(data_high[max(0,this_x-bw):this_x+1+bw, max(0,this_y-bw):this_y+1+bw],axis=(0,1)) # data_high[this_x, this_y]

        # combine two data
        this_freq = np.r_[this_freq_low, this_freq_high]
        this_data = np.r_[this_data_low, this_data_high]
        this_err  = np.r_[this_err_low, this_err_high]

        # p0 guess
        p0 = [
        np.max(this_data)/2.0,
        np.max(this_data),
        (restfreq_low - this_freq_low[np.nanargmax(this_data_low)]) / restfreq_low * 299792.458,
        40.,
        ]

        # fit
        if np.max(this_data_low/this_err_low)>=snr and np.max(this_data_high/this_err_high)>=snr:
            # fitting
            this_f_two = lambda x, a1, a2, b, c: _f_two(x, a1, a2, b, c, restfreq_low, restfreq_high)
            popt,pcov = curve_fit(this_f_two,this_freq,this_data,sigma=this_err,p0=p0,maxfev=100000)

            if popt[1]/popt[0]>0 and popt[1]/popt[0]<=ratio_max
                # add pixel
                mom0_low[this_x,this_y]  = popt[0] * popt[3] * np.sqrt(2*np.pi)
                mom0_high[this_x,this_y] = popt[1] * popt[3] * np.sqrt(2*np.pi)
                mom1[this_x,this_y]      = popt[2]
                mom2[this_x,this_y]      = popt[3]
            else:
                # add pixel
                mom0_low[this_x,this_y]  = 0
                mom0_high[this_x,this_y] = 0
                mom1[this_x,this_y]      = 0
                mom2[this_x,this_y]      = 0
        else:
            # add pixel
            mom0_low[this_x,this_y]  = 0
            mom0_high[this_x,this_y] = 0
            mom1[this_x,this_y]      = 0
            mom2[this_x,this_y]      = 0

    # ratio
    ratio = mom0_high.T/mom0_low.T

    # fits
    fits_creation(mom0_low.T,"mom0_low.fits")
    fits_creation(mom0_high.T,"mom0_high.fits")
    fits_creation(ratio,"ratio.fits")
    fits_creation(mom1.T,"mom1.fits")
    fits_creation(mom2.T,"mom2.fits")

"""
def fitting_two(
    cubelow,
    cubehigh,
    ra_cnt=40.669625, # 1068 agn, deg
    dec_cnt=-0.01331667, # 1068 agn, deg
    box="92,103,363,333",
    factor=None,
    snr=7.0,
    smooth=0,
    ):

    # preamble
    taskname = sys._getframe().f_code.co_name
    mytask.check_first(cubelow, taskname)

    cubelow_original = cubelow
    cubehigh_original = cubehigh

    # constants
    header_low    = imhead(cubelow,mode="list")
    header_high   = imhead(cubehigh,mode="list")
    restfreq_low  = header_low["restfreq"][0] / 1e9
    restfreq_high = header_high["restfreq"][0] / 1e9

    # read cube
    data_low,freq_low,ra_deg,dec_deg = _get_data(cubelow,ra_cnt,dec_cnt)
    data_high,freq_high,_,_ = _get_data(cubehigh,ra_cnt,dec_cnt)
    ra = ra_deg[:,:,0] * 3600
    dec = dec_deg[:,:,0] * 3600

    # measure noise
    rms_low = plot_hist(data_low,"plot_voxel_hist_cubelow.png")
    snr_low = np.max(data_low) / rms_low

    rms_high = plot_hist(data_high,"plot_voxel_hist_cubehigh.png")
    snr_high = np.max(data_high) / rms_high

    os.system("rm -rf plot_voxel_hist_cubelow.png plot_voxel_hist_cubehigh.png")

    # determine the base data
    index     = np.argmax([snr_low,snr_high])
    this_data = [data_low,data_high][index]
    this_rms  = [rms_low,rms_high][index]
    this_snr  = [snr_low,snr_high][index]
    this_cube = [cubelow,cubehigh][index]

    print("### " + this_cube.split("/")[-1] + " has better SNR.")
    print("# rms     = " + str(np.round(this_rms,4)))
    print("# max SNR = " + str(np.round(this_snr,1)))
    print("")

    # fitting spectra
    bw  = smooth
    lim = np.max([np.max(abs(ra)), np.max(abs(dec))])
    x   = range(np.shape(data_low)[0])
    y   = range(np.shape(data_low)[1])
    xy  = itertools.product(x, y)

    mom0_low  = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    mom0_high = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    mom1      = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    mom2      = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))

    for i in xy:
        this_x,this_y  = i[0],i[1]

        this_freq_high = freq_high[this_x, this_y]
        this_freq_low  = freq_low[this_x, this_y]
        chanwidth_GHz  = abs(this_freq_high[1] - this_freq_high[0])
        this_data_low  = np.mean(data_low[max(0,this_x-bw):this_x+1+bw, max(0,this_y-bw):this_y+1+bw],axis=(0,1)) # data_low[this_x, this_y]
        this_data_high = np.mean(data_high[max(0,this_x-bw):this_x+1+bw, max(0,this_y-bw):this_y+1+bw],axis=(0,1)) # data_high[this_x, this_y]

        this_freq = np.r_[this_freq_low, this_freq_high]
        this_data = np.r_[this_data_low, this_data_high]

        # p0 guess
        guess_b = (restfreq_low - this_freq_low[np.nanargmax(this_data_low)]) / restfreq_low * 299792.458

        p0 = [
        np.max(this_data)/2.0,
        np.max(this_data),
        guess_b,
        40.,
        ]

        # fit
        if np.max(this_data)<this_rms*snr:
            # add pixel
            mom0_low[this_x,this_y]  = 0
            mom0_high[this_x,this_y] = 0
            mom1[this_x,this_y]      = 0
            mom2[this_x,this_y]      = 0
        else:
            # fitting
            this_f_two = lambda x, a1, a2, b, c: _f_two(x, a1, a2, b, c, restfreq_low, restfreq_high)
            popt,pcov = curve_fit(this_f_two,this_freq,this_data,p0=p0,maxfev=100000)

            # add pixel
            mom0_low[this_x,this_y]  = popt[0]
            mom0_high[this_x,this_y] = popt[1]
            mom1[this_x,this_y]      = popt[2]
            mom2[this_x,this_y]      = popt[3]

            # plot
            if this_x==4 and this_y==23:
                print(p0)
                print(popt[0],popt[1],popt[2],popt[3])

                fig = plt.figure(figsize=(13,7))
                plt.rcParams["font.size"] = 22
                plt.rcParams["legend.fontsize"] = 20

                gs = gridspec.GridSpec(nrows=30, ncols=30)
                ax1 = plt.subplot(gs[0:15,0:15])
                ax2 = plt.subplot(gs[0:15,15:30])
                ax3 = plt.subplot(gs[15:30,0:30])
                ax1.grid(axis="both", ls="--")
                ax2.grid(axis="both", ls="--")
                ax3.grid(axis="both", ls="--")

                ax1.plot(this_freq, this_data, "-", lw=2, color="tomato")
                ax1.plot(this_freq, this_f_two(this_freq, *popt), "-", lw=2, color="deepskyblue", alpha=0.5)
                ax1.plot(this_freq, this_f_two(this_freq, *p0), "-", lw=2, color="black", alpha=0.5)

                ax2.plot(this_freq, this_data, "-", lw=2, color="tomato")
                ax2.plot(this_freq, this_f_two(this_freq, *popt), "-", lw=2, color="deepskyblue", alpha=0.5)
                ax2.plot(this_freq, this_f_two(this_freq, *p0), "-", lw=2, color="black", alpha=0.5)

                ax3.plot(this_freq, this_data, "-", lw=2, color="tomato")
                ax3.plot(this_freq, this_f_two(this_freq, *popt), "-", lw=2, color="deepskyblue", alpha=0.5)
                ax3.plot(this_freq, this_f_two(this_freq, *p0), "-", lw=2, color="black", alpha=0.5)

                ax1.set_xlim([87.52,87.72])
                ax2.set_xlim([109.425,109.625])

                peak = this_freq[np.argmax(this_f_two(this_freq, *popt))]
                plt.title( str(np.round(peak,3)) )

                plt.savefig("spectra.png", dpi=100)

    # fits
    fits_creation(mom0_low.T,"mom0_low.fits")
    fits_creation(mom0_high.T,"mom0_high.fits")
    fits_creation(mom0_high.T/mom0_low.T,"ratio.fits")
    fits_creation(mom1.T,"mom1.fits")
    fits_creation(mom2.T,"mom2.fits")

    # cleanup
    os.system("rm -rf " + cubelow_original + ".boxed")
    os.system("rm -rf " + cubehigh_original + ".boxed")
    os.system("rm -rf " + cubelow_original + ".regrid")
    os.system("rm -rf " + cubehigh_original + ".regrid")
    os.system("rm -rf " + cubelow_original + ".regrid.boxed")
    os.system("rm -rf " + cubehigh_original + ".regrid.boxed")
"""

#################
# fits_creation #
#################

def fits_creation(
    input_map,
    output_map,
    ):
    """
    """

    os.system("rm -rf " + output_map)
    hdu = pyfits.PrimaryHDU(input_map)
    hdul = pyfits.HDUList([hdu])
    hdul.writeto(output_map)

##########
# _f_two #
##########

def _f_two(x, a1, a2, b, c, freq1, freq2):

    offset1 = b /299792.458 * freq1 # km/s
    offset2 = b /299792.458 * freq2 # km/s

    width1 = c /299792.458 * freq1 # km/s
    width2 = c /299792.458 * freq2 # km/s

    func = \
        a1 * np.exp( -(x-freq1+offset1)**2/(2*width1**2) ) + \
        a2 * np.exp( -(x-freq2+offset2)**2/(2*width2**2) )

    return func

############
# _f_gauss #
############

def _f_noise(x, a, c):

    return a*np.exp(-(x)**2/(2*c**2))

#############
# _get_data #
#############

def _get_data(
    cubeimage,
    cubeerr,
    ra_cnt,
    dec_cnt,
    ):
    """
    """

    data,_    = mytask.imval_all(cubeimage)
    err,_     = mytask.imval_all(cubeerr)
    coords    = data["coords"]
    data      = data["data"]
    err       = err["data"]   
    ra_deg    = coords[:,:,:,0] * 180/np.pi - ra_cnt
    dec_deg   = coords[:,:,:,1] * 180/np.pi - dec_cnt
    freq      = coords[:,:,:,2] / 1e9

    data[np.isnan(data)] = 0
    err[np.isnan(err)]   = 0
    spec_data = np.sum(data,axis=(0,1))
    spec_err  = np.sum(err,axis=(0,1))

    index     = np.where(spec_data==0)[0]
    data      = np.delete(data,index,2)
    err       = np.delete(err,index,2)
    freq      = np.delete(freq,index,2)

    index     = np.where(spec_err==0)[0]
    data      = np.delete(data,index,2)
    err       = np.delete(err,index,2)
    freq      = np.delete(freq,index,2)

    return data, err, freq, ra_deg, dec_deg

#############
# _get_grid #
#############

def _get_grid(imagename):

    print("# _get_grid " + imagename.split("/")[-1])

    head  = imhead(imagename,mode="list")
    shape = head["shape"][0:2]
    pix   = abs(head["cdelt1"]) * 3600 * 180/np.pi
    
    return shape, pix

#######
# end #
#######
