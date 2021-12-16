"""
Standalone routines that are used for hyperfine fitting using CASA.

contents:
    hf_cn10

history:
2021-12-14   created
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys
import mycasa_tasks as mytask
reload(mytask)
import mycasa_plots as myplot
reload(myplot)

execfile(os.environ["HOME"] + "/myUtils/stuff_casa.py")

#

###########
# hf_cn10 #
###########

def hf_cn10(
    cubeimage,
    snr=10.0,
    ra_cnt=40.669625, # 1068 agn, deg
    dec_cnt=-0.01331667, # 1068 agn, deg
    outpng_hist="plot_voxel_hist.png",
    ):
    """
    Reference:
    CN10 rest frequencies: Skatrud et al. 1983, Journal of Mol. Spec. 99, 35
    neiboring 8 pixels: https://qiita.com/yuji96/items/bfae04a043d35260ffb1
    """

    # preamble
    taskname = sys._getframe().f_code.co_name
    mytask.check_first(cubeimage, taskname)

    # constants
    z          = 0.00379
    freq_cn10l = 113.123337/(1+z)
    freq_cn10h = 113.488126/(1+z)

    # get cube
    data,_  = mytask.imval_all(cubeimage)
    coords  = data["coords"]
    data    = data["data"]
    ra_deg  = coords[:,:,:,0] * 180/np.pi - ra_cnt
    dec_deg = coords[:,:,:,1] * 180/np.pi - dec_cnt
    freq    = coords[:,:,:,2] / 1e9
    data[np.isnan(data)] = 0

    # remove NaN from the cube
    spec_mean = np.sum(data,axis=(0,1))
    index     = np.where(spec_mean==0)[0]
    data      = np.delete(data,index,2)
    freq      = np.delete(freq,index,2)

    # measure noise
    rms = plot_hist(data,outpng_hist)
    print("rms", rms)

    # extract neiboring 8 cells and gaussian fit
    ra  = ra_deg[:,:,0] * 3600
    dec = dec_deg[:,:,0] * 3600
    lim = np.max([np.max(abs(ra)), np.max(abs(dec))])

    x   = range(np.shape(data)[0])
    y   = range(np.shape(data)[1])
    xy  = itertools.product(x, y)

    tau      = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    mom0     = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    mom1     = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    mom2     = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    mom0_res = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    ratio    = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    r_squared_index = [0,0]
    r_squared_value = 100.0
    for i in xy:
        bw                 = 1
        this_x,this_y      = i[0],i[1]
        this_data_center   = data[this_x, this_y]
        this_freq_center   = freq[this_x, this_y]
        this_data_neighbor = data[max(0,this_x-bw):this_x+1+bw, max(0,this_y-bw):this_y+1+bw]
        this_freq_neighbor = freq[max(0,this_x-bw):this_x+1+bw, max(0,this_y-bw):this_y+1+bw]
        this_vel_center    = (freq_cn10h - this_freq_center) / freq_cn10h * 299792.458 # km/s
        chanwidth          = abs(this_vel_center[1] - this_vel_center[0])
        chanwidth_GHz      = abs(this_freq_center[1] - this_freq_center[0])

        p0 = [
        np.max(this_data_center),
        0.2,
        freq_cn10l,
        freq_cn10h,
        np.max([len(this_data_center[this_data_center>=rms*snr])/2.35,2.0]) * chanwidth_GHz / 4.0,
        ]

        if np.max(this_data_center)<rms*snr:
            # add pixel
            tau[this_x,this_y]      = 0
            mom0[this_x,this_y]     = 0
            mom1[this_x,this_y]     = 0
            mom2[this_x,this_y]     = 0
            mom0_res[this_x,this_y] = 0
            ratio[this_x,this_y]    = 0
        else:
            # fitting
            popt,pcov = curve_fit(
                _f_cn10,
                this_freq_center,
                np.mean(this_data_neighbor,axis=(0,1)),
                #sigma=1./np.mean(this_data_neighbor,axis=(0,1)),
                p0 = p0,
                maxfev = 1000000,
                )
            area1,area2 = _f_cn10_areas(this_freq_center, *popt)

            # keep worst residual pixel
            residuals      = np.mean(this_data_neighbor,axis=(0,1)) - _f_cn10(this_freq_center, *popt)
            ss_res         = np.sum(residuals**2)
            ss_tot         = np.sum((this_data_center-np.mean(this_data_center))**2)
            this_r_squared = 1 - (ss_res / ss_tot)
            if ss_res<r_squared_value:
                r_squared_value = ss_res
                r_squared_index = [this_x,this_y]

            # calc values
            this_model    = _f_cn10(this_freq_center, *popt)
            this_tau      = np.min([np.max([0.01, popt[1]]),10])
            this_mom0     = np.sum(this_model) * chanwidth
            this_mom1     = (freq_cn10h - popt[3]) / freq_cn10h * 299792.458 # km/s
            this_mom2     = popt[4] * 299792.458 / freq_cn10h # km/s
            this_mom0_res = np.sum(residuals) * chanwidth
            this_ratio    = area1/area2

            # calc errors
            perr          = np.sqrt(np.diag(pcov))
            this_tau_snr  = this_tau / perr[1]

            print("tau = " + str(np.round(popt[1],2)).rjust(6) + " (SNR = " + str(np.round(popt[1]/perr[1],1)).rjust(4) + "), pos = " + str(this_x).rjust(3) + ", " + str(this_y).rjust(3) + ", ratio = " + str(np.round(area1/area2,2)))

            if abs(this_mom2)>300 or this_mom2<0:
                this_tau      = 0.0
                this_mom0     = 0.0
                this_mom1     = 0.0
                this_mom2     = 0.0
                this_mom0_res = 0.0
                this_ratio    = 0.0

            #if this_tau_snr<=1:
            #    this_tau      = 0.0

            # add pixel
            tau[this_x,this_y]      = this_tau
            mom0[this_x,this_y]     = this_mom0
            mom1[this_x,this_y]     = this_mom1
            mom2[this_x,this_y]     = this_mom2
            mom0_res[this_x,this_y] = this_mom0_res
            ratio[this_x,this_y]    = this_ratio

    # fits
    fits_creation(tau.T,"tau.fits")
    fits_creation(ratio.T,"ratio.fits")
    fits_creation(mom0.T,"mom0.fits")
    fits_creation(mom1.T,"mom1.fits")
    fits_creation(mom2.T,"mom2.fits")
    fits_creation(mom0_res.T,"mom0_residual.fits")

    # plot worst spectrum r_squared_index
    plot_spectrum(data,freq,[17,57],"plot_worst_spectrum.png")

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
    hdu = fits.PrimaryHDU(input_map)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_map)

#################
# plot_spectrum #
#################

def plot_spectrum(
    data,
    freq,
    index_worst_fit,
    outpng="plot_worst_spectrum.png",
    ):
    """
    """

    z          = 0.00379
    freq_cn10l = 113.123337/(1+z)
    freq_cn10h = 113.490970/(1+z)

    this_data_worst = data[index_worst_fit[0], index_worst_fit[1]]
    this_freq_worst = freq[index_worst_fit[0], index_worst_fit[1]]

    popt,_ = curve_fit(
        _f_cn10,
        this_freq_worst,
        this_data_worst,
        p0 = [1,0.1,freq_cn10l,freq_cn10h,0.05],
        maxfev = 10000,
        )
    this_residual = this_data_worst - _f_cn10(this_freq_worst, *popt)

    fig = plt.figure(figsize=(10,10))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20

    gs = gridspec.GridSpec(nrows=30, ncols=30)
    ax = plt.subplot(gs[0:30,0:30])
    ax.grid(axis="both", ls="--")

    ax.plot(
        this_freq_worst,
        this_data_worst,
        linewidth=2,
        color="tomato",
        )
    ax.plot(
        this_freq_worst,
        this_residual,
        linewidth=2,
        color="deepskyblue",
        )
    ax.plot(
        this_freq_worst,
        _f_cn10(this_freq_worst, *popt),
        linewidth=2,
        color="black",
        )
    ax.plot([np.min(this_freq_worst),np.max(this_freq_worst)],[0,0],"--",c="black")

    os.system("rm -rf " + outpng)
    plt.savefig(outpng, dpi=100)

############
# plot_mom #
############

def plot_mom(
    ra,
    dec,
    mom,
    lim,
    output,
    plot_cbar=True,
    label="K km s$^{-1}$",
    logmom=False,
    size=90,
    ):
    """
    """

    this_ra  = ra[mom!=0]
    this_dec = dec[mom!=0]
    this_mom = mom[mom!=0]
    if logmom==True:
        color = np.log10(this_mom).flatten()
    else:
        color = this_mom.flatten()

    fig = plt.figure(figsize=(13,10))
    plt.rcParams["font.size"] = 16
    plt.rcParams["legend.fontsize"] = 20
    gs = gridspec.GridSpec(nrows=10, ncols=10)
    ax = plt.subplot(gs[0:10,0:10])
    ax.grid(axis="both", ls="--")

    # plot
    im = ax.scatter(
        -1*this_ra.flatten(),
        this_dec.flatten(),
        c=color,
        cmap="rainbow",
        s=size, marker="s",
        linewidths=0,
        )

    # cbar
    cbar = plt.colorbar(im)
    #if plot_cbar==True:
    #    cax  = fig.add_axes([0.19, 0.12, 0.025, 0.35])
    #    fig.colorbar(im, cax=cax).set_label(label)

    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])

    os.system("rm -rf " + output)
    plt.savefig(output, dpi=100)

#############
# plot_hist #
#############

def plot_hist(data,output,snr_for_fit=1.0):
    """
    plot voxel distribution in the input data. Zero will be ignored.
    output units will be mK.
    """

    # prepare
    data = data.flatten()
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    data = data[data!=0] * 1000

    if output.endswith("image.png"):
        title = ".image histogram"
    elif output.endswith("residual.png"):
        title = ".residual histogram"
    else:
        title = "None"

    # get voxel histogram
    bins   = int( np.ceil( np.log2(len(data))+1 ) * 10 )
    p84    = np.percentile(data, 16) * -1
    hrange = [-5*p84, 5*p84]
    hist   = np.histogram(data, bins=bins, range=hrange)

    x,y   = hist[1][:-1], hist[0]/float(np.sum(hist[0])) * 1000
    histx = x[x<p84*snr_for_fit]
    histy = y[x<p84*snr_for_fit]

    # fit
    x_bestfit = np.linspace(hrange[0], hrange[1], bins)
    popt,_ = curve_fit(_f_gauss, histx, histy, p0=[np.max(histy),p84], maxfev=10000)

    # best fit
    peak      = popt[0]
    rms_mK    = np.round(popt[1],3)
    y_bestfit = _f_gauss(x_bestfit, peak, rms_mK)

    # plot
    ymax  = np.max(y) * 1.1
    cpos  = "tomato"
    cneg  = "deepskyblue"
    cfit  = "black"
    snr_a = 1.0
    snr_b = 3.0

    tpos  = "positive side"
    tneg  = "(flipped) negative side"
    tfit  = "best fit Gaussian"
    ha    = "left"
    w     = "bold"
    tsnr_a = "1$\sigma$ = \n"+str(np.round(rms_mK*snr_a,3))+" mJy"
    tsnr_b = "3$\sigma$ = \n"+str(np.round(rms_mK*snr_b,3))+" mJy"

    fig = plt.figure(figsize=(10,10))
    plt.rcParams["font.size"] = 22
    plt.rcParams["legend.fontsize"] = 20

    gs = gridspec.GridSpec(nrows=30, ncols=30)
    ax = plt.subplot(gs[0:30,0:30])
    ax.grid(axis="both", ls="--")

    ax.step(x, y, color=cpos, lw=4, where="mid")
    ax.bar(x, y, lw=0, color=cpos, alpha=0.2, width=x[1]-x[0], align="center")
    ax.step(-x, y, color=cneg, lw=4, where="mid")
    ax.bar(-x, y, lw=0, color=cneg, alpha=0.2, width=x[1]-x[0], align="center")
    ax.plot(x_bestfit, y_bestfit, "k-", lw=3)
    ax.plot([snr_a*rms_mK,snr_a*rms_mK], [0,ymax], "k--", lw=1)
    ax.plot([snr_b*rms_mK,snr_b*rms_mK], [0,ymax], "k--", lw=1)

    tf = ax.transAxes
    ax.text(0.45,0.93,tpos,color=cpos,transform=tf,horizontalalignment=ha,weight=w)
    ax.text(0.45,0.88,tneg,color=cneg,transform=tf,horizontalalignment=ha,weight=w)
    ax.text(0.45,0.83,tfit,color=cfit,transform=tf,horizontalalignment=ha,weight=w)
    ax.text(rms_mK*snr_a,ymax*0.8,tsnr_a,rotation=90)
    ax.text(rms_mK*snr_b,ymax*0.5,tsnr_b,rotation=90)

    ax.set_title(title)
    ax.set_xlabel("Pixel value (mJy)")
    ax.set_ylabel("Pixel count density * 10$^{3}$")
    ax.set_xlim([0,5*p84])
    ax.set_ylim([0,ymax])

    plt.savefig(output, dpi=100)

    return rms_mK / 1000.

###########$
# _f_gauss #
###########$

def _f_gauss(x, a, c):

    return a*np.exp(-(x)**2/(2*c**2))

###########
# _f_cn10 #
###########

def _f_cn10(x,a,tau,b1,b2,c):

    z = 0.00379

    # k factor
    k_l1 =  1.23 / 33.33
    k_l2 =  9.88 / 33.33
    k_l3 =  9.88 / 33.33
    k_l4 = 12.35 / 33.33

    k_h1 = 12.35 / 33.33
    k_h2 = 33.33 / 33.33
    k_h3 =  9.88 / 33.33
    k_h4 =  9.88 / 33.33
    k_h5 =  1.23 / 33.33

    # freq
    f_l1 = 113.123337 / (1+z)
    f_h1 = 113.488126 / (1+z)

    b_l1 = 113.123337 / (1+z) - f_l1
    b_l2 = 113.144122 / (1+z) - f_l1
    b_l3 = 113.170502 / (1+z) - f_l1
    b_l4 = 113.191287 / (1+z) - f_l1

    b_h1 = 113.488126 / (1+z) - f_h1
    b_h2 = 113.490943 / (1+z) - f_h1
    b_h3 = 113.499629 / (1+z) - f_h1
    b_h4 = 113.508911 / (1+z) - f_h1
    b_h5 = 113.520414 / (1+z) - f_h1

    gauss_cn10_l1 = a * np.exp( -(x+b_l1-b1)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_l1) )
    gauss_cn10_l2 = a * np.exp( -(x+b_l2-b1)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_l2) )
    gauss_cn10_l3 = a * np.exp( -(x+b_l3-b1)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_l3) )
    gauss_cn10_l4 = a * np.exp( -(x+b_l4-b1)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_l4) )

    gauss_cn10_h1 = a * np.exp( -(x+b_h1-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h1) )
    gauss_cn10_h2 = a * np.exp( -(x+b_h2-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h2) )
    gauss_cn10_h3 = a * np.exp( -(x+b_h3-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h3) )
    gauss_cn10_h4 = a * np.exp( -(x+b_h4-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h4) )
    gauss_cn10_h5 = a * np.exp( -(x+b_h5-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h5) )

    func = gauss_cn10_l1 + gauss_cn10_l2 + gauss_cn10_l3 + gauss_cn10_l4 + gauss_cn10_h1 + gauss_cn10_h2 + gauss_cn10_h3 + gauss_cn10_h4 + gauss_cn10_h5

    return func

def _f_cn10_areas(x,a,tau,b1,b2,c):

    z = 0.00379

    # k factor
    k_l1 =  1.23 / 33.33
    k_l2 =  9.88 / 33.33
    k_l3 =  9.88 / 33.33
    k_l4 = 12.35 / 33.33

    k_h1 = 12.35 / 33.33
    k_h2 = 33.33 / 33.33
    k_h3 =  9.88 / 33.33
    k_h4 =  9.88 / 33.33
    k_h5 =  1.23 / 33.33

    # freq
    f_l1 = 113.123337 / (1+z)
    f_h1 = 113.488126 / (1+z)

    b_l1 = 113.123337 / (1+z) - f_l1
    b_l2 = 113.144122 / (1+z) - f_l1
    b_l3 = 113.170502 / (1+z) - f_l1
    b_l4 = 113.191287 / (1+z) - f_l1

    b_h1 = 113.488126 / (1+z) - f_h1
    b_h2 = 113.490943 / (1+z) - f_h1
    b_h3 = 113.499629 / (1+z) - f_h1
    b_h4 = 113.508911 / (1+z) - f_h1
    b_h5 = 113.520414 / (1+z) - f_h1

    gauss_cn10_l1 = a * np.exp( -(x+b_l1-b1)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_l1) )
    gauss_cn10_l2 = a * np.exp( -(x+b_l2-b1)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_l2) )
    gauss_cn10_l3 = a * np.exp( -(x+b_l3-b1)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_l3) )
    gauss_cn10_l4 = a * np.exp( -(x+b_l4-b1)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_l4) )

    gauss_cn10_h1 = a * np.exp( -(x+b_h1-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h1) )
    gauss_cn10_h2 = a * np.exp( -(x+b_h2-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h2) )
    gauss_cn10_h3 = a * np.exp( -(x+b_h3-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h3) )
    gauss_cn10_h4 = a * np.exp( -(x+b_h4-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h4) )
    gauss_cn10_h5 = a * np.exp( -(x+b_h5-b2)**2 / (2*c**2) ) * ( 1-np.exp(-tau*k_h5) )

    area1 = np.sum(gauss_cn10_l1 + gauss_cn10_l2 + gauss_cn10_l3 + gauss_cn10_l4)
    area2 = np.sum(gauss_cn10_h1 + gauss_cn10_h2 + gauss_cn10_h3 + gauss_cn10_h4 + gauss_cn10_h5)

    return area1, area2

"""
def _f_cn10(x,a,tau,b1,b2,c1,c2):

    k = 0.5

    gauss_cn10l = a * np.exp( -(x-b1)**2 / (2*c1**2) ) * (1-np.exp(tau*k))/(1-np.exp(tau)) * c1/c2
    gauss_cn10h = a * np.exp( -(x-b2)**2 / (2*c2**2) )

    func = gauss_cn10l + gauss_cn10h

    return func
"""

#############
# _get_grid #
#############

def _get_grid(imagename):

    print("# _get_grid " + imagename.split("/")[-1])

    head  = imhead(imagename,mode="list")
    shape = head["shape"][0:2]
    pix   = abs(head["cdelt1"]) * 3600 * 180/np.pi
    
    return shape, pix
