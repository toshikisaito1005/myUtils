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

execfile(os.environ["HOME"] + "/myUtils/stuff_casa.py")

#

###########
# hf_cn10 #
###########

def hf_cn10(
    cubeimage,
    snr=4.0,
    ra_cnt=40.669625, # 1068 agn, deg
    dec_cnt=-0.01331667, # 1068 agn, deg
    outpng_hist="plot_voxel_hist.png",
    ):
    """
    Reference:
    CN10 rest frequencies: Skatrud et al. 1983, Journal of Mol. Spec. 99, 35
    neiboring 8 pixels: https://qiita.com/yuji96/items/bfae04a043d35260ffb1
    """

    taskname = sys._getframe().f_code.co_name
    mytask.check_first(cubeimage, taskname)

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

    mom0            = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    mom1            = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    mom2            = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    mom0_residual   = np.zeros((np.shape(data)[0],np.shape(data)[1]))
    r_squared_index = [0,0]
    r_squared_value = 0.0#1.0
    for i in xy:
        this_x,this_y      = i[0],i[1]
        this_data_center   = data[this_x, this_y]
        this_freq_center   = freq[this_x, this_y]
        this_data_neighbor = data[max(0,this_x-1):this_x+2, max(0,this_y-1):this_y+2]
        this_freq_neighbor = freq[max(0,this_x-1):this_x+2, max(0,this_y-1):this_y+2]
        this_vel_center    = (freq_cn10h - this_freq_center) / freq_cn10h * 299792.458 # km/s
        chanwidth          = abs(this_vel_center[1] - this_vel_center[0])

        p0 = [
        np.max(this_data_center),
        this_freq_center[np.argmax(this_data_center)],
        len(this_data_center[this_data_center>=rms*snr]) / 2.35 * abs(this_freq_center[1]-this_freq_center[0]),
        ]

        if np.max(this_data_center)<rms*snr:
            # add pixel
            mom0[this_x,this_y]          = 0
            mom1[this_x,this_y]          = 0
            mom2[this_x,this_y]          = 0
            mom0_residual[this_x,this_y] = 0
        else:
            # fitting
            popt,_ = curve_fit(
                _f_cn10,
                this_freq_center,
                this_data_center, #np.mean(this_data_neighbor,axis=(0,1))
                p0 = p0,
                maxfev = 10000,
                )

            # keep worst residual pixel
            residuals      = this_data_center - _f_cn10(this_freq_center, *popt)
            ss_res         = np.sum(residuals**2)
            ss_tot         = np.sum((this_data_center-np.mean(this_data_center))**2)
            this_r_squared = 1 - (ss_res / ss_tot)
            if ss_res>r_squared_value:
                r_squared_value = ss_res
                r_squared_index = [this_x,this_y]

            # add pixel
            this_model = _f_cn10(this_freq_center, *popt)
            mom0[this_x,this_y] = np.sum(this_model)*chanwidth
            mom1[this_x,this_y] = popt[1]
            mom2[this_x,this_y] = popt[2]
            mom0_residual[this_x,this_y] = np.sum(residuals)*chanwidth

    # plot moments
    plot_mom(ra,dec,mom0,lim,"mom0.png",logmom=True)
    plot_mom(ra,dec,mom1,lim,"mom1.png")
    plot_mom(ra,dec,mom2,lim,"mom2.png")
    plot_mom(ra,dec,mom0_residual,lim,"mom0_residual.png")

    # plot worst spectrum
    plot_spectrum(data,freq,r_squared_index,"plot_worst_spectrum.png")

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

    this_data_worst = data[index_worst_fit[0], index_worst_fit[1]]
    this_freq_worst = freq[index_worst_fit[0], index_worst_fit[1]]

    popt,_ = curve_fit(
        _f_cn10,
        this_freq_worst,
        this_data_worst,
        p0 = [np.max(this_data_worst),this_freq_worst[np.argmax(this_data_worst)],0.05],
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
        s=23, marker="s",
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

def _f_cn10(x,a,b,c):

    return a*np.exp(-(x-b)**2/(2*c**2))

"""
def _f_cn10(x,eta,Tex,tau,vel,width):

    # prepare
    z       = 0.00379

    freq_l1 = 113.123337/(1+z)
    freq_l2 = 113.144122/(1+z)
    freq_l3 = 113.170502/(1+z)
    freq_l4 = 113.191287/(1+z)
    freq_h1 = 113.488126/(1+z)
    freq_h2 = 113.490943/(1+z)
    freq_h3 = 113.499629/(1+z)
    freq_h4 = 113.508911/(1+z)
    freq_h5 = 113.520414/(1+z)

    k_l1    =  1.23 / 33.33
    k_l2    =  9.88 / 33.33
    k_l3    =  9.88 / 33.33
    k_l4    = 12.35 / 33.33
    k_h1    = 12.35 / 33.33
    k_h2    = 33.33 / 33.33
    k_h3    =  9.88 / 33.33
    k_h4    =  9.88 / 33.33
    k_h5    =  1.23 / 33.33

    k_B     = 1.38064852e-16 # erg.K^-1
    h_p     = 6.6260755e-27 # erg.s
    Tbg     = 2.73 K

    # define a
    tpeak_l1 = eta * ( (h_p*freq_l1/k_B)/(np.exp(h_p*freq_l1/(k_B*Tex))-1) - (h_p*freq_l1/k_B)/(np.exp(h_p*freq_l1/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_l1) )
    tpeak_l2 = eta * ( (h_p*freq_l2/k_B)/(np.exp(h_p*freq_l2/(k_B*Tex))-1) - (h_p*freq_l2/k_B)/(np.exp(h_p*freq_l2/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_l2) )
    tpeak_l3 = eta * ( (h_p*freq_l3/k_B)/(np.exp(h_p*freq_l3/(k_B*Tex))-1) - (h_p*freq_l3/k_B)/(np.exp(h_p*freq_l3/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_l3) )
    tpeak_l4 = eta * ( (h_p*freq_l4/k_B)/(np.exp(h_p*freq_l4/(k_B*Tex))-1) - (h_p*freq_l4/k_B)/(np.exp(h_p*freq_l4/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_l4) )
    tpeak_h1 = eta * ( (h_p*freq_h1/k_B)/(np.exp(h_p*freq_h1/(k_B*Tex))-1) - (h_p*freq_h1/k_B)/(np.exp(h_p*freq_h1/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_h1) )
    tpeak_h2 = eta * ( (h_p*freq_h2/k_B)/(np.exp(h_p*freq_h2/(k_B*Tex))-1) - (h_p*freq_h2/k_B)/(np.exp(h_p*freq_h2/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_h2) )
    tpeak_h3 = eta * ( (h_p*freq_h3/k_B)/(np.exp(h_p*freq_h3/(k_B*Tex))-1) - (h_p*freq_h3/k_B)/(np.exp(h_p*freq_h3/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_h3) )
    tpeak_h4 = eta * ( (h_p*freq_h4/k_B)/(np.exp(h_p*freq_h4/(k_B*Tex))-1) - (h_p*freq_h4/k_B)/(np.exp(h_p*freq_h4/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_h4) )
    tpeak_h5 = eta * ( (h_p*freq_h5/k_B)/(np.exp(h_p*freq_h5/(k_B*Tex))-1) - (h_p*freq_h5/k_B)/(np.exp(h_p*freq_h5/(k_B*Tbg))-1) ) * ( 1-np.exp(-tau*k_h5) )

    # define b
    b_l1 = 113.123337/(1+z) - freq_l1
    b_l2 = 113.144122/(1+z) - freq_l1
    b_l3 = 113.170502/(1+z) - freq_l1
    b_l4 = 113.191287/(1+z) - freq_l1
    b_h1 = 113.488126/(1+z) - freq_l1
    b_h2 = 113.490943/(1+z) - freq_l1
    b_h3 = 113.499629/(1+z) - freq_l1
    b_h4 = 113.508911/(1+z) - freq_l1
    b_h5 = 113.520414/(1+z) - freq_l1

    func = \
    tpeak_l1 * np.exp( -(x+b_l1-vel)**2 / (2*width**2) ) + \
    tpeak_l2 * np.exp( -(x+b_l2-vel)**2 / (2*width**2) ) + \
    tpeak_l3 * np.exp( -(x+b_l3-vel)**2 / (2*width**2) ) + \
    tpeak_l4 * np.exp( -(x+b_l4-vel)**2 / (2*width**2) ) + \
    tpeak_m1 * np.exp( -(x+b_h1-vel)**2 / (2*width**2) ) + \
    tpeak_m2 * np.exp( -(x+b_h2-vel)**2 / (2*width**2) ) + \
    tpeak_m3 * np.exp( -(x+b_h3-vel)**2 / (2*width**2) ) + \
    tpeak_m4 * np.exp( -(x+b_h4-vel)**2 / (2*width**2) ) + \
    tpeak_m5 * np.exp( -(x+b_h5-vel)**2 / (2*width**2) )

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
