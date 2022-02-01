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

    max_low  = np.nanmax(data_low)
    max_high = np.nanmax(data_high)

    # fitting spectra
    bw  = smooth
    lim = np.max([np.max(abs(ra)), np.max(abs(dec))])
    x   = range(np.shape(data_low)[0])
    y   = range(np.shape(data_low)[1])
    xy  = itertools.product(x, y)

    map_Trot  = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_Nmol  = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
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
        guess_b = (restfreq_low - this_freq_low[np.nanargmax(this_data_low)]) / restfreq_low * 299792.458
        p0 = [
        15.0,#np.max(this_data)/2.0,
        2*10**16,#np.max(this_data),
        guess_b,
        40.,
        ]

        # fit
        if np.max(this_data_low/this_err_low)>=snr and np.max(this_data_high/this_err_high)>=snr:
            # fitting
            #this_f_two = lambda x, a1, a2, b, c: _f_two(x, a1, a2, b, c, restfreq_low, restfreq_high)
            this_f_two = lambda x, Trot, Nmol, b, c: _f_two_rot_13co21_13co10(x, Trot, Nmol, b, c, restfreq_low, restfreq_high)
            popt,pcov = curve_fit(this_f_two,this_freq,this_data,sigma=this_err,p0=p0,maxfev=100000)
            print(popt)

            #if popt[1]/popt[0]>0 and popt[1]/popt[0]<=ratio_max and popt[2]!=guess_b and popt[3]!=40 and popt[0]<max_low*2 and popt[1]<max_low*2:
            if popt[0]>0 and popt[0]<=100. and popt[2]!=guess_b and popt[3]!=40:
                # add pixel
                map_Trot[this_x,this_y]   = popt[0]
                map_Nmol[this_x,this_y]   = np.log10(popt[1])
                #mom0_low[this_x,this_y]  = popt[0] * abs(popt[3]) * np.sqrt(2*np.pi)
                #mom0_high[this_x,this_y] = popt[1] * abs(popt[3]) * np.sqrt(2*np.pi)
                mom1[this_x,this_y]      = popt[2]
                mom2[this_x,this_y]      = abs(popt[3])
            else:
                map_Trot[this_x,this_y]   = np.nan
                map_Nmol[this_x,this_y]   = np.nan
                # add pixel
                #mom0_low[this_x,this_y]  = np.nan
                #mom0_high[this_x,this_y] = np.nan
                mom1[this_x,this_y]      = np.nan
                mom2[this_x,this_y]      = np.nan
        else:
            # add pixel
            map_Trot[this_x,this_y]   = np.nan
            map_Nmol[this_x,this_y]   = np.nan
            #mom0_low[this_x,this_y]  = np.nan
            #mom0_high[this_x,this_y] = np.nan
            mom1[this_x,this_y]      = np.nan
            mom2[this_x,this_y]      = np.nan

    # fits
    fits_creation(map_Trot.T,"Trot.fits")
    fits_creation(map_Nmol.T,"Nmol.fits")
    #fits_creation(mom0_low.T,"mom0_low.fits")
    #fits_creation(mom0_high.T,"mom0_high.fits")
    #fits_creation(mom0_high.T/mom0_low.T,"ratio.fits")
    fits_creation(mom1.T,"mom1.fits")
    fits_creation(mom2.T,"mom2.fits")

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

############################
# _f_two_rot_13co21_13co10 #
############################

def _f_two_rot_13co21_13co10(x, Trot, Nmol, b, c, freq_l, freq_u):
    """
    equation 1 of Nakajima et al. 2018

    input parameters:
    freq_l in GHz
    freq_u in GHz

    free parameters:
    Nmol
    Trot
    """

    # constants
    Z = 3*np.exp(-5.28880/Trot) \
      + 5*np.exp(-15.86618/Trot) \
      + 7*np.exp(-31.73179/Trot) \
      + 9*np.exp(-52.88517/Trot) \
      + 11*np.exp(-79.32525/Trot) \
      + 13*np.exp(-111.05126/Trot) \
      + 15*np.exp(-148.06215/Trot) \
      + 17*np.exp(-190.35628/Trot) \
      + 19*np.exp(-237.93232/Trot) \
      + 21*np.exp(-290.78848/Trot) \
      + 21*np.exp(-348.92271/Trot)

    k = 1.38 * 10**-16 # erg/K
    h = 6.626 * 10**-27 # erg.s
    c = 299792458.0 # m/s

    Aul_l = 10**-7.198
    gu_l  = 3
    gl_l  = 1
    gk_l  = 1
    Eu_l  = 5.28880

    Aul_u = 10**-6.216
    gu_u  = 5
    gl_u  = 1
    gk_u  = 1
    Eu_u  = 15.86618

    # lower-J equation
    A_l = np.log10( Nmol/Z )
    B_l = np.log10( (8*np.pi*k*(freq_l*10**9)**2)/(h*c**3*Aul_l*gu_l*gl_l*gk_l) )
    C_l = Eu_l / k * np.log10(np.e) / Trot
    W_l = 10**(A_l - B_l - C_l)

    A_u = np.log10( Nmol/Z )
    B_u = np.log10( (8*np.pi*k*(freq_u*10**9)**2)/(h*c**3*Aul_u*gu_u*gl_u*gk_u) )
    C_u = Eu_u / k * np.log10(np.e) / Trot
    W_u = 10**(A_u - B_u - C_u)

    # fitting function
    offset1 = b /299792.458 * freq_l # km/s
    offset2 = b /299792.458 * freq_u # km/s

    width1 = c /299792.458 * freq_l # km/s
    width2 = c /299792.458 * freq_u # km/s

    a1 = W_l / (width1 * np.sqrt(2*np.pi))
    a2 = W_u / (width2 * np.sqrt(2*np.pi))

    func = \
        a1 * np.exp( -(x-freq_l+offset1)**2/(2*width1**2) ) + \
        a2 * np.exp( -(x-freq_u+offset2)**2/(2*width2**2) )

    return func

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

#######
# end #
#######