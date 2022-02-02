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

k = 1.38 * 10**-16 # erg/K
h = 6.626 * 10**-27 # erg.s
c = 299792458.0 # m/s

##########################
# rotation_13co21_13co10 #
##########################

def rotation_13co21_13co10(
    cubelow,
    cubehigh,
    ecubelow,
    ecubehigh,
    ra_cnt=40.669625, # deg
    dec_cnt=-0.01331667, # deg
    snr=5.0,
    ratio_max=2.0,
    restfreq_low=None,
    restfreq_high=None,
    Aul_low=10**-7.198,
    Aul_high=10**-6.216,
    gu_low=3,
    gu_high=5,
    Eu_low=5.28880,
    Eu_high=15.86618,
    p0_rotation=[8.0,16.5], # initial guess for rotation diagram, Trot and log10 Ncol
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

    ra       = ra_deg[:,:,0] * 3600
    dec      = dec_deg[:,:,0] * 3600
    max_low  = np.nanmax(data_low) * 1.5
    max_high = np.nanmax(data_high) * 1.5

    # fitting spectra
    xy  = itertools.product(range(np.shape(data_low)[0]), range(np.shape(data_low)[1]))

    array_nan      = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    array_nan[:,:] = np.nan

    map_Trot, map_eTrot           = array_nan, array_nan
    map_logN, map_elogN           = array_nan, array_nan
    map_mom0_low, map_emom0_low   = array_nan, array_nan
    map_mom0_high, map_emom0_high = array_nan, array_nan
    map_mom1, map_emom1           = array_nan, array_nan
    map_mom2, map_emom2           = array_nan, array_nan
    map_ratio, map_eratio         = array_nan, array_nan

    for i in xy:
        # get data of this sightline
        this_x,this_y  = i[0],i[1]

        this_freq_low  = freq_low[this_x, this_y]
        this_data_low  = data_low[this_x, this_y]
        this_err_low   = err_low[this_x, this_y]

        this_freq_high = freq_high[this_x, this_y]
        this_data_high = data_high[this_x, this_y]
        this_err_high  = err_high[this_x, this_y]

        max_snr_low    = np.max(this_data_low/this_err_low)
        max_snr_high   = np.max(this_data_high/this_err_high)

        # avareging neibors
        #this_data_low  = np.mean(data_low[max(0,this_x-bw):this_x+1+bw, max(0,this_y-bw):this_y+1+bw],axis=(0,1)) # data_low[this_x, this_y]
        #this_data_high = np.mean(data_high[max(0,this_x-bw):this_x+1+bw, max(0,this_y-bw):this_y+1+bw],axis=(0,1)) # data_high[this_x, this_y]

        # combine two data
        this_freq = np.r_[this_freq_low, this_freq_high]
        this_data = np.r_[this_data_low, this_data_high]
        this_err  = np.r_[this_err_low, this_err_high]

        # p0 guess
        guess_b = (restfreq_low - this_freq_low[np.nanargmax(this_data_low)]) / restfreq_low * 299792.458
        p0 = [np.max(this_data)/2.0, np.max(this_data), guess_b, 40.]

        # fit
        if max_snr_low>=snr and max_snr_high>=snr:
            # fitting
            this_f_two = lambda x, a1, a2, b, c: _f_two(x, a1, a2, b, c, restfreq_low, restfreq_high)
            popt,pcov  = curve_fit(
                this_f_two,
                this_freq,
                this_data,
                sigma  = this_err,
                p0     = p0,
                maxfev = 100000,
                )
            perr = np.sqrt(np.diag(pcov))

            mom0_13co10 = popt[0]
            mom0_13co21 = popt[1]
            popt_ratio  = mom0_13co21/mom0_13co10
            mom1        = popt[2]
            mom2        = abs(popt[3])

            err_mom0_13co10 = perr[0]
            err_mom0_13co21 = perr[1]
            err_mom2        = abs(perr[3])

            snr_13co10 = mom0_13co10/err_mom0_13co10
            snr_13co21 = mom0_13co21/err_mom0_13co21

            if popt_ratio>0 and popt_ratio<=ratio_max and mom1!=guess_b and mom2!=40 and mom0_13co10<max_low and mom0_13co21<max_high and snr_13co10>1 and snr_13co21>1:
                # rotation diagram fitting
                this_mom0_low    = mom0_13co10 * mom2 * np.sqrt(2*np.pi)
                this_mom0_high   = mom0_13co21 * mom2 * np.sqrt(2*np.pi)

                this_emom0_low   = np.sqrt(2*np.pi) * np.sqrt(mom0_13co10**2*err_mom2**2 + mom2**2*err_mom0_13co10**2)
                this_emom0_high  = np.sqrt(2*np.pi) * np.sqrt(mom0_13co21**2*err_mom2**2 + mom2**2*err_mom0_13co21**2)

                log10_Nugu_low   = np.log10(derive_Nu(this_mom0_low, restfreq_low, Aul_low) / gu_low)
                log10_Nugu_high  = np.log10(derive_Nu(this_mom0_high, restfreq_high, Aul_high) / gu_high)

                elog10_Nugu_low  = derive_Nu(this_emom0_low, restfreq_low, Aul_low) / abs(derive_Nu(this_mom0_low, restfreq_low, Aul_low)) / np.log(10)
                elog10_Nugu_high = derive_Nu(this_emom0_high, restfreq_high, Aul_high) / abs(derive_Nu(this_mom0_high, restfreq_high, Aul_high)) / np.log(10)

                x_data       = np.array([Eu_low, Eu_high])
                y_data       = np.array([log10_Nugu_low, log10_Nugu_high])
                y_err        = np.array([elog10_Nugu_low, elog10_Nugu_high])
                popt2, pcov2 = curve_fit(_f_linear,x_data,y_data,sigma=y_err,p0=p0_rotation,maxfev=100000)
                perr2        = np.sqrt(np.diag(pcov2))

                Trot  = popt2[0]
                eTrot = perr2[0]

                Z = derive_Z_13co(Trot)

                logNmol  = popt2[1] + np.log10(Z)
                elogNmol = perr2[1] + np.log10(Z)

                # add pixel
                print(this_mom0_low)
                map_mom0_low[this_x,this_y]   = this_mom0_low
                map_mom0_high[this_x,this_y]  = this_mom0_high
                map_mom1[this_x,this_y]       = popt[2]
                map_mom2[this_x,this_y]       = abs(popt[3])
                map_ratio[this_x,this_y]      = popt[1]/popt[0]
                map_emom0_low[this_x,this_y]  = this_emom0_low
                map_emom0_high[this_x,this_y] = this_emom0_high
                map_emom1[this_x,this_y]      = perr[2]
                map_emom2[this_x,this_y]      = abs(perr[3])
                map_eratio[this_x,this_y]     = popt[1]/popt[0] * np.sqrt(perr[0]**2/popt[0]**2 + perr[1]**2/popt[1]**2)

                if Trot>2.7:
                    map_Trot[this_x,this_y]   = Trot
                    map_logN[this_x,this_y]   = logNmol
                    map_eTrot[this_x,this_y]  = eTrot
                    map_elogN[this_x,this_y]  = elogNmol

    # fits
    map_mom0_low[np.isnan(map_mom0_low)] = 0
    print(map_mom0_low[map_mom0_low!=0])
    fits_creation(map_Trot.T,"Trot.fits",cubelow,"K")
    fits_creation(map_logN.T,"logN.fits",cubelow,"cm**-2 in log10")
    fits_creation(map_mom0_low.T,"mom0_low.fits",cubelow,"K.km/s")
    fits_creation(map_mom0_high.T,"mom0_high.fits",cubelow,"K.km/s")
    fits_creation(map_ratio.T,"ratio.fits",cubelow,"")
    fits_creation(map_mom1.T,"mom1.fits",cubelow,"km/s")
    fits_creation(map_mom2.T,"mom2.fits",cubelow,"km/s")

    # efits
    fits_creation(map_eTrot.T,"eTrot.fits",cubelow,"K")
    fits_creation(map_elogN.T,"elogN.fits",cubelow,"cm**-2 in log10")
    fits_creation(map_emom0_low.T,"emom0_low.fits",cubelow,"K.km/s")
    fits_creation(map_emom0_high.T,"emom0_high.fits",cubelow,"K.km/s")
    fits_creation(map_eratio.T,"eratio.fits",cubelow,"")
    fits_creation(map_emom1.T,"emom1.fits",cubelow,"km/s")
    fits_creation(map_emom2.T,"emom2.fits",cubelow,"km/s")

#############
# derive_Nu #
#############

def derive_Nu(
    mom0, # K.km/s => the *10**3 term
    freq, # GHz    => the *10**9 term
    Aul,  # s^-1
    ):
    """
    return Nu (not in m^-2) in cm^-2 => the /10**4 term
    """

    Nu_m2  = (8*np.pi*k*(freq*10**9)**2) / (h*c**3*Aul) * mom0 * 10**3
    Nu_cm2 = Nu_m2 / 10**4

    return Nu_cm2

#################
# derive_Z_13co #
#################

def derive_Z_13co(
    Trot,
    ):
    """
    """

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

    return Z

#################
# fits_creation #
#################

def fits_creation(
    input_array,
    output_map,
    coords_template,
    bunit="K",
    ):
    """
    Reference:
    https://stackoverflow.com/questions/45744394/write-a-new-fits-file-after-modification-in-pixel-values
    """
    os.system("rm -rf " + output_map)
    os.system("cp " + coords_template + " " + output_map)

    obj = pyfits.open(output_map)
    obj[0].data = input_array
    #obj[0].header.append(("BUNIT", bunit))
    obj.writeto(output_map, clobber=True)

    #hdu = pyfits.PrimaryHDU(input_array)
    #hdul = pyfits.HDUList([hdu])
    #hdul.writeto(output_map)

#####################
# fitting functions #
#####################

def _f_linear(x, Trot, b):
    func = b - x * ( np.log10(np.e)/Trot )

    return func

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