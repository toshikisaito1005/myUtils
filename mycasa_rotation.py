"""
Standalone routines that are used for rotation diagram using CASA.

contents:
    rotation_13co21_13co10

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
    p0_rotation=[np.log10(np.e)/8.0,16.5], # initial guess for rotation diagram, Trot and log10 Ncol
    ):
    """
    References:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    """

    # preamble
    taskname = sys._getframe().f_code.co_name
    mytask.check_first(cubelow, taskname)

    # template
    template = "template.fits"
    os.system("rm -rf template.fits " + template+".image")
    imsubimage(cubelow,template+".image",chans="1")
    exportfits(template+".image",template)
    os.system("rm -rf " + template+".image")

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
    x   = range(np.shape(data_low)[0])
    y   = range(np.shape(data_low)[1])
    xy  = itertools.product(x, y)

    map_Trot            = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_logN            = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_mom0_low        = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_mom0_high       = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_mom1            = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_mom2            = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_ratio           = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))

    map_eTrot           = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_elogN           = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_emom0_low       = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_emom0_high      = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_emom1           = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_emom2           = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))
    map_eratio          = np.zeros((np.shape(data_low)[0],np.shape(data_low)[1]))

    map_Trot[:,:]       = np.nan
    map_logN[:,:]       = np.nan
    map_mom0_low[:,:]   = np.nan
    map_mom0_high[:,:]  = np.nan
    map_mom1[:,:]       = np.nan
    map_mom2[:,:]       = np.nan
    map_ratio[:,:]      = np.nan

    map_eTrot[:,:]      = np.nan
    map_elogN[:,:]      = np.nan
    map_emom0_low[:,:]  = np.nan
    map_emom0_high[:,:] = np.nan
    map_emom1[:,:]      = np.nan
    map_emom2[:,:]      = np.nan
    map_eratio[:,:]     = np.nan

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

        # fit when both 2-1 and 1-0 detected
        if max_snr_low>=snr and max_snr_high>=snr:
            # fitting
            this_f_two = lambda x, a1, a2, b, c: _f_two(x, a1, a2, b, c, restfreq_low, restfreq_high)
            popt,pcov  = curve_fit(
                this_f_two,
                this_freq,
                this_data,
                sigma          = this_err,
                p0             = p0,
                maxfev         = 100000,
                absolute_sigma = True,
                )
            perr = np.sqrt(np.diag(pcov))

            p0 = popt[0] # 1-0
            p1 = popt[1] # 2-1
            pr = p1/p0   # 2-1/1-0
            p2 = popt[2]
            p3 = abs(popt[3])

            e0 = perr[0]
            e1 = perr[1]
            e2 = perr[2]
            e3 = abs(perr[3])

            if p0>0 and p0<max_low and p1>0 and p1<max_high and pr>0 and pr<=ratio_max and p2!=guess_b and p3!=40 and p0/e0>snr and p1/e1>snr:
                # derive parameters
                this_mom0_low   = p0 * p3 * np.sqrt(2*np.pi)
                this_mom0_high  = p1 * p3 * np.sqrt(2*np.pi)
                this_mom1       = p2
                this_mom2       = p3
                this_ratio      = this_mom0_high / this_mom0_low

                # writing them
                map_mom0_low[this_x,this_y]   = this_mom0_low
                map_mom0_high[this_x,this_y]  = this_mom0_high
                map_mom1[this_x,this_y]       = this_mom1
                map_mom2[this_x,this_y]       = this_mom2
                map_ratio[this_x,this_y]      = this_ratio

                # derive error parameters
                this_emom0_low  = np.sqrt(2*np.pi) * np.sqrt(p0**2*e3**2 + p3**2*e0**2)
                this_emom0_high = np.sqrt(2*np.pi) * np.sqrt(p1**2*e3**2 + p3**2*e1**2)
                this_emom1      = e2
                this_emom2      = e3
                this_eratio     = p1/p0 * np.sqrt(e0**2/p0**2 + e1**2/p1**2)

                # writing them
                map_emom0_low[this_x,this_y]  = this_emom0_low
                map_emom0_high[this_x,this_y] = this_emom0_high
                map_emom1[this_x,this_y]      = this_emom1
                map_emom2[this_x,this_y]      = this_emom2
                map_eratio[this_x,this_y]     = this_eratio

                # rotation diagram fitting
                log10_Nugu_low   = np.log10(derive_Nu(this_mom0_low, restfreq_low, Aul_low) / gu_low)
                log10_Nugu_high  = np.log10(derive_Nu(this_mom0_high, restfreq_high, Aul_high) / gu_high)

                elog10_Nugu_low  = derive_Nu(this_emom0_low, restfreq_low, Aul_low) / abs(derive_Nu(this_mom0_low, restfreq_low, Aul_low))
                elog10_Nugu_high = derive_Nu(this_emom0_high, restfreq_high, Aul_high) / abs(derive_Nu(this_mom0_high, restfreq_high, Aul_high))

                x_data       = np.array([Eu_low, Eu_high])
                y_data       = np.array([log10_Nugu_low, log10_Nugu_high])
                y_err        = np.array([elog10_Nugu_low, elog10_Nugu_high])
                popt2, pcov2 = curve_fit(_f_linear,x_data,y_data,sigma=y_err,p0=p0_rotation,maxfev=100000,absolute_sigma=True)
                perr2        = np.sqrt(np.diag(pcov2))

                Trot  = np.log10(np.e) / popt2[0]
                eTrot = np.log10(np.e) / popt2[0]**2 * perr2[0]

                Z = derive_Z_13co(Trot)

                logNmol  = popt2[1] + np.log10(Z)
                elogNmol = perr2[1] + np.log10(Z)

                if Trot>eTrot*3.0:
                    # add pixel
                    map_Trot[this_x,this_y]   = Trot
                    map_logN[this_x,this_y]   = logNmol
                    map_eTrot[this_x,this_y]  = eTrot
                    map_elogN[this_x,this_y]  = elogNmol

        # fit when only 1-0 detected
        elif max_snr_low>=snr and max_snr_high<snr:
            # fitting
            this_f_two = lambda x, a1, a2, b, c: _f_one(x, a1, b, c, restfreq_low)
            popt,pcov  = curve_fit(
                this_f_two,
                this_freq_low,
                this_data_low,
                sigma          = this_err_low,
                p0             = p0,
                maxfev         = 100000,
                absolute_sigma = True,
                )
            perr = np.sqrt(np.diag(pcov))

            rms_high = np.sqrt(np.square(this_data_high).mean())

            p0 = popt[0] # 1-0
            p2 = popt[1]
            p3 = abs(popt[2])

            p1 = rms_high * snr # 2-1 tpeak upper limit
            pr = p1/p0   # 2-1/1-0

            e0 = perr[0]
            e2 = perr[1]
            e3 = abs(perr[2])
            print(p0>0,p0<max_low,pr>0,pr<=ratio_max,p2!=guess_b,p3!=40,p0/e0>snr)

            if p0>0 and p0<max_low and pr>0 and pr<=ratio_max and p2!=guess_b and p3!=40 and p0/e0>snr:
                print("# fit only 1-0")
                # derive parameters
                this_mom0_low   = p0 * p3 * np.sqrt(2*np.pi)
                this_mom0_high  = p1 * p3 * np.sqrt(2*np.pi) # upper limit
                this_mom1       = p2
                this_mom2       = p3
                this_ratio      = this_mom0_high / this_mom0_low # upper limit

                # writing them
                map_mom0_low[this_x,this_y]   = this_mom0_low
                map_mom0_high[this_x,this_y]  = this_mom0_high
                map_mom1[this_x,this_y]       = this_mom1
                map_mom2[this_x,this_y]       = this_mom2
                map_ratio[this_x,this_y]      = this_ratio

                # derive error parameters
                this_emom0_low  = np.sqrt(2*np.pi) * np.sqrt(p0**2*e3**2 + p3**2*e0**2)
                this_emom1      = e2
                this_emom2      = e3

                # writing them
                map_emom0_low[this_x,this_y]  = this_emom0_low
                map_emom1[this_x,this_y]      = this_emom1
                map_emom2[this_x,this_y]      = this_emom2

                # rotation diagram fitting
                log10_Nugu_low   = np.log10(derive_Nu(this_mom0_low, restfreq_low, Aul_low) / gu_low)
                log10_Nugu_high  = np.log10(derive_Nu(this_mom0_high, restfreq_high, Aul_high) / gu_high)

                x_data       = np.array([Eu_low, Eu_high])
                y_data       = np.array([log10_Nugu_low, log10_Nugu_high])
                popt2, pcov2 = curve_fit(_f_linear,x_data,y_data,p0=p0_rotation,maxfev=100000)

                Trot  = np.log10(np.e) / popt2[0]

                Z = derive_Z_13co(Trot)

                logNmol  = popt2[1] + np.log10(Z)

                # add pixel
                map_Trot[this_x,this_y]   = Trot
                map_logN[this_x,this_y]   = logNmol

    # low-J mom0 to fits
    fits_creation(map_mom0_low.T,"mom0_low.fits",template,"K.km/s")
    fits_creation(map_emom0_low.T,"emom0_low.fits",cubelow,"K.km/s")

    # high-J mom0 to fits
    fits_creation(map_mom0_high.T,"mom0_high.fits",template,"K.km/s")
    fits_creation(map_emom0_high.T,"emom0_high.fits",cubelow,"K.km/s")

    # mom1 to fits
    fits_creation(map_mom1.T,"mom1.fits",cubelow,"km/s")
    fits_creation(map_emom1.T,"emom1.fits",cubelow,"km/s")

    # mom2 to fits
    fits_creation(map_mom2.T,"mom2.fits",cubelow,"km/s")
    fits_creation(map_emom2.T,"emom2.fits",cubelow,"km/s")

    # ratio to fits
    fits_creation(map_ratio.T,"ratio.fits",cubelow,"")
    fits_creation(map_eratio.T,"eratio.fits",cubelow,"")

    # Trot to fits
    fits_creation(map_Trot.T,"Trot.fits",cubelow,"K")
    fits_creation(map_eTrot.T,"eTrot.fits",cubelow,"K")

    # Ncol in log to fits
    fits_creation(map_logN.T,"logN.fits",cubelow,"cm**-2 in log10")
    fits_creation(map_elogN.T,"elogN.fits",cubelow,"cm**-2 in log10")

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
    obj[0].header.append(("BUNIT", bunit))
    obj.writeto(output_map, clobber=True)

    #hdu = pyfits.PrimaryHDU(input_array)
    #hdul = pyfits.HDUList([hdu])
    #hdul.writeto(output_map)

#####################
# fitting functions #
#####################

def _f_linear(x, a, b):
    func = b - a * x

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

def _f_one(x, a1, b, c, freq1):

    offset1 = b /299792.458 * freq1 # km/s

    width1 = c /299792.458 * freq1 # km/s

    func = a1 * np.exp( -(x-freq1+offset1)**2/(2*width1**2) )

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