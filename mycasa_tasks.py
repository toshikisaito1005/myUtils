"""
Standalone routines that take input and soemtimes output using CASA.

contents:
    relabelimage
    remove_small_masks
    beam_area
    imval_all
    measure_rms
    boolean_masking (similar to PHANGS-ALMA's get_mask)
    signal_masking
    run_immath_three
    run_immath_two
    run_immath_one
    run_imregrid
    unit_Jyb_K
    run_exportfits
    run_importfits
    run_immoments
    run_impbcor

history:
2021-07-21   major updates
2021-09-02   add run_run_impbcor, handle data containing nan when measure_rms
2021-09-02   remove ":mask0" from inpmask@makemask
2021-09-06   add remove_small_masks
Toshiki Saito@Nichidai/NAOJ

references:
https://github.com/akleroy/phangs_imaging_scripts/blob/master/phangsPipeline/casaImagingRoutines.py
"""

import os, glob, copy, inspect, datetime
import numpy as np
import pyfits # CASA has pyfits, not astropy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# import CASA and python stuff
exec(open(os.environ["HOME"]+"/myUtils/stuff_casa.py").read())

##########
# common #
##########
modname = "mytask."

def check_first(
    imagename,
    taskname=None,
    ):
    """
    """
    
    if not os.path.isdir(imagename):
        logger.error('Error! The input file "' + imagename + '" was not found!')
        return

    if taskname!=None:
        print("# run " + taskname)
        timestamp(taskname=taskname)
    else:
        print("# no taskname found!")

def timestamp(
    taskname,
    txtfile="task_timestamp.txt",
    ljust1=33,
    ljust2=10,
    ):
    """
    """
    
    ### get timestamp
    this_timestamp = str(datetime.datetime.today()).split(".")[0]
    this_day       = this_timestamp.split(" ")[0]
    this_time      = this_timestamp.split(" ")[1]
    header         = "taskname".ljust(ljust1-2)+" day".ljust(ljust2+1)+" time"
    data_array     = [taskname.ljust(ljust1), this_day.ljust(ljust2), this_time]

    # make sure txtfile
    done = glob.glob(txtfile)

    # save for the first time
    if not done:
        os.system("touch " + txtfile)
        with open(txtfile, "a") as f_handle:
            np.savetxt(f_handle, [data_array], fmt="%s", header=header)

    else:
        data = np.loadtxt(txtfile, dtype="str")
        if data.ndim>1:
            data_limit = data[data[:,0]!=taskname]
            data_save  = np.r_[data_limit, [data_array]]
            data_save[:,0] = [s.ljust(ljust1) for s in data_save[:,0]]
            data_save[:,1] = [s.ljust(ljust2) for s in data_save[:,1]]
            with open(txtfile, "w") as f_handle:
                np.savetxt(f_handle, data_save, fmt="%s", header=header)

        else:
            data_save = [data_array]
            if data[0]==taskname:
                with open(txtfile, "w") as f_handle:
                    np.savetxt(f_handle, data_save, fmt="%s", header=header)

            else:
                with open(txtfile, "a") as f_handle:
                    np.savetxt(f_handle, data_save, fmt="%s")

def f_gauss0(x, a, c):
    """
    """
    
    return a * np.exp( -(x)**2 / (2*c**2) )

def f_lin(x, a, b):
    """
    """
    return a * x + b

############
# imrebin2 #
############

def imrebin2(
    imagename,
    outfile,
    imsize,
    direction_ra,
    direction_dec,
    beam=None,
    oversamplingfactor=4.0,
    delin=False,
    ):
    """
    input : imagename, imsize, direction_ra, direction_dec
    output: ./template.image
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    ### main
    os.system("rm -rf template.im template.fits template.image")

    # prepare
    obsfreq = 115.27120

    # create tempalte image
    blc_ra_tmp  = imstat(imagename)["blcf"].split(", ")[0]
    blc_dec_tmp = imstat(imagename)["blcf"].split(", ")[1]
    blc_ra      = blc_ra_tmp.replace(":","h",1).replace(":","m",1)+"s"
    blc_dec     = blc_dec_tmp.replace(".","d",1).replace(".","m",1)+"s"

    if beam!=None:
        beamsize = round(imhead(imagename,"list")["beamminor"]["value"], 2)
    else:
        beamsize = beam

    pix_size    = round(beamsize/oversamplingfactor, 2)
    size_x      = int(imsize / pix_size)
    size_y      = size_x
    
    print("print test")
    print("print test")
    print("print test")
    print(pix_size, size_x)

    direction   = "J2000 " + direction_ra + " " + direction_dec
    direction   = "J2000 " + blc_ra + " " + blc_dec
    mycl.done()
    mycl.addcomponent(dir=direction,
                      flux=1.0,
                      fluxunit="Jy",
                      freq=str(obsfreq)+"GHz",
                      shape="Gaussian",
                      majoraxis="0.1arcmin",
                      minoraxis="0.05arcmin",
                      positionangle="45.0deg")

    myia.fromshape("template.im",[size_x,size_y,1,1],overwrite=True)
    mycs        = myia.coordsys()
    mycs.setunits(["rad","rad","","Hz"])
    cell_rad    = myqa.convert(myqa.quantity(str(pix_size) + "arcsec"),"rad")["value"]
    mycs.setincrement([-cell_rad,cell_rad],"direction")
    mycs.setreferencevalue([myqa.convert(blc_ra,"rad")["value"],
                            myqa.convert(blc_dec,"rad")["value"]],
                           type = "direction")
    mycs.setreferencevalue(str(obsfreq)+"GHz","spectral")
    mycs.setincrement("1GHz","spectral")
    myia.setcoordsys(mycs.torecord())
    myia.setbrightnessunit("Jy/pixel")
    myia.modify(mycl.torecord(),subtract=False)
    exportfits(imagename = "template.im",
               fitsimage = "template.fits",
               overwrite = True)

    os.system("rm -rf template.image")
    importfits(fitsimage = "template.fits",
               imagename = "template.image")

    myia.close()
    mycl.close()

    os.system("rm -rf template.im template.fits")

    # regrid
    run_imregrid(imagename,"template.image",outfile,axes=[0,1])

    os.system("rm -rf template.image")

    if delin==False:
        os.system("rm -rf " + imagename)

################
# relabelimage #
################
def relabelimage(
    imagename,
    icrs_to_j2000=False,
    j2000_to_icrs=False,
    ):
    """
    # Relabel a ICRS (J2000) image to J2000 (ICRS)
    # Modified version of relabelimagetoicrs.py by D.Petry (ESO), 2016-03-04
    # https://help.almascience.org/index.php?/Knowledgebase/Article/View/352
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    myia.open(imagename)
    mycs = myia.coordsys().torecord()

    if icrs_to_j2000==True:
        if mycs['direction0']['conversionSystem'] == 'ICRS':
            mycs['direction0']['conversionSystem'] = 'J2000'
            print("Found ICRS conversion system and changed it to J2000")

        if mycs['direction0']['system'] == 'ICRS':
            mycs['direction0']['system'] = 'J2000'
            print("Found ICRS direction system and changed it to J2000")
    
    if j2000_to_icrs==True:
        if mycs['direction0']['conversionSystem'] == 'J2000':
            mycs['direction0']['conversionSystem'] = 'ICRS'
            print("Found J2000 conversion system and changed it to ICRS")

        if mycs['direction0']['system'] == 'J2000':
            mycs['direction0']['system'] = 'ICRS'
            print("Found J2000 direction system and changed it to ICRS")

    myia.setcoordsys(mycs)
    myia.close()

######################
# remove_small_masks #
######################
def remove_small_masks(
    maskname,
    output=None,
    imagename=None,
    pixelmin=1.0,
    ):
    """
    remove small masks from a mask image or cube based on beam size info.
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    if imagename==None:
        print("# measure beam area of " + maskname)
        beamarea = beam_area(maskname)
    else:
        print("# measure beam area of " + imagename)
        beamarea = beam_area(imagename)

    if output!=None:
        os.system("cp -r " + maskname + " " + output)
    else:
        output = maskname

    myia.open(output)
    mask=myia.getchunk()
    labeled,j=scipy.ndimage.label(mask)
    myhistogram=scipy.ndimage.measurements.histogram(labeled,0,j+1,j+1)
    object_slices=scipy.ndimage.find_objects(labeled)
    threshold_area=beamarea*pixelmin
    for i in range(j):
        if myhistogram[i+1]<threshold_area:
            mask[object_slices[i]]=0

    myia.putchunk(mask)
    myia.done()

#############
# beam_area #
#############
def beam_area(
    imagename,
    ):
    """
    measure beam area per pixel
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    bmaj_as   = imhead(imagename=imagename,mode="get",hdkey="beammajor")["value"]
    bmin_as   = imhead(imagename=imagename,mode="get",hdkey="beamminor")["value"]
    pix_as    = abs(imhead(imagename=imagename,mode="list")["cdelt1"]) * 3600 * 180 / np.pi
    barea_as  = bmaj_as * bmin_as * np.pi/(4*np.log(2))
    barea_pix = barea_as / (pix_as ** 2)

    return barea_pix

#############
# imval_all #
#############
def imval_all(
    imagename,
    region=None,
    ):

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)
    
    # imval
    if region==None:
        shape = imhead(imagename,mode="list")["shape"]
        box   = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)
        data  = imval(imagename,box=box)
        return data, box

    else:
        data  = imval(imagename,region=region)
        return data
    
###############
# measure_rms #
###############
def measure_rms(
    imagename,
    snr=3.0,
    rms_or_p84 = "p84",
    ):
    """
    measure single rms value per image (2D or 3D)
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)
    
    #imval
    data,_ = imval_all(imagename)
    data = data["data"]*data["mask"]
    data = data.flatten()
    data = data[abs(data)!=0]
    p84  = np.nanpercentile(data, 16) * -1
    data = data[data<p84*snr]

    # measure rms assuming Gaussian noise
    if rms_or_p84=="rms":
        # get histogram
        bins   = (np.ceil(np.log2(len(data))) + 1) * 20 # Sturges equation * 20
        hrange = [np.nanmin(data), np.nanmax(data)]
        hdata  = np.histogram(data, bins=bins, range=hrange)
        histx, histy = hdata[1][:-1], hdata[0]
        hmax   = np.nanmax(histy)

        # Gaussian fit
        popt, pcov = curve_fit(
            f_gauss0,
            histx,
            histy,
            p0=[hmax,p84],
            maxfev=10000,
            )

        return popt[1]

    elif rms_or_p84=="p84":
        # use percentile
        return p84

###################
# boolean_masking #
###################
def boolean_masking(
    imagename,
    outfile,
    delin=False,
    ):
    """
    see this: https://github.com/akleroy/phangs_imaging_scripts/blob/61be1520f0a17360406db940bf1cd8eed84b4c4a/phangsPipeline/casaCubeRoutines.py#L103
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)
    
    imhead(imagename,mode="del",hdkey="beammajor")
    
    os.system("rm -rf " + outfile)
    makemask(
        mode      = "copy",
        inpimage  = imagename,
        inpmask   = imagename, # + ":mask0",
        output    = outfile,
        overwrite = True,
        )
    
    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

##################
# signal_masking #
##################
def signal_masking(
    imagename,
    outfile,
    threshold,
    delin=False,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    os.system("rm -rf " + outfile)
    os.system("rm -rf " + outfile + "_tmp0_signal_masking")
    expr = "iif(IM0 > " + str(threshold) + ", 1.0, 0.0)"
    immath(
        imagename = imagename,
        mode      = "evalexpr",
        expr      = expr,
        outfile   = outfile + "_tmp0_signal_masking",
        )

    imhead(
        imagename = outfile + "_tmp0_signal_masking",
        mode      = "del",
        hdkey     = "beammajor",
        )

    boolean_masking(
        imagename = outfile + "_tmp0_signal_masking",
        outfile   = outfile,
        delin     = True,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

####################
# run_immath_three #
####################
def run_immath_three(
    imagename1,
    imagename2,
    imagename3,
    outfile,
    expr,
    chans="",
    delin=False,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename1, taskname)

    os.system("rm -rf " + outfile)
    immath(
        imagename = [imagename1,imagename2,imagename3],
        expr      = expr,
        chans     = chans,
        outfile   = outfile,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename1)
        os.system("rm -rf " + imagename2)
        os.system("rm -rf " + imagename3)

##################
# run_immath_two #
##################
def run_immath_two(
    imagename1,
    imagename2,
    outfile,
    expr,
    chans="",
    delin=False,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename1, taskname)
    check_first(imagename2)
    
    os.system("rm -rf " + outfile)
    immath(
        imagename = [imagename1,imagename2],
        expr      = expr,
        chans     = chans,
        outfile   = outfile,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename1)
        os.system("rm -rf " + imagename2)

##################
# run_immath_one #
##################
def run_immath_one(
    imagename,
    outfile,
    expr,
    chans="",
    delin=False,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    os.system("rm -rf " + outfile)
    immath(
        imagename = imagename,
        expr      = expr,
        chans     = chans,
        outfile   = outfile,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

################
# run_imregrid #
################
def run_imregrid(
    imagename,
    template,
    outfile,
    axes=-1,
    delin=False,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    os.system("rm -rf " + outfile)
    imregrid(
        imagename = imagename,
        template  = template,
        output    = outfile,
        axes      = axes,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

##############
# unit_Jyb_K #
##############
def unitconv_Jyb_K(
    imagename,
    outfile,
    restfreq_ghz=None,
    unitto="K",
    delin=False,
    ):
    """
    unit conversion between Jy/beam and Kelvin.
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)
    
    # check imagename unit
    this_unit = imhead(imagename,mode="list")["bunit"]

    if this_unit==unitto:
        print("# " + taskname + ": copy imagename to outfile")
        os.system("cp -r " + imagename + " " + outfile)

    else:
        print("# " + taskname + ": do conversion")
        hlist = imhead(imagename,mode="list")
        bmaj  = hlist["beammajor"]["value"]
        bmin  = hlist["beamminor"]["value"]

        if restfreq_ghz!=None:
            restfreq_ghz = hlist["restfreq"][0]/1e9

        factor = 1.222e6 / bmaj / bmin / restfreq_ghz**2

        if unitto=="Jy/beam":
            print("# " + taskname + ": convert to " + unitto)
            os.system("rm -rf " + outfile)
            immath(
                imagename = imagename,
                expr      = "IM0/" + str(factor),
                outfile   = outfile)
            imhead(outfile,mode="put",hdkey="bunit",hdvalue=unitto)

        elif unitto=="K":
            print("# " + taskname + ": convert to " + unitto)
            os.system("rm -rf " + outfile)
            immath(
                imagename = imagename,
                expr      = "IM0*" + str(factor),
                outfile   = outfile)
            imhead(outfile,mode="put",hdkey="bunit",hdvalue=unitto)

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

###############
# run_imrebin #
###############
def run_imrebin(
    imagename,
    outfile,
    factor=[2,2,1],
    delin=False,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    # check if Stokes axis...
    header = imhead(imagename,mode="list")
    shape  = header["shape"]

    if "Stokes" in header.values():
        if shape[-1]!=1:
            if not glob.glob(imagename + ".trans"):
                index = np.where(shape==1)[0][0]
                order = np.array(range(len(shape)))
                order = np.array(np.r_[order[order!=index],index], dtype=str)
                order = "".join(order)

                print("# imstrans 0123 to " + str(order) + ".")
                imtrans(
                    imagename = imagename,
                    outfile   = imagename + ".trans",
                    order     = order,
                    )

            imagename += ".trans"
        factor += [1]

    #
    os.system("rm -rf " + outfile)
    imrebin(
        imagename = imagename,
        outfile   = outfile,
        factor    = factor,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

##################
# run_exportfits #
##################
def run_exportfits(
    imagename,
    fitsimage,
    dropdeg=False,
    dropstokes=False,
    delin=False,
    velocity=False,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    # run exportfits
    os.system("rm -rf " + fitsimage)
    exportfits(
        fitsimage  = fitsimage,
        imagename  = imagename,
        dropdeg    = dropdeg,
        dropstokes = dropstokes,
        velocity   = velocity,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

##################
# run_importfits #
##################
def run_importfits(
    fitsimage,
    imagename,
    defaultaxes=False,
    delin=False,
    defaultaxesvalues=["RA","Dec","Frequency","Stokes"],
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    # importfits
    os.system("rm -rf " + imagename)
    importfits(
        fitsimage         = fitsimage,
        imagename         = imagename,
        defaultaxes       = defaultaxes,
        defaultaxesvalues = defaultaxesvalues,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + fitsimage)

#################
# run_immoments #
#################
def run_immoments(
    imagename,
    maskimage,
    outfile,
    mom=0,
    rms=1.0,
    snr=3.0,
    outfile_err=None,
    outfile_snr=None,
    vdim=2,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    # masking
    run_immath_two(
        imagename,
        maskimage,
        imagename + "_tmp1",
        "IM0*IM1",
        )

    # moment
    os.system("rm -rf " + outfile)
    immoments(
        imagename  = imagename + "_tmp1",
        moments    = [mom],
        includepix = [rms*snr,1e11],
        outfile    = outfile,
        )

    # mom0 err
    if mom==0 and outfile_err!=None:
        os.system("rm -rf " + outfile_err + "_tmp1")
        immoments(
            imagename = maskimage,
            moments   = [0],
            outfile   = outfile_err + "_tmp1",
            )

        # measure channel width
        data = imval(imagename)["coords"][:,vdim]
        restfreq = imhead(imagename,mode="list")["restfreq"][0]
        chanwidth = str(np.round(abs(data[1]-data[0])/restfreq * 299792.458, 2))
        print("# channel width = " + chanwidth + " km/s")

        # error map
        run_immath_one(
            outfile_err + "_tmp1",
            outfile_err,
            str(rms) + "*" + chanwidth + "*sqrt(IM0/" + chanwidth + ")",
            delin=True,
            )

        # snr map
        if outfile_snr!=None:
            run_immath_two(
                outfile,
                outfile_err,
                "IM0/IM1",
                outfile_snr,
                )

###############
# run_impbcor #
###############
def run_impbcor(
    imagename,
    pbimage,
    outfile,
    mode="divide",
    cutoff=0.25,
    delin=False,
    ):
    """
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    # importfits
    os.system("rm -rf " + outfile)
    impbcor(
        imagename = imagename,
        outfile = outfile,
        pbimage = pbimage,
        mode = mode,
        cutoff = cutoff,
        )

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

###################
# run_roundsmooth #
###################
def run_roundsmooth(
    imagename,
    outfile,
    targetbeam, # float, arcsec unit
    inputbeam=None,
    delin=False,
    targetres=True,
    ):
    """
    input : imagename, targetbeam
    output: outfile
    """

    taskname = modname + sys._getframe().f_code.co_name
    check_first(imagename, taskname)

    # check the input beam << targetbeam
    if inputbeam==None:
        inputbeam = np.round(imhead(imagename,mode="list")["beammajor"]["value"],2)

    # imsmooth
    os.system("rm -rf " + outfile)
    if inputbeam>=targetbeam:
        print("# skip run_roundsmooth because beam of " + imagename + " is larger than " + str(targetbeam) + " arcsec")
        os.system("cp -r " + imagename + " " + outfile)

    else:
        print("# run_roundsmooth")
        imsmooth(
            imagename = imagename,
            targetres = targetres,
            major     = str(targetbeam)+"arcsec",
            minor     = str(targetbeam)+"arcsec",
            pa        = "0deg",
            outfile   = outfile)

    # delete input
    if delin==True:
        os.system("rm -rf " + imagename)

#######
# end #
#######