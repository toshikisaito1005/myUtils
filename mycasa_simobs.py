# This script runs inside CASA. It generates an image appropriate to
# be fed into the CASA simulator to simulate a real PHANGS-alike
# galaxy. It starts with the mask and cube from a full-imaging run
# targeting NGC3059.

import os
import sys
import glob
import numpy as np
import pyfits
import shutil

# import CASA and python stuff
exec(open(os.environ["HOME"]+"/myUtils/stuff_casa.py").read())

############
# gen_cube #
############

def gen_cube(
    template_dir,
    template_file,
    template_mask,
    working_dir,
    template_in_jypix,
    template_clipped,
    template_mask_imported,
    template_rotated,
    template_shrunk,
    template_fullspec,
    template_fullspec_div3,
    template_fullspec_div5,
    template_fullspec_div10,
    template_fullspec_div30,
    template_fullspec_div100,
    template_withcont,
    template_withcont_div3,
    template_withcont_div5,
    template_withcont_div10,
    template_withcont_div30,
    template_withcont_div100,
    pa="23deg", # rotation angle
    shrink=0.5,
    ):
    """
    version from phangs-alma pipeline (sim_ngc3059.py)
    """

    # cleanup directories
    input_dir  = working_dir + "inputs/"
    output_dir = working_dir + "outputs/"
    ms_dir     = working_dir + "ms/"
    #os.system("rm -rf " + input_dir)
    #os.system("rm -rf " + output_dir)
    #os.system("rm -rf " + ms_dir)
    if not glob.glob(input_dir):
        os.mkdir(input_dir)
    if not glob.glob(output_dir):
        os.mkdir(output_dir)
    if not glob.glob(ms_dir):
        os.mkdir(ms_dir)

    template_in_jypix        = input_dir + template_in_jypix
    template_clipped         = input_dir + template_clipped
    template_mask_imported   = input_dir + template_mask_imported
    template_rotated         = input_dir + template_rotated
    template_shrunk          = input_dir + template_shrunk
    template_fullspec        = input_dir + template_fullspec
    template_fullspec_div3   = input_dir + template_fullspec_div3
    template_fullspec_div5   = input_dir + template_fullspec_div5
    template_fullspec_div10  = input_dir + template_fullspec_div10
    template_fullspec_div30  = input_dir + template_fullspec_div30
    template_fullspec_div100 = input_dir + template_fullspec_div100
    template_withcont        = input_dir + template_withcont
    template_withcont_div3   = input_dir + template_withcont_div3
    template_withcont_div5   = input_dir + template_withcont_div5
    template_withcont_div10  = input_dir + template_withcont_div10
    template_withcont_div30  = input_dir + template_withcont_div30
    template_withcont_div100 = input_dir + template_withcont_div100

    ##############################################
    # Convert the template to Jansky/pixel units #
    ##############################################

    bmaj = imhead(template_dir+template_file)["restoringbeam"]["major"]["value"]
    bmin = imhead(template_dir+template_file)["restoringbeam"]["minor"]["value"]
    obsfreq = imhead(template_dir+template_file)["refval"][2] / 1e9 # GHz

    # ... calculate Kelvin-to-Jy per beam using GHz and arcsec
    ktoj = str(8.18255e-7*(obsfreq**2)*(bmaj*bmin))

    # ... calculate pixel scale in arcsec
    pix_arcsec = abs(imhead(template_dir+template_file)["incr"][0])*3600*180/np.pi

    # ... calculate beam area in pixels
    pix_arcsec2 = pix_arcsec**2
    pix_per_beam = str((bmaj/2.*bmin/2.) * np.pi / np.log(2) / pix_arcsec2)

    # ... scale the template to units of Jy per pixel
    os.system("rm -rf " + template_in_jypix)
    immath(
        imagename=template_dir+template_file, 
        expr="IM0*"+ktoj+"/"+pix_per_beam,
        outfile=template_in_jypix,
        )
    imhead(imagename=template_in_jypix,
        mode="put",
        hdkey="bunit",
        hdvalue="Jy/pixel",
        )

    ###############################
    # Apply the mask to the image #
    ###############################

    os.system("rm -rf " + template_clipped)

    importfits(
        template_dir+template_mask,
        imagename=template_mask_imported,
        overwrite=True,
        )
    immath(
        imagename=[template_in_jypix,template_mask_imported],
        expr="IM0*IM1",
        outfile=template_clipped,
        )
    exportfits(
        imagename=template_clipped,
        fitsimage=template_clipped.replace(".image",".fits"),
        overwrite=True,
        )

    ######################################################
    # Rotate the image, includes trimming and NaNs->zero #
    ######################################################

    os.system("rm -rf " + template_rotated+".temp")
    os.system("rm -rf " + template_rotated+".temp.fits")
    os.system("rm -rf " + template_rotated+".temp2")
    os.system("rm -rf " + template_rotated+".temp2.fits")
    os.system("rm -rf " + template_rotated)

    myia.open(template_clipped)
    myia.rotate(outfile=template_rotated+".temp",pa=pa)
    myia.close()

    print("Rotation done.")

    exportfits(
        imagename=template_rotated+".temp",
        fitsimage=template_rotated+".temp.fits",
        overwrite=True,
        )
    hdu = pyfits.open(template_rotated+".temp.fits")
    hdr = hdu[0].header
    del hdr['PC1_1']
    del hdr['PC2_1']
    del hdr['PC3_1']

    del hdr['PC1_2']
    del hdr['PC2_2']
    del hdr['PC3_2']

    del hdr['PC1_3']
    del hdr['PC2_3']
    del hdr['PC3_3']

    del hdr['PV2_1']
    del hdr['PV2_2']
    hdu.writeto(template_rotated+".temp2.fits")
    importfits(
        fitsimage=template_rotated+".temp2.fits",
        imagename=template_rotated+".temp2",           
        overwrite=True,
        )

    myia.open(template_rotated+".temp2")
    cube = myia.getchunk(getmask=True)
    covmap = np.max(cube*1.0,axis=2)
    xspec = np.max(covmap,axis=1)
    yspec = np.max(covmap,axis=0)
    xind = np.where(xspec > 0)
    yind = np.where(yspec > 0)
    xmin = np.min(xind)
    xmax = np.max(xind)
    ymin = np.min(yind)
    ymax = np.max(yind)
    print("Proposed trimming range:")
    print(xmin, xmax, ymin, ymax)
    myia.close()

    imsubimage(
        imagename=template_rotated+".temp2",
        outfile=template_rotated,
        region="box[["+str(xmin)+"pix,"+str(ymin)+"pix],["+str(xmax)+"pix,"+str(ymax)+"pix]]",
        )

    exportfits(
        imagename=template_rotated,
        fitsimage=template_rotated.replace(".image",".fits"),
        overwrite=True,
        )

    #######################################
    # Shrink the pixel scale of the image #
    #######################################

    os.system('rm -rf '+ template_shrunk)
    os.system('cp -r '+ template_rotated + ' '+template_shrunk)

    old_cdelt1 = imhead(imagename = template_shrunk, mode="get", hdkey="cdelt1")
    old_cdelt2 = imhead(imagename = template_shrunk, mode="get", hdkey="cdelt2")

    new_cdelt1 = old_cdelt1
    new_cdelt2 = old_cdelt2
    new_cdelt1['value'] = old_cdelt1['value']*shrink
    new_cdelt2['value'] = old_cdelt2['value']*shrink

    imhead(imagename = template_shrunk, mode="put", hdkey="cdelt1", hdvalue=new_cdelt1)
    imhead(imagename = template_shrunk, mode="put", hdkey="cdelt2", hdvalue=new_cdelt2)

    ##############################################
    # Regrid to an expanded, finer velocity grid #
    ##############################################

    os.system("rm -rf " + template_fullspec)

    target = imregrid(imagename=template_shrunk, template='get')

    target['shap'][2]=110 # 1100
    target['csys']['spectral1']['wcs']['crpix'] = 55 # 550
    target['csys']['spectral1']['wcs']['cdelt'] = 651236.988 # Hz

    imregrid(
        imagename=template_shrunk,
        template=target, 
        output=template_fullspec,
        axes=[2],
        overwrite=True,
        )

    ##############################
    # Add continuum to the image #
    ##############################

    os.system("rm -rf " + template_withcont)
    os.system('cp -r '+template_fullspec + ' '+template_withcont)

    line_to_cont = 30.
    myia.open(template_withcont)
    cube = myia.getchunk()
    maxmap = np.nanmax(cube,axis=2)
    maxmap[np.isfinite(maxmap)==False] = 0.0
    contmap = maxmap/line_to_cont
    nchan = cube.shape[2]
    for ii in range(nchan):
        cube[:,:,ii] = cube[:,:,ii]+contmap
    myia.putchunk(cube)
    myia.close()

    ################################
    # Make several scaled versions #
    ################################

    os.system("rm -rf " + template_withcont_div3)
    os.system("rm -rf " + template_withcont_div5)
    os.system("rm -rf " + template_withcont_div10)
    os.system("rm -rf " + template_withcont_div30)
    os.system("rm -rf " + template_withcont_div100)

    immath(imagename = template_withcont, expr="IM0/3.0", outfile=template_withcont_div3)
    immath(imagename = template_withcont, expr="IM0/5.0", outfile=template_withcont_div5)
    immath(imagename = template_withcont, expr="IM0/10.0", outfile=template_withcont_div10)
    immath(imagename = template_withcont, expr="IM0/30.0", outfile=template_withcont_div30)
    immath(imagename = template_withcont, expr="IM0/100.0", outfile=template_withcont_div100)

    os.system("rm -rf " + template_fullspec_div3)
    os.system("rm -rf " + template_fullspec_div5)
    os.system("rm -rf " + template_fullspec_div10)
    os.system("rm -rf " + template_fullspec_div30)
    os.system("rm -rf " + template_fullspec_div100)

    immath(imagename = template_fullspec, expr="IM0/3.0", outfile=template_fullspec_div3)
    immath(imagename = template_fullspec, expr="IM0/5.0", outfile=template_fullspec_div5)
    immath(imagename = template_fullspec, expr="IM0/10.0", outfile=template_fullspec_div10)
    immath(imagename = template_fullspec, expr="IM0/30.0", outfile=template_fullspec_div30)
    immath(imagename = template_fullspec, expr="IM0/100.0", outfile=template_fullspec_div100)

    # Cleanup
    os.system("rm -rf " + template_rotated+".temp2")
    os.system("rm -rf " + template_rotated+".temp2.fits")

    #######################
    ### Drop back to FITS #
    #######################

    exportfits(imagename=template_fullspec, fitsimage=template_fullspec.replace('.image','.fits'), dropstokes=True, overwrite=True)
    exportfits(imagename=template_withcont, fitsimage=template_withcont.replace('.image','.fits'), dropstokes=True, overwrite=True)

#########
# simtp #
#########
def simtp(
    working_dir,
    template_fullspec,
    sdimage_fullspec,
    sdnoise_image,
    singledish_res="11.8arcsec", # resolution
    singledish_noise=0.102, # Jy/beam at final res
    observed_freq=230538000000.0,
    ):
    """
    Usage:
        Check required sensitivity (singledish_noise) using ACS
    Reference:
        https://almascience.nao.ac.jp/proposing/sensitivity-calculator
    """

    # cleanup directories
    input_dir  = working_dir + "inputs/"
    output_dir = working_dir + "outputs/"
    ms_dir     = working_dir + "ms/"

    if not glob.glob(input_dir):
        os.mkdir(input_dir)
    if not glob.glob(output_dir):
        os.mkdir(output_dir)
    if not glob.glob(ms_dir):
        os.mkdir(ms_dir)

    ###############################################################
    # Make convolved versions that mimic single-dish observations #
    ###############################################################
    template_fullspec = input_dir + template_fullspec
    sdimage_fullspec  = output_dir + sdimage_fullspec
    sdnoise_image     = output_dir + sdnoise_image

    # result should be in Jy/beam

    os.system("rm -rf " + sdimage_fullspec)
    os.system("rm -rf " + sdimage_fullspec + ".temp")
    os.system("rm -rf " + sdnoise_image)
    os.system("rm -rf " + sdnoise_image + ".temp")
    os.system("rm -rf " + sdnoise_image + ".temp2")

    # Make one noise image at the appropriate resolution
    # (NB - noise is the same for all cases)

    # ... create placeholder temp image
    immath(imagename = template_fullspec, expr="IM0*0.0", outfile=sdnoise_image+'.temp')

    # ... add noise to ia
    myia.open(sdnoise_image+'.temp')
    myia.addnoise(type='normal', pars=[0.0,1.0], zero=True)
    myia.close()

    # ... smooth to the singledish res
    imsmooth(imagename = sdnoise_image+'.temp', kernel='gaussian',
             targetres=True, major=singledish_res, minor=singledish_res, pa='0.0deg',
             outfile=sdnoise_image+'.temp2', overwrite=True)

    # ... renormalize to the target value
    stats = imstat(sdnoise_image+'.temp2')
    rescale = singledish_noise/stats['rms'][0]
    os.system("rm -rf " + sdnoise_image)
    immath(imagename = sdnoise_image+'.temp2', expr="IM0*"+str(rescale), outfile=sdnoise_image)

    # Make convolved versions at the single dish resolution
    imsmooth(imagename = template_fullspec,  kernel='gaussian',
             targetres=True, major=singledish_res, minor=singledish_res, pa='0.0deg',
             outfile=sdimage_fullspec+'.temp', overwrite=True)

    # Add the noise to the convolved version
    immath(imagename = [sdimage_fullspec+'.temp', sdnoise_image], 
           expr="IM0+IM1", outfile=sdimage_fullspec+'.temp2')

    # rebin pixel
    pix_as = imhead(sdimage_fullspec+'.temp2',mode="list")["cdelt2"]*3600*180/np.pi
    beam_per_pix = float(singledish_res.replace("arcsec","")) / pix_as

    if beam_per_pix/4.0>=2.0:
        imrebin(imagename=sdimage_fullspec+'.temp2',factor=[np.floor(beam_per_pix/4.0),np.floor(beam_per_pix/4.0),1],
            outfile=sdimage_fullspec)
    else:
        os.system('cp -r '+sdimage_fullspec+'.temp2 '+sdimage_fullspec)

    # put header
    imhead(sdimage_fullspec,mode="put",hdkey="restfreq",hdvalue="4.92160651e+11")
    imhead(sdimage_fullspec,mode="put",hdkey="crval3",hdvalue="4.92160651e+11")

    # J2000
    relabelimage(sdimage_fullspec,icrs_to_j2000=True)

    # Cleanup
    os.system("rm -rf " + sdimage_fullspec+".temp")
    os.system("rm -rf " + sdimage_fullspec+".temp2")
    os.system("rm -rf " + sdnoise_image)
    os.system("rm -rf " + sdnoise_image+".temp")
    os.system("rm -rf " + sdnoise_image+".temp2")

##############
# simobserve #
##############

def run_simobserve(
    working_dir,
    template,
    antennalist,
    project,
    totaltime="1.5h",
    incenter="230GHz",
    ):
    """
    """

    # make sure directories
    input_dir       = working_dir + "inputs/"
    output_dir      = working_dir + "outputs/"
    ms_dir          = working_dir + "./"
    move_ms_to_here = working_dir + "ms/"

    done = glob.glob(move_ms_to_here)
    if not done:
        os.mkdir(move_ms_to_here)

    # simobserve
    simobserve(
        antennalist = antennalist,
        skymodel = input_dir+template,
        project = project,
        setpointings = True,
        integration = "10s",
        graphics = "none",
        obsmode = "int",
        totaltime = totaltime, 
        thermalnoise = 'tsys-atm',
        user_pwv = 1.0,
        overwrite = True,
        incenter = incenter,
        )
    os.system("rm -rf " + move_ms_to_here + "/" + project)
    os.system("mv "+ project + " " + move_ms_to_here + ".")

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

#######
# end #
#######