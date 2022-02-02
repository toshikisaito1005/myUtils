import os, sys, glob
import numpy as np

cube_co10      = "ngc1068_b3_12m+7m_co10.image"
cube_hcn10     = "ngc1068_b3_12m_hcn10.image"
cube_ci10_fov1 = "ngc1068_b8_1_12m+7m_ci10.image"
cube_ci10_fov2 = "ngc1068_b8_2_12m+7m_ci10.image"
cube_ci10_fov3 = "ngc1068_b8_3_12m+7m_ci10.image"
pb_ci10_fov1   = "ngc1068_b8_1_12m+7m_ci10.pb"
pb_ci10_fov2   = "ngc1068_b8_2_12m+7m_ci10.pb"
pb_ci10_fov3   = "ngc1068_b8_3_12m+7m_ci10.pb"

fits_co10_fov1  = "ngc1068_1_12m+7m_co10.fits"
fits_co10_fov2  = "ngc1068_2_12m+7m_co10.fits"
fits_co10_fov3  = "ngc1068_3_12m+7m_co10.fits"
fits_hcn10_fov1 = "ngc1068_1_12m_hcn10.fits"
fits_hcn10_fov2 = "ngc1068_2_12m_hcn10.fits"
fits_hcn10_fov3 = "ngc1068_3_12m_hcn10.fits"
fits_ci10_fov1  = "ngc1068_1_12m+7m_ci10.fits"
fits_ci10_fov2  = "ngc1068_2_12m+7m_ci10.fits"
fits_ci10_fov3  = "ngc1068_3_12m+7m_ci10.fits"

def do_imsmooth(
    imagename,
    beam="0.8arcsec",
    ):
    print("# do_imsmooth: " + imagename)
    outfile = imagename.replace(".image",".smooth")
    os.system("rm -rf " + outfile)
    imsmooth(
    imagename = imagename,
    outfile   = outfile,
    targetres = True,
    major     = beam,
    minor     = beam,
    pa        = "0deg",
    )

    return outfile

def do_align_to_ci10(
    imagename,
    fitsimage,
    cube_ci,
    pb_ci,
    ):
    print("# do_align_to_ci10: " + imagename)
    os.system("rm -rf " + fitsimage + "_tmp1")
    imregrid(
        imagename = imagename,
        template  = cube_ci,
        output    = fitsimage + "_tmp1",
        axes      = [0,1],
        )
    os.system("rm -rf " + pb_ci + "_tmp1")
    imregrid(
        imagename = pb_ci,
        template  = fitsimage + "_tmp1",
        output    = pb_ci + "_tmp1",
        axes      = -1,
        )
    os.system("rm -rf " + fitsimage + "_tmp2")
    immath(
        imagename = [fitsimage + "_tmp1", pb_ci + "_tmp1"],
        mode      = "evalexpr",
        expr      = "iif(IM1>0.25,IM0,0)",
        outfile   = fitsimage + "_tmp2",
        box       = "100,100,299,299",
        )
    imhead(
        imagename = fitsimage + "_tmp2",
        mode      = "put",
        hdkey     = "beammajor",
        hdvalue   = "0.8arcsec",
        )
    imhead(
        imagename = fitsimage + "_tmp2",
        mode      = "put",
        hdkey     = "bunit",
        hdvalue   = "Jy/beam",
        )
    os.system("rm -rf " + fitsimage + "_tmp3")
    imrebin(
        imagename = fitsimage + "_tmp2",
        factor    = [2,2,1,1],
        outfile   = fitsimage + "_tmp3",
        )
    os.system("rm -rf " + fitsimage)
    exportfits(
        imagename = fitsimage + "_tmp3",
        fitsimage = fitsimage,
        velocity  = True,
        )
    os.system("rm -rf " + fitsimage + "_tmp*")
    os.system("rm -rf " + pb_ci + "_tmp1")

def do_exportfits(
    imagename,
    fitsimage,
    ):
    print("# do_exportfits: " + imagename)
    os.system("rm -rf " + fitsimage + "_tmp0")
    immath(
        imagename = imagename,
        mode      = "evalexpr",
        expr      = "IM0",
        outfile   = fitsimage + "_tmp0",
        box       = "100,100,299,299",
        )
    os.system("rm -rf " + fitsimage + "_tmp1")
    imrebin(
        imagename = fitsimage + "_tmp0",
        factor    = [2,2,1,1],
        outfile   = fitsimage + "_tmp1",
        )
    os.system("rm -rf " + fitsimage)
    exportfits(
        imagename = fitsimage + "_tmp1",
        fitsimage = fitsimage,
        velocity  = True,
        )
    os.system("rm -rf " + fitsimage + "_tmp*")

### main
## do_imsmooth
smooth_co10      = do_imsmooth(cube_co10)
smooth_hcn10     = do_imsmooth(cube_hcn10)
smooth_ci10_fov1 = do_imsmooth(cube_ci10_fov1)
smooth_ci10_fov2 = do_imsmooth(cube_ci10_fov2)
smooth_ci10_fov3 = do_imsmooth(cube_ci10_fov3)

## do_align_to_ci10
do_align_to_ci10(cube_co10, fits_co10_fov1, cube_ci10_fov1, pb_ci10_fov1)
do_align_to_ci10(cube_co10, fits_co10_fov2, cube_ci10_fov2, pb_ci10_fov2)
do_align_to_ci10(cube_co10, fits_co10_fov3, cube_ci10_fov3, pb_ci10_fov3)
do_align_to_ci10(cube_hcn10, fits_hcn10_fov1, cube_ci10_fov1, pb_ci10_fov1)
do_align_to_ci10(cube_hcn10, fits_hcn10_fov2, cube_ci10_fov2, pb_ci10_fov2)
do_align_to_ci10(cube_hcn10, fits_hcn10_fov3, cube_ci10_fov3, pb_ci10_fov3)

## do_exportfits
do_exportfits(smooth_ci10_fov1, fits_ci10_fov1)
do_exportfits(smooth_ci10_fov2, fits_ci10_fov2)
do_exportfits(smooth_ci10_fov3, fits_ci10_fov3)

## delete
os.system("rm -rf *.last")
