import os, sys, glob

# data  beam  pixel  imsize
# co10  0.602 0.08   918
# co21  0.600 0.08   452

# /lfs06/saitots/proj_jwu_n6240/derived

# import
co10 = "ngc6240_b3/ngc6240_b3_12m+7m_co10.fits"
co21 = "ngc6240_b6/ngc6240_b6_12m_co21_0p6as.fits"
width = 5

# export
output_co10 = "n6240_co10_cube.fits"
output_co21 = "n6240_co21_cube.fits"

# main
importfits(
    fitsimage = co10,
    imagename = co10.replace(".fits",".image"),
    defaultaxes = True,
    defaultaxesvalues = ["RA","Dec","Frequency","Stokes"],
    )

importfits(
    fitsimage = co21,
    imagename = co21.replace(".fits",".image"),
    defaultaxes = True,
    defaultaxesvalues = ["RA","Dec","Frequency","Stokes"],
    )

os.system("rm -rf " + co21 + "_tmp1")
imrebin(
    imagename = co21.replace(".fits",".image"),
    factor    = [3,3,1],
    box       = "78,78,372,372",
    outfile   = co21 + "_tmp1",
    )
os.system("rm -rf " + co21.replace(".fits",".image"))

os.system("rm -rf " + co10 + "_tmp1")
imregrid(
    imagename = co10.replace(".fits",".image"),
    template  = co21 + "_tmp1",
    axes      = [0,1],
    output    = co10 + "_tmp1",
    )
os.system("rm -rf " + co10.replace(".fits",".image"))

# masking co10
thres_co10 = str(imstat(co10 + "_tmp1")["rms"][0] * 2.5)
os.system("rm -rf " + co10 + "_tmp2")
immath(
    imagename = co10 + "_tmp1",
    expr = "iif(IM0>"+thres_co10+",1.0,0.0)",
    outfile = co10 + "_tmp2",
    )
os.system("rm -rf " + co10 + "_tmp3")
imsmooth(
    imagename = co10 + "_tmp2",
    major = "2.4arcsec",
    minor = "2.4arcsec",
    pa = "0deg",
    outfile = co10 + "_tmp3",
    )
os.system("rm -rf " + co10 + "_tmp4")
specsmooth(
    imagename = co10 + "_tmp3",
    function = "boxcar",
    width = width,
    dmethod = "",
    outfile = co10 + "_tmp4",
    )
os.system("rm -rf " + co10 + "_tmp5")
immath(
    imagename = co10 + "_tmp4",
    expr = "iif(IM0>0.01,1.0,0.0)",
    outfile = co10 + "_tmp5",
    )
os.system("rm -rf " + co10 + "_tmp6")
makemask(
    mode = "copy",
    inpimage = co10 + "_tmp5",
    inpmask = co10 + "_tmp5",
    output = co10 + "_tmp6:mask0",
    overwrite = False,
    )
chans = imhead(co10 + "_tmp1",mode="list")["shape"][3] - 1 - (width-1)/2
chans = str((width-1)/2) + "~" + str(chans)
os.system("rm -rf " + co10 + "_tmp7")
imsubimage(
    imagename = co10 + "_tmp1",
    chans = chans,
    outfile = co10 + "_tmp7",
    )
os.system("rm -rf " + co10 + "_tmp8")
immath(
    imagename = [co10 + "_tmp7", co10 + "_tmp6"],
    expr = "IM0*IM1",
    outfile = co10 + "_tmp8",
    )

# masking co21
thres_co21 = str(imstat(co21 + "_tmp1")['rms'][0] * 2.5)
os.system("rm -rf " + co21 + "_tmp2")
immath(
    imagename = co21 + "_tmp1",
    expr = "iif(IM0>"+thres_co21+",1.0,0.0)",
    outfile = co21 + "_tmp2",
    )
os.system("rm -rf " + co21 + "_tmp3")
imsmooth(
    imagename = co21 + "_tmp2",
    major = "2.4arcsec",
    minor = "2.4arcsec",
    pa = "0deg",
    outfile = co21 + "_tmp3",
    )
os.system("rm -rf " + co21 + "_tmp4")
specsmooth(
    imagename = co21 + "_tmp3",
    function = "boxcar",
    width = width,
    dmethod = "",
    outfile = co21 + "_tmp4",
    )
os.system("rm -rf " + co21 + "_tmp5")
immath(
    imagename = co21 + "_tmp4",
    expr = "iif(IM0>0.01,1.0,0.0)",
    outfile = co21 + "_tmp5",
    )
os.system("rm -rf " + co21 + "_tmp6")
makemask(
    mode = "copy",
    inpimage = co21 + "_tmp5",
    inpmask = co21 + "_tmp5",
    output = co21 + "_tmp6:mask0",
    overwrite = False,
    )
chans = imhead(co21 + "_tmp1",mode="list")["shape"][3] - 1 - (width-1)/2
chans = str((width-1)/2) + "~" + str(chans)
os.system("rm -rf " + co21 + "_tmp7")
imsubimage(
    imagename = co21 + "_tmp1",
    chans = chans,
    outfile = co21 + "_tmp7",
    )
os.system("rm -rf " + co21 + "_tmp8")
immath(
    imagename = [co21 + "_tmp7", co21 + "_tmp6"],
    expr = "IM0*IM1",
    outfile = co21 + "_tmp8",
    )

# export
os.system("rm -rf " + output_co10)
exportfits(
    imagename = co10 + "_tmp8",
    fitsimage = output_co10,
    dropstokes = True,
    dropdeg = True,
    )
os.system("rm -rf " + co10 + "_tmp*")

os.system("rm -rf " + output_co21)
exportfits(
    imagename = co21 + "_tmp8",
    fitsimage = output_co21,
    dropstokes = True,
    dropdeg = True,
    )
os.system("rm -rf " + co21 + "_tmp*")

os.system("rm -rf *.last")

#######
# end #
#######