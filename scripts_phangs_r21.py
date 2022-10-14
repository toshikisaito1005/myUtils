"""
Python class for the PHANGS-R21 project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:


usage:
> import os
> from scripts_phangs_r21 import ToolsR21 as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_projects/galkey_phangs.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_projects/key_phangs_r21.txt",
>     )
>
> # main
> tl.run_phangs_r21(
>     do_all           = True,
>     # analysis
>     do_prepare       = True,
>     # plot
>     plot_showcase    = True,
>     # supplement
>     )
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To

history:
2017-11-01   created
2022-07-28   constructed align_cubes
2022-07-29   constructed multismooth
Toshiki Saito@NAOJ
"""

import os, sys, glob
import numpy as np
from scipy.stats import gaussian_kde

from mycasa_rotation import *
from mycasa_sampling import *
from mycasa_lowess import *
from mycasa_tasks import *
from mycasa_plots import *
from mycasa_pca import *

############
# ToolsR21 #2
############
class ToolsR21():
    """
    Class for the PHANGS-R21 project.
    """

    ############
    # __init__ #
    ############

    def __init__(
        self,
        keyfile_fig  = None,
        keyfile_gal  = None,
        refresh      = False,
        delete_inter = True,
        ):
        # initialize keys
        self.keyfile_gal = keyfile_gal
        self.keyfile_fig = keyfile_fig

        # intialize task
        self.refresh      = refresh
        self.delete_inter = delete_inter
        self.taskname     = None

        # initialize directories
        self.dir_raw      = None
        self.dir_ready    = None
        self.dir_other    = None
        self.dir_products = None
        self.fig_dpi      = 200

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ToolsR21."
            self._set_dir()            # directories
            self._set_input_fits()     # input maps
            self._set_output_fits()    # output maps
            self._set_input_param()    # input parameters
            self._set_output_txt_png() # output txt and png

    def _set_dir(self):
        """
        """

        self.dir_proj     = self._read_key("dir_proj")
        self.dir_raw      = self.dir_proj + self._read_key("dir_raw")
        self.dir_cprops   = self.dir_proj + self._read_key("dir_cprops")
        self.dir_env      = self.dir_proj + self._read_key("dir_env")
        self.dir_piechart = self.dir_proj + self._read_key("dir_piechart")
        self.dir_wise     = self.dir_proj + self._read_key("dir_wise")
        self.dir_ready    = self.dir_proj + self._read_key("dir_ready")
        self.dir_products = self.dir_proj + self._read_key("dir_products")
        self.dir_final    = self.dir_proj + self._read_key("dir_final")

        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_final)

    def _set_input_fits(self):
        """
        """

        self.cube_co10_n0628 = self.dir_raw + self._read_key("cube_co10_n0628")
        self.cube_co10_n3627 = self.dir_raw + self._read_key("cube_co10_n3627")
        self.cube_co10_n4254 = self.dir_raw + self._read_key("cube_co10_n4254")
        self.cube_co10_n4321 = self.dir_raw + self._read_key("cube_co10_n4321")

        self.cube_co21_n0628 = self.dir_raw + self._read_key("cube_co21_n0628")
        self.cube_co21_n3627 = self.dir_raw + self._read_key("cube_co21_n3627")
        self.cube_co21_n4254 = self.dir_raw + self._read_key("cube_co21_n4254")
        self.cube_co21_n4321 = self.dir_raw + self._read_key("cube_co21_n4321")

        self.wise1_n0628     = self.dir_wise + self._read_key("wise1_n0628")
        self.wise1_n3627     = self.dir_wise + self._read_key("wise1_n3627")
        self.wise1_n4254     = self.dir_wise + self._read_key("wise1_n4254")
        self.wise1_n4321     = self.dir_wise + self._read_key("wise1_n4321")

        self.wise2_n0628     = self.dir_wise + self._read_key("wise2_n0628")
        self.wise2_n3627     = self.dir_wise + self._read_key("wise2_n3627")
        self.wise2_n4254     = self.dir_wise + self._read_key("wise2_n4254")
        self.wise2_n4321     = self.dir_wise + self._read_key("wise2_n4321")

        self.wise3_n0628     = self.dir_wise + self._read_key("wise3_n0628")
        self.wise3_n3627     = self.dir_wise + self._read_key("wise3_n3627")
        self.wise3_n4254     = self.dir_wise + self._read_key("wise3_n4254")
        self.wise3_n4321     = self.dir_wise + self._read_key("wise3_n4321")

    def _set_output_fits(self):
        """
        """

        self.outcube_co10_n0628  = self.dir_ready + self._read_key("outcube_co10_n0628")
        self.outcube_co10_n3627  = self.dir_ready + self._read_key("outcube_co10_n3627")
        self.outcube_co10_n4254  = self.dir_ready + self._read_key("outcube_co10_n4254")
        self.outcube_co10_n4321  = self.dir_ready + self._read_key("outcube_co10_n4321")

        self.outcube_co21_n0628  = self.dir_ready + self._read_key("outcube_co21_n0628")
        self.outcube_co21_n3627  = self.dir_ready + self._read_key("outcube_co21_n3627")
        self.outcube_co21_n4254  = self.dir_ready + self._read_key("outcube_co21_n4254")
        self.outcube_co21_n4321  = self.dir_ready + self._read_key("outcube_co21_n4321")

        self.outfits_wise1_n0628 = self.dir_ready + self._read_key("outfits_wise1_n0628")
        self.outfits_wise1_n3627 = self.dir_ready + self._read_key("outfits_wise1_n3627")
        self.outfits_wise1_n4254 = self.dir_ready + self._read_key("outfits_wise1_n4254")
        self.outfits_wise1_n4321 = self.dir_ready + self._read_key("outfits_wise1_n4321")

        self.outfits_wise2_n0628 = self.dir_ready + self._read_key("outfits_wise2_n0628")
        self.outfits_wise2_n3627 = self.dir_ready + self._read_key("outfits_wise2_n3627")
        self.outfits_wise2_n4254 = self.dir_ready + self._read_key("outfits_wise2_n4254")
        self.outfits_wise2_n4321 = self.dir_ready + self._read_key("outfits_wise2_n4321")

        self.outfits_wise3_n0628 = self.dir_ready + self._read_key("outfits_wise3_n0628")
        self.outfits_wise3_n3627 = self.dir_ready + self._read_key("outfits_wise3_n3627")
        self.outfits_wise3_n4254 = self.dir_ready + self._read_key("outfits_wise3_n4254")
        self.outfits_wise3_n4321 = self.dir_ready + self._read_key("outfits_wise3_n4321")

        self.outmom_co10_n0628   = self.outcube_co10_n0628.replace(".image",".momX")
        self.outmom_co10_n3627   = self.outcube_co10_n3627.replace(".image",".momX")
        self.outmom_co10_n4254   = self.outcube_co10_n4254.replace(".image",".momX")
        self.outmom_co10_n4321   = self.outcube_co10_n4321.replace(".image",".momX")

        self.outmom_co21_n0628   = self.outcube_co21_n0628.replace(".image",".momX")
        self.outmom_co21_n3627   = self.outcube_co21_n3627.replace(".image",".momX")
        self.outmom_co21_n4254   = self.outcube_co21_n4254.replace(".image",".momX")
        self.outmom_co21_n4321   = self.outcube_co21_n4321.replace(".image",".momX")


    def _set_input_param(self):
        """
        """

        self.ra_n0628       = self._read_key("ra_n0628", "gal").split("deg")[0]
        self.ra_n3627       = self._read_key("ra_n3627", "gal").split("deg")[0]
        self.ra_n4254       = self._read_key("ra_n4254", "gal").split("deg")[0]
        self.ra_n4321       = self._read_key("ra_n4321", "gal").split("deg")[0]

        self.dec_n0628      = self._read_key("dec_n0628", "gal").split("deg")[0]
        self.dec_n3627      = self._read_key("dec_n3627", "gal").split("deg")[0]
        self.dec_n4254      = self._read_key("dec_n4254", "gal").split("deg")[0]
        self.dec_n4321      = self._read_key("dec_n4321", "gal").split("deg")[0]

        self.basebeam_n0628 = float(self._read_key("basebeam_n0628"))
        self.basebeam_n3627 = float(self._read_key("basebeam_n3627"))
        self.basebeam_n4254 = float(self._read_key("basebeam_n4254"))
        self.basebeam_n4321 = float(self._read_key("basebeam_n4321"))

        self.imsize_n0628   = float(self._read_key("imsize_n0628"))
        self.imsize_n3627   = float(self._read_key("imsize_n3627"))
        self.imsize_n4254   = float(self._read_key("imsize_n4254"))
        self.imsize_n4321   = float(self._read_key("imsize_n4321"))

        self.chans_n0628    = self._read_key("chans_n0628")
        self.chans_n3627    = self._read_key("chans_n3627")
        self.chans_n4254    = self._read_key("chans_n4254")
        self.chans_n4321    = self._read_key("chans_n4321")

        self.beams_n0628    = [float(s) for s in self._read_key("beams_n0628").split(",")]
        self.beams_n3627    = [float(s) for s in self._read_key("beams_n3627").split(",")]
        self.beams_n4254    = [float(s) for s in self._read_key("beams_n4254").split(",")]
        self.beams_n4321    = [float(s) for s in self._read_key("beams_n4321").split(",")]

        self.freq_co10      = 115.27120
        self.freq_co21      = 230.53800

        self.snr_mom        = 4.0

        self.nchan_thres_n0628 = 2
        self.nchan_thres_n3627 = 3
        self.nchan_thres_n4254 = 3
        self.nchan_thres_n4321 = 2

    def _set_output_txt_png(self):
        """
        """

        # output txt and png
        #self.outpng_mom0_13co10 = self.dir_products + self._read_key("outpng_mom0_13co10")

    ##################
    # run_phangs_r21 #
    ##################

    def run_phangs_r21(
        self,
        do_all         = False,
        # analysis
        do_align       = False,
        do_multismooth = False,
        do_moments     = False,
        # plot figures in paper
        plot_showcase  = False,
        # supplement
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        if do_all==True:
            do_prepare = True

        # analysis
        if do_align==True:
            self.align_cubes()
        
        if do_multismooth==True:
            self.multismooth()

        if do_moments==True:
            self.multimoments()

        # plot figures in paper
        #if plot_showcase==True:
        #    self.showcase()

    ################
    # multimoments #
    ################

    def multimoments(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        incube_co10 = self.outcube_co10_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image")
        incube_co21 = self.outcube_co21_n0628.replace(str(self.basebeam_n0628).replace(".","p").zfill(4),"????").replace(".image","_k.image")
        outmom_co10 = self.outmom_co10_n0628
        outmom_co21 = self.outmom_co21_n0628
        this_beams  = self.beams_n0628
        nchan_thres = self.nchan_thres_n0628

        for i in range(len(this_beams)):
            this_beam       = this_beams[i]
            this_beamstr    = str(this_beam).replace(".","p").zfill(4)
            print(this_beamstr)
            mask_co10       = "co10.mask"
            mask_co21       = "co21.mask"
            mask_combine    = "comb_" + this_beamstr + ".mask"
            this_input_co10 = incube_co10.replace("????",this_beamstr)
            this_input_co21 = incube_co21.replace("????",this_beamstr)

            # snr-based masking
            self._maskig_cube_snr(this_input_co10,mask_co10+"_snr")
            self._maskig_cube_snr(this_input_co21,mask_co21+"_snr")
            run_immath_two(mask_co10+"_snr",mask_co21+"_snr",mask_combine,"IM0*IM1",delin=True)

            # nchan-based masking
            self._masking_cube_nchan(this_input_co10,mask_co10+"_nchan",nchan_thres=nchan_thres)
            run_immath_two(mask_combine,mask_co10+"_nchan",mask_co10,"IM0*IM1",delin=False)
            os.system("rm -rf " + mask_co10 + "_nchan")

            self._masking_cube_nchan(this_input_co21,mask_co21+"_nchan",nchan_thres=nchan_thres)
            run_immath_two(mask_combine,mask_co21+"_nchan",mask_co21,"IM0*IM1",delin=True)

            # mom
            self._eazy_immoments(this_input_co10,mask_co10,outmom_co10)
            self._eazy_immoments(this_input_co21,mask_co21,outmom_co21)

            # clean up
            os.system("rm -rf " + mask_co10)
            os.system("rm -rf " + mask_co21)

    ###################
    # _eazy_immoments #
    ###################

    def _eazy_immoments(
        self,
        incube,
        inmask,
        baseoutmom,
        ):
        """
        """

        rms = measure_rms(incube)

        outfile  = baseoutmom.replace("momX","mom0")
        outefile = baseoutmom.replace("momX","emom0")
        run_immoments(incube,inmask,outfile,mom=0,rms=rms,snr=self.snr_mom,outfile_err=outefile,vdim=3)
        outfile  = baseoutmom.replace("momX","mom1")
        run_immoments(incube,inmask,outfile,mom=1,rms=rms,snr=self.snr_mom,vdim=3)
        outfile  = baseoutmom.replace("momX","mom2")
        run_immoments(incube,inmask,outfile,mom=2,rms=rms,snr=self.snr_mom,vdim=3)
        outfile  = baseoutmom.replace("momX","mom8")
        run_immoments(incube,inmask,outfile,mom=8,rms=rms,snr=self.snr_mom,vdim=3)

    #######################
    # _masking_cube_nchan #
    #######################

    def _masking_cube_nchan(
        self,
        incube,
        outmask,
        snr=2.0,
        pixelmin=1,
        nchan_thres=2,
        ):
        """
        """

        thres  = str( measure_rms(incube) * snr )
        data   = imval(incube)["coords"][:,3]
        cwidth = str(np.round(abs(data[1]-data[0])/imhead(incube,mode="list")["restfreq"][0] * 299792.458, 2))

        # create nchan 3d mask
        expr = "iif( IM0>=" + thres + ",1.0/" + cwidth + ",0.0 )"
        run_immath_one(incube,incube+"_tmp1",expr)
        immoments(imagename=incube+"_tmp1",moments=[0],outfile=incube+"_tmp2")

        # remove islands
        maskfile = incube + "_tmp2"
        beamarea = beam_area(maskfile)

        myia.open(maskfile)
        mask           = myia.getchunk()
        labeled, j     = scipy.ndimage.label(mask)
        myhistogram    = scipy.ndimage.measurements.histogram(labeled,0,j+1,j+1)
        object_slices  = scipy.ndimage.find_objects(labeled)
        threshold_area = beamarea*pixelmin
        for i in range(j):
            if myhistogram[i+1]<threshold_area:
                mask[object_slices[i]] = 0
        myia.putchunk(mask)
        myia.done()

        # create nchan 2d mask
        expr = "iif( IM0>="+str(nchan_thres)+", 1, 0 )"
        run_immath_one(incube+"_tmp2",incube+"_tmp3",expr,delin=True)
        boolean_masking(incube+"_tmp3",outmask,delin=True)

    ####################
    # _maskig_cube_snr #
    ####################

    def _maskig_cube_snr(
        self,
        incube,
        outmask,
        convtos=[3.0,5.0,7.0],
        snrs=[1.0,1.0,1.0],
        ):
        """
        """

        smcubes = [incube+".sm1",incube+".sm2",incube+".sm3"]
        smmasks = [s+".mask" for s in smcubes]
        bmaj    = imhead(incube, mode="list")["beammajor"]["value"]

        # multi smooth
        for i in range(len(smcubes)):
            this_smcube = smcubes[i]
            this_smmask = smmasks[i]
            this_sm     = convtos[i]
            this_snr    = snrs[i]
            this_smbeam = bmaj*this_sm # float, arcsec

            run_roundsmooth(incube,this_smcube,this_smbeam,inputbeam=bmaj)
            this_smrms = measure_rms(this_smcube)
            signal_masking(this_smcube,this_smmask,this_smrms*this_snr,delin=True)

        # combine
        expr = "iif( IM0+IM1+IM2>=2.0, 1, 0 )"
        run_immath_three(smmasks[0],smmasks[1],smmasks[2],outmask,expr,delin=True)

    ###############
    # multismooth #
    ###############

    def multismooth(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outcube_co10_n0628,taskname)

        self._loop_roundsmooth(
            self.outcube_co10_n0628,self.beams_n0628[1:],self.basebeam_n0628,
            self.imsize_n0628,self.ra_n0628,self.dec_n0628,self.freq_co10)
        self._loop_roundsmooth(
            self.outcube_co21_n0628,self.beams_n0628[1:],self.basebeam_n0628,
            self.imsize_n0628,self.ra_n0628,self.dec_n0628,self.freq_co21)

        self._loop_roundsmooth(
            self.outcube_co10_n3627,self.beams_n3627[1:],self.basebeam_n3627,
            self.imsize_n3627,self.ra_n3627,self.dec_n3627,self.freq_co10)
        self._loop_roundsmooth(
            self.outcube_co21_n3627,self.beams_n3627[1:],self.basebeam_n3627,
            self.imsize_n3627,self.ra_n3627,self.dec_n3627,self.freq_co21)

        self._loop_roundsmooth(
            self.outcube_co10_n4254,self.beams_n4254[1:],self.basebeam_n4254,
            self.imsize_n4254,self.ra_n4254,self.dec_n4254,self.freq_co10)
        self._loop_roundsmooth(
            self.outcube_co21_n4254,self.beams_n4254[1:],self.basebeam_n4254,
            self.imsize_n4254,self.ra_n4254,self.dec_n4254,self.freq_co21)

        self._loop_roundsmooth(
            self.outcube_co10_n4321,self.beams_n4321[1:],self.basebeam_n4321,
            self.imsize_n4321,self.ra_n4321,self.dec_n4321,self.freq_co10)
        self._loop_roundsmooth(
            self.outcube_co21_n4321,self.beams_n4321[1:],self.basebeam_n4321,
            self.imsize_n4321,self.ra_n4321,self.dec_n4321,self.freq_co21)

    #####################
    # _loop_roundsmooth #
    #####################

    def _loop_roundsmooth(
        self,
        incube,
        beams,
        basebeam,
        imsize,
        ra,
        dec,
        freq,
        ):
        """
        multismooth
        """

        outcube_template = incube.replace(str(basebeam).replace(".","p").zfill(4),"????")
        this_beams       = beams

        unitconv_Jyb_K(incube,incube.replace(".image","_k.image"),freq)

        for i in range(len(this_beams)):
            this_beam    = this_beams[i]
            this_beamstr = str(this_beam).replace(".","p").zfill(4)
            this_outfile = outcube_template.replace("????",this_beamstr)

            print("# create " + this_outfile.split("/")[-1])

            run_roundsmooth(incube,this_outfile+"_tmp1",this_beam,inputbeam=basebeam)
            make_gridtemplate(this_outfile+"_tmp1",this_outfile,imsize,ra,dec,this_beam)
            unitconv_Jyb_K(this_outfile,this_outfile.replace(".image","_k.image"),freq)
            os.system("rm -rf template.image")
            os.system("rm -rf " + this_outfile + "_tmp1")

    ###############
    # align_cubes #
    ###############

    def align_cubes(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cube_co10_n0628,taskname)

        self._align_cube_gal(self.cube_co10_n0628,self.cube_co21_n0628,
            self.outcube_co10_n0628,self.outcube_co21_n0628,self.basebeam_n0628,
            self.imsize_n0628,self.ra_n0628,self.dec_n0628,self.chans_n0628)

        self._align_cube_gal(self.cube_co10_n3627,self.cube_co21_n3627,
            self.outcube_co10_n3627,self.outcube_co21_n3627,self.basebeam_n3627,
            self.imsize_n3627,self.ra_n3627,self.dec_n3627,self.chans_n3627)

        self._align_cube_gal(self.cube_co10_n4254,self.cube_co21_n4254,
            self.outcube_co10_n4254,self.outcube_co21_n4254,self.basebeam_n4254,
            self.imsize_n4254,self.ra_n4254,self.dec_n4254,self.chans_n4254)

        self._align_cube_gal(self.cube_co10_n4321,self.cube_co21_n4321,
            self.outcube_co10_n4321,self.outcube_co21_n4321,self.basebeam_n4321,
            self.imsize_n4321,self.ra_n4321,self.dec_n4321,self.chans_n4321)

    ###################
    # _align_cube_gal #
    ###################

    def _align_cube_gal(
        self,
        incube1,
        incube2,
        outcube1,
        outcube2,
        beam,
        imsize,
        ra,
        dec,
        chans,
        ):
        """
        align_cubes
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(incube1,taskname)

        # staging cubes
        self._stage_cube(incube1,outcube1+"_tmp1",beam,imsize,ra,dec,115.27120)
        self._stage_cube(incube2,outcube2+"_tmp1",beam,imsize,ra,dec,230.53800)

        # align cubes
        make_gridtemplate(outcube1+"_tmp1",outcube1+"_tmp2",imsize,ra,dec,beam)
        print(outcube1+"_tmp2")
        print(glob.glob(outcube1+"_tmp2"))
        run_imregrid(outcube2+"_tmp1",outcube1+"_tmp2",outcube2+"_tmp1p5",
            axes=[0,1])

        os.system("rm -rf " + outcube1 + "_tmp1")
        os.system("rm -rf " + outcube2 + "_tmp1")
        run_imregrid(outcube2+"_tmp1p5",outcube1+"_tmp2",outcube2+"_tmp2")
        os.system("rm -rf " + outcube2 + "_tmp1p5")

        # clip edge channels
        run_immath_one(outcube1+"_tmp2",outcube1+"_tmp3","IM0",chans,delin=True)
        run_immath_one(outcube2+"_tmp2",outcube2+"_tmp3","IM0",chans,delin=True)
        run_exportfits(outcube1+"_tmp3",outcube1+"_tmp3.fits",delin=True)
        run_exportfits(outcube2+"_tmp3",outcube2+"_tmp3.fits",delin=True)
        run_importfits(outcube1+"_tmp3.fits",outcube1+"_tmp3p5",defaultaxes=True,delin=True)
        run_importfits(outcube2+"_tmp3.fits",outcube2+"_tmp3p5",defaultaxes=True,delin=True)

        # masking
        run_immath_one(outcube1+"_tmp3p5",outcube1+"_tmp4","iif(IM0>-10000000.0,1,0)", "")
        run_immath_one(outcube2+"_tmp3p5",outcube2+"_tmp4","iif(IM0>-10000000.0,1,0)", "")
        run_immath_two(outcube1+"_tmp4",outcube2+"_tmp4",outcube1+"_combined_mask",
            "IM0*IM1",delin=True)

        run_immath_two(outcube1+"_tmp3p5",outcube1+"_combined_mask",outcube1+"_tmp4","iif(IM1>0,IM0,0)")
        run_immath_two(outcube2+"_tmp3p5",outcube1+"_combined_mask",outcube2+"_tmp4","iif(IM1>0,IM0,0)",
            delin=True)
        os.system("rm -rf " + outcube1 + "_tmp3p5")
        os.system("rm -rf " + outcube2 + "_tmp3p5")

        imhead(outcube1+"_tmp4",mode="put",hdkey="beamminor",hdvalue=str(beam)+"arcsec")
        imhead(outcube1+"_tmp4",mode="put",hdkey="beammajor",hdvalue=str(beam)+"arcsec")
        imhead(outcube2+"_tmp4",mode="put",hdkey="beamminor",hdvalue=str(beam)+"arcsec")
        imhead(outcube2+"_tmp4",mode="put",hdkey="beammajor",hdvalue=str(beam)+"arcsec")

        unitconv_Jyb_K(outcube1+"_tmp4",outcube1,115.27120,unitto="Jy/beam",delin=True)
        unitconv_Jyb_K(outcube2+"_tmp4",outcube2,230.53800,unitto="Jy/beam",delin=True)

    ###############
    # _stage_cube #
    ###############

    def _stage_cube(
        self,
        incube,
        outcube,
        beam,
        imsize,
        ra,
        dec,
        restfreq=115.27120,
        ):
        """
        _align_cube_gal
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(incube,taskname)

        run_importfits(incube,outcube+"_tmp1")
        run_roundsmooth(outcube+"_tmp1",outcube+"_tmp2",
            beam,delin=True)
        unitconv_Jyb_K(outcube+"_tmp2",outcube+"_tmp3",restfreq,delin=True)
        self._mask_fov_edges(outcube+"_tmp3",outcube+"_fovmask")
        run_immath_two(outcube+"_tmp3",outcube+"_fovmask",outcube,
            "iif(IM1>0,IM0,0)",delin=True)
        imhead(outcube,mode="put",hdkey="beamminor",hdvalue=str(beam)+"arcsec")
        imhead(outcube,mode="put",hdkey="beammajor",hdvalue=str(beam)+"arcsec")

    ###################
    # _mask_fov_edges #
    ###################

    def _mask_fov_edges(
        self,
        imagename,
        outfile,
        delin=False,
        ):
        """
        _stage_cube
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(imagename,taskname)

        expr1 = "iif(IM0>=-100000000., 1, 0)"
        run_immath_one(imagename,imagename+"_mask_fov_edges_tmp1",expr1,"")
        run_roundsmooth(imagename+"_mask_fov_edges_tmp1",
            imagename+"_mask_fov_edges_tmp2",55.0,delin=True)

        maxval = imstat(imagename+"_mask_fov_edges_tmp2")["max"][0]
        expr2 = "iif(IM0>=" + str(maxval*0.6) + ", 1, 0)"
        run_immath_one(imagename+"_mask_fov_edges_tmp2",
            outfile,expr2,"",delin=True)
        #boolean_masking(imagename+"_mask_fov_edges_tmp3",outfile,delin=True)

    ###############
    # _create_dir #
    ###############

    def _create_dir(self, this_dir):

        if self.refresh==True:
            print("## refresh " + this_dir)
            os.system("rm -rf " + this_dir)

        if not glob.glob(this_dir):
            print("## create " + this_dir)
            os.mkdir(this_dir)

        else:
            print("## not refresh " + this_dir)

    #############
    # _read_key #
    #############

    def _read_key(self, key, keyfile="fig", delimiter=",,,"):

        if keyfile=="gal":
            keyfile = self.keyfile_gal
        elif keyfile=="fig":
            keyfile = self.keyfile_fig

        keydata  = np.loadtxt(keyfile,dtype="str",delimiter=delimiter)
        keywords =\
             np.array([s.replace(" ","") for s in keydata[:,0]])
        values   = keydata[:,1]
        value    = values[np.where(keywords==key)[0][0]]

        return value

###################
# end of ToolsR21 #
###################