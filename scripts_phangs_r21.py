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
2022-07-28   
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
# ToolsR21 #
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

    def _set_input_param(self):
        """
        """

        self.ra_n0628  = self._read_key("ra_n0628", "gal").split("deg")[0]
        self.ra_n3627  = self._read_key("ra_n3627", "gal").split("deg")[0]
        self.ra_n4254  = self._read_key("ra_n4254", "gal").split("deg")[0]
        self.ra_n4321  = self._read_key("ra_n4321", "gal").split("deg")[0]

        self.dec_n0628 = self._read_key("dec_n0628", "gal").split("deg")[0]
        self.dec_n3627 = self._read_key("dec_n3627", "gal").split("deg")[0]
        self.dec_n4254 = self._read_key("dec_n4254", "gal").split("deg")[0]
        self.dec_n4321 = self._read_key("dec_n4321", "gal").split("deg")[0]

        self.basebeam_n0628 = float(self._read_key("basebeam_n0628"))
        self.basebeam_n3627 = float(self._read_key("basebeam_n3627"))
        self.basebeam_n4254 = float(self._read_key("basebeam_n4254"))
        self.basebeam_n4321 = float(self._read_key("basebeam_n4321"))

        self.imsize_n0628 = float(self._read_key("imsize_n0628"))
        self.imsize_n3627 = float(self._read_key("imsize_n3627"))
        self.imsize_n4254 = float(self._read_key("imsize_n4254"))
        self.imsize_n4321 = float(self._read_key("imsize_n4321"))

        self.chans_n0628 = self._read_key("chans_n0628")
        self.chans_n3627 = self._read_key("chans_n3627")
        self.chans_n4254 = self._read_key("chans_n4254")
        self.chans_n4321 = self._read_key("chans_n4321")

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
        do_all        = False,
        # analysis
        do_prepare    = False,
        # plot figures in paper
        plot_showcase = False,
        # supplement
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        if do_all==True:
            do_prepare = True

        # analysis
        if do_prepare==True:
            self.align_cubes()

        # plot figures in paper
        #if plot_showcase==True:
        #    self.showcase()

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

        self._align_cube_gal(
            self.cube_co10_n0628,
            self.cube_co21_n0628,
            self.outcube_co10_n0628,
            self.outcube_co21_n0628,
            self.basebeam_n0628,
            self.imsize_n0628,
            self.ra_n0628,
            self.dec_n0628,
            )

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
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(incube1,taskname)

        # staging cubes
        self._stage_cube(incube1,outcube1+"_tmp1",beam,imsize,ra,dec)
        self._stage_cube(incube2,outcube2+"_tmp1",beam,imsize,ra,dec)

        # align cubes
        imrebin2(outcube1+"_tmp1",outcube1+"_tmp2",imsize,ra,dec,delin=True)
        run_imregrid(outcube2+"_tmp1",outcube1+"_tmp2",outcube2+"_tmp1p5",
            axes=[0,1])
        os.system("rm -rf " + outcube2 + "_tmp1")
        run_imregrid(outcube2+"_tmp1p5",outcube1+"_tmp2",outcube2+"_tmp2")
        os.system("rm -rf " + outcube2 + "_tmp1p5")

        # from line 801 of scripts_phangs_r21_tasks.py

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
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(incube,taskname)

        run_importfits(incube,outcube+"_tmp1")
        run_roundsmooth(outcube+"_tmp1",outcube+"_tmp2",
            beam,delin=True)
        unitconv_Jyb_K(outcube+"_tmp2",outcube+"_tmp3",115.27120,delin=True)
        self._mask_fov_edges(outcube+"_tmp3",outcube+"_fovmask")
        run_immath_two(outcube+"_tmp3",outcube+"_fovmask",outcube+"_tmp4",
            "iif(IM1>0,IM0,0)",delin=True)
        imhead(outcube+"_tmp4",mode="put",hdkey="beamminor",hdvalue=str(beam)+"arcsec")
        imhead(outcube+"_tmp4",mode="put",hdkey="beammajor",hdvalue=str(beam)+"arcsec")

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