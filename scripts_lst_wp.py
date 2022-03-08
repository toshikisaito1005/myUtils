"""
Python class for the NGC 1068 PCA project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:

usage:
> import os
> from scripts_sim_lst_wp import ToolsLSTSim as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_sim_lst_wp/key_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_sim_lst_wp/key_figures.txt",
>     )
>
> # main
> tl.run_sim_lst_alma(
>     # analysis
>     do_prepare             = True,
>     )
>
> os.system("rm -rf *.last")

white paper drafts:
Date         Filename                To
2022-??-??

history:
2022-03-08   created
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_tasks import *
from mycasa_plots import *
from mycasa_simobs import *

###############
# ToolsLSTSim #
###############
class ToolsLSTSim():
    """
    Class for the L S white paper 2022 project.
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
        self.dir_keyfile = "/".join(self.keyfile_gal.split("/")[:-1]) + "/"

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
            self.modname = "ToolsPCA."
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
        self.dir_ready    = self.dir_proj + self._read_key("dir_ready")
        self.dir_other    = self.dir_proj + self._read_key("dir_other")
        self.dir_products = self.dir_proj + self._read_key("dir_products")
        self.dir_final    = self.dir_proj + self._read_key("dir_final")
        self.dir_final    = self.dir_proj + self._read_key("dir_final")

        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_final)

        # simobserve
        self.project_sim  = self.dir_ready + "ngc3059sim"
        self.config_12m   = self.dir_keyfile + "alma.cycle7.1.cfg"
        self.config_7m    = self.dir_keyfile + "aca.cycle7.cfg"

    def _set_input_fits(self):
        """
        """

        self.template_file = "ngc3059_12m+7m+tp_co21.fits"
        self.template_mask = "ngc3059_12m+7m+tp_co21_strictmask.fits"

    def _set_output_fits(self):
        """
        """

        self.template_in_jypix        = "ngc3059_template_jypixel.image"
        self.template_clipped         = "ngc3059_template_clipped.image"
        self.template_mask_imported   = "ngc3059_template_mask.image"
        self.template_rotated         = "ngc3059_template_rotated.image"
        self.template_shrunk          = "ngc3059_template_shrunk.image"
        self.template_fullspec        = "ngc3059_template_fullspec.image"
        self.template_fullspec_div3   = "ngc3059_template_fullspec_div3.image"
        self.template_fullspec_div10  = "ngc3059_template_fullspec_div10.image"
        self.template_fullspec_div30  = "ngc3059_template_fullspec_div30.image"
        self.template_fullspec_div100 = "ngc3059_template_fullspec_div100.image"
        self.template_withcont        = "ngc3059_template_withcont.image"
        self.template_withcont_div3   = "ngc3059_template_withcont_div3.image"
        self.template_withcont_div10  = "ngc3059_template_withcont_div10.image"
        self.template_withcont_div30  = "ngc3059_template_withcont_div30.image"
        self.template_withcont_div100 = "ngc3059_template_withcont_div100.image"
        self.sdnoise_image            = "ngc3059_singledish_noise_nocont.image"
        self.sdimage_fullspec         = "ngc3059_singledish_nocont.image"
        self.sdimage_div3             = "ngc3059_singledish_div3.image"
        self.sdimage_div10            = "ngc3059_singledish_div10.image"
        self.sdimage_div30            = "ngc3059_singledish_div30.image"
        self.sdimage_div100           = "ngc3059_singledish_div100.image"

        self.sdnoise_image            = "ngc3059_singledish_noise_nocont.image"
        self.sdimage_fullspec         = "ngc3059_singledish_nocont.image"
        self.sdimage_div3             = "ngc3059_singledish_div3.image"
        self.sdimage_div10            = "ngc3059_singledish_div10.image"
        self.sdimage_div30            = "ngc3059_singledish_div30.image"
        self.sdimage_div100           = "ngc3059_singledish_div100.image"

    def _set_input_param(self):
        """
        """

        # sim properties
        self.singledish_noise = 0.102 # Jy/beam at final res
        self.singledish_res   = "28.37arcsec" # resolution
        self.image_roration   = "23deg"

        # ngc1068 properties
        self.ra_agn    = float(self._read_key("ra_agn", "gal").split("deg")[0])
        self.dec_agn   = float(self._read_key("dec_agn", "gal").split("deg")[0])
        self.scale_pc  = float(self._read_key("scale", "gal"))
        self.scale_kpc = self.scale_pc / 1000.

        self.beam      = 2.14859173174056 # 150pc in arcsec
        self.snr_mom   = 4.0
        self.r_cnd     = 3.0 * self.scale_pc / 1000. # kpc
        self.r_cnd_as  = 3.0
        self.r_sbr     = 10.0 * self.scale_pc / 1000. # kpc
        self.r_sbr_as  = 10.0
        self.gridsize  = 27 # int(np.ceil(self.r_sbr_as*2/self.beam))

    def _set_output_txt_png(self):
        """
        """

    ####################
    # run_sim_lst_alma #
    ####################

    def run_sim_lst_alma(
        self,
        # analysis
        do_template = False,
        do_simint   = False,
        ):
        """
        This method runs all the methods which will create figures in the white paper.
        """

        # analysis
        if do_template==True:
            self.prepare_do_template()

        if do_simint==True:
            self.simint()

    ##########
    # simint #
    ##########

    def simint(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.template_file,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.template_fullspec,
            antennalist=self.config_12m,
            self.project_sim,
            project=totaltime="1.5h",
            )

    #######################
    # prepare_do_template #
    #######################

    def prepare_do_template(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.template_file,taskname)

        gen_cube(
            template_dir=self.dir_raw,
            template_file=self.template_file,
            template_mask=self.template_mask,
            working_dir=self.dir_ready,
            template_in_jypix=self.template_in_jypix,
            template_clipped=self.template_clipped,
            template_mask_imported=self.template_mask_imported,
            template_rotated=self.template_rotated,
            template_shrunk=self.template_shrunk,
            template_fullspec=self.template_fullspec,
            template_fullspec_div3=self.template_fullspec_div3,
            template_fullspec_div10=self.template_fullspec_div10,
            template_fullspec_div30=self.template_fullspec_div30,
            template_fullspec_div100=self.template_fullspec_div100,
            template_withcont=self.template_withcont,
            template_withcont_div3=self.template_withcont_div3,
            template_withcont_div10=self.template_withcont_div10,
            template_withcont_div30=self.template_withcont_div30,
            template_withcont_div100=self.template_withcont_div100,
            sdnoise_image=self.sdnoise_image,
            sdimage_fullspec=self.sdimage_fullspec,
            sdimage_div3=self.sdimage_div3,
            sdimage_div10=self.sdimage_div10,
            sdimage_div30=self.sdimage_div30,
            sdimage_div100=self.sdimage_div100,
            pa=self.image_roration, # rotation angle
            singledish_res=self.singledish_res, # 12m TP resolution
            singledish_noise=self.singledish_noise, # Jy/beam at final res
            )

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

######################
# end of ToolsLSTSim #
######################