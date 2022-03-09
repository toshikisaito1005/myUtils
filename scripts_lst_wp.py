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
        self.project_sim  = self._read_key("project_sim")
        self.config_c1    = self.dir_keyfile + self._read_key("config_c1")
        self.config_c10   = self.dir_keyfile + self._read_key("config_c10")
        self.config_7m    = self.dir_keyfile + self._read_key("config_7m")

    def _set_input_fits(self):
        """
        """

        self.template_file = self._read_key("template_file")
        self.template_mask = self._read_key("template_mask")

    def _set_output_fits(self):
        """
        """

        self.tempgal = "ngc1097"

        self.template_in_jypix        = self.tempgal + "_template_jypixel.image"
        self.template_clipped         = self.tempgal + "_template_clipped.image"
        self.template_mask_imported   = self.tempgal + "_template_mask.image"
        self.template_rotated         = self.tempgal + "_template_rotated.image"
        self.template_shrunk          = self.tempgal + "_template_shrunk.image"
        self.template_fullspec        = self.tempgal + "_template_fullspec.image"
        self.template_fullspec_div3   = self.tempgal + "_template_fullspec_div3.image"
        self.template_fullspec_div5   = self.tempgal + "_template_fullspec_div5.image"
        self.template_fullspec_div10  = self.tempgal + "_template_fullspec_div10.image"
        self.template_fullspec_div30  = self.tempgal + "_template_fullspec_div30.image"
        self.template_fullspec_div100 = self.tempgal + "_template_fullspec_div100.image"
        self.template_withcont        = self.tempgal + "_template_withcont.image"
        self.template_withcont_div3   = self.tempgal + "_template_withcont_div3.image"
        self.template_withcont_div5   = self.tempgal + "_template_withcont_div5.image"
        self.template_withcont_div10  = self.tempgal + "_template_withcont_div10.image"
        self.template_withcont_div30  = self.tempgal + "_template_withcont_div30.image"
        self.template_withcont_div100 = self.tempgal + "_template_withcont_div100.image"
        self.sdnoise_image            = self.tempgal + "_singledish_noise_nocont.image"
        self.sdimage_fullspec         = self.tempgal + "_singledish_nocont.image"
        self.sdimage_div3             = self.tempgal + "_singledish_div3.image"
        self.sdimage_div5             = self.tempgal + "_singledish_div5.image"
        self.sdimage_div10            = self.tempgal + "_singledish_div10.image"
        self.sdimage_div30            = self.tempgal + "_singledish_div30.image"
        self.sdimage_div100           = self.tempgal + "_singledish_div100.image"

        self.sdnoise_image            = self.tempgal + "_singledish_noise_nocont.image"
        self.sdimage_fullspec         = self.tempgal + "_singledish_nocont.image"
        self.sdimage_div3             = self.tempgal + "_singledish_div3.image"
        self.sdimage_div5             = self.tempgal + "_singledish_div5.image"
        self.sdimage_div10            = self.tempgal + "_singledish_div10.image"
        self.sdimage_div30            = self.tempgal + "_singledish_div30.image"
        self.sdimage_div100           = self.tempgal + "_singledish_div100.image"

    def _set_input_param(self):
        """
        """

        # sim properties
        self.singledish_noise = 0.102 # Jy/beam at final res
        self.singledish_res   = "28.37arcsec" # resolution
        self.image_roration   = "23deg"
        self.incenter         = "492.16065100GHz" #CI 3P1-3P0

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

    def _set_output_txt_png(self):
        """
        """

        self.outpng_config_12m = self.dir_products + self._read_key("outpng_config_12m")

    ####################
    # run_sim_lst_alma #
    ####################

    def run_sim_lst_alma(
        self,
        # analysis
        do_template   = False, # create template cube for simobserve
        do_simint     = False, # sim ALMA-only
        do_simsynergy = False, # sim LST+ALMA
        do_imaging    = False, # imaging sim ms
        # plot
        plot_config   = False,
        ):
        """
        This method runs all the methods which will create figures in the white paper.
        """

        # analysis
        if do_template==True:
            self.prepare_do_template()

        if do_simint==True:
            self.simint()

        if do_imaging==True:
            self.phangs_pipeline_imaging()

        # plot
        if plot_config==True:
            self.plot_config()

    ###############
    # plot_config #
    ###############

    def plot_config(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.config_c10,taskname)

        # get data
        data  = np.loadtxt(self.config_c10,"str")
        x_12m = data[:,0].astype(np.float32) / 1000.
        y_12m = data[:,1].astype(np.float32) / 1000.

        data  = np.loadtxt(self.config_7m,"str")
        x_7m  = data[:,0].astype(np.float32) / 1000.
        y_7m  = data[:,1].astype(np.float32) / 1000.

        # plot
        ad    = [0.215,0.83,0.10,0.90]
        xlim  = [-10,10]
        ylim  = [-10,10]
        title = "Antenna positions"
        xlabel = "East-West (km)"
        ylabel = "North-South (km)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(x_12m, y_12m, color="black", lw=0, s=100)
        ax1.scatter(x_7m, y_7m, color="deepskyblue", lw=0, s=100)
        ax1.scatter(800, 800, color="tomato", marker="*", lw=0, s=150)
        ax1.scatter(800, -800, color="tomato", marker="*", lw=0, s=150)
        ax1.scatter(800, 0, color="tomato", marker="*", lw=0, s=150)

        # text
        ax1.text(0.05,0.92, "ALMA - 12-m array", color="black", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "ALMA - ACA", color="deepskyblue", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.82, "LSTsim", color="tomato", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_config_12m)
        plt.savefig(self.outpng_config_12m, dpi=self.fig_dpi)

    ###########################
    # phangs_pipeline_imaging #
    ###########################

    def phangs_pipeline_imaging(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.template_fullspec,taskname)

        # ms
        sim_12m_ms_orig = self.dir_ready + "ms/" + self.project_sim + "." + self.config_c1 + ".noisy.ms"
        sim_7m_ms_orig  = self.dir_ready + "ms/" + self.project_sim + "." + self.config_7m + ".noisy.ms"

        # prepare dir
        dir_cleanmask = self.dir_ready + "outputs/cleanmasks/"
        if not dir_cleanmask:
            os.mkdir(dir_cleanmask)

        dir_singledish = self.dir_ready + "outputs/singledish/"
        if not dir_singledish:
            os.mkdir(dir_singledish)

        # set piepline
        relpath_master_key = "/keys_sim/master_key.txt"

        pipepath = os.environ.get('PHANGSPIPE')
        if pipepath is not None:
            sys.path.append(os.environ.get('PHANGSPIPE'))
        else:
            sys.path.append(os.getcwd())

        # Check casa environment by importing CASA-only packages
        try:
            import taskinit
        except ImportError:
            print('Please run this script inside CASA!')
            sys.exit()

        # Set the logging
        from phangsPipeline import phangsLogger as pl
        reload(pl)
        pl.setup_logger(level='DEBUG', logfile=None)
        # Imports

        from phangsPipeline import handlerKeys as kh
        from phangsPipeline import handlerVis as uvh
        from phangsPipeline import handlerImaging as imh
        from phangsPipeline import handlerPostprocess as pph

        # Reloads for debugging
        reload(kh)
        reload(uvh)
        reload(imh)
        reload(pph)

        # Initialize key handler
        this_kh  = kh.KeyHandler(master_key = master_key)
        this_uvh = uvh.VisHandler(key_handler = this_kh)
        this_imh = imh.ImagingHandler(key_handler = this_kh)
        this_pph = pph.PostProcessHandler(key_handler= this_kh)
        dry_run_key = False
        this_uvh.set_dry_run(dry_run_key)
        this_imh.set_dry_run(dry_run_key)
        this_pph.set_dry_run(dry_run_key)

        # set handlers
        target = [self.tempgal]
        array  = ["12m"]
        line   = ["ci10"]
        for this_hander in [this_uvh,this_imh,this_pph]:
            this_hander.set_targets(only=target)
            this_hander.set_line_products(only=line)
            this_hander.set_no_cont_products(True)
            this_hander.set_no_line_products(True)
            this_hander.set_interf_configs(only=array)

        # run piepline
        this_uvh.loop_stage_uvdata(\
                do_copy           = True,
                do_remove_staging = True,
                do_contsub        = False,
                do_extract_line   = True,
                do_extract_cont   = False,
                overwrite         = False,
                )
        this_imh.loop_imaging(\
                do_dirty_image          = True,
                do_revert_to_dirty      = False,
                do_read_clean_mask      = False,
                do_multiscale_clean     = False,
                do_revert_to_multiscale = False,
                do_singlescale_mask     = False,
                do_singlescale_clean    = True,
                do_export_to_fits       = False,
                extra_ext_in            = '',
                extra_ext_out           = '',
                )
        this_pph.loop_postprocess(\
                do_prep               = True,
                do_feather            = False,
                do_mosaic             = True,
                do_cleanup            = True,
                do_summarize          = True,
                # feather_apod          = True,
                feather_noapod        = True,
                # feather_before_mosaic = False,
                # feather_after_mosaic  = False,
                )

    ##########
    # simint #
    ##########

    def simint(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.template_fullspec,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.template_fullspec_div3,
            antennalist=self.config_c1,
            project=self.project_sim+"_12m",
            totaltime="1.5h",
            incenter=self.incenter,
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
            template_fullspec_div5=self.template_fullspec_div5,
            template_fullspec_div10=self.template_fullspec_div10,
            template_fullspec_div30=self.template_fullspec_div30,
            template_fullspec_div100=self.template_fullspec_div100,
            template_withcont=self.template_withcont,
            template_withcont_div3=self.template_withcont_div3,
            template_withcont_div5=self.template_withcont_div5,
            template_withcont_div10=self.template_withcont_div10,
            template_withcont_div30=self.template_withcont_div30,
            template_withcont_div100=self.template_withcont_div100,
            sdnoise_image=self.sdnoise_image,
            sdimage_fullspec=self.sdimage_fullspec,
            sdimage_div3=self.sdimage_div3,
            sdimage_div5=self.sdimage_div5,
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