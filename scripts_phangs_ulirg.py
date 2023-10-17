"""
Python class for the PHANGS-(U)LIRG project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:


usage:
> import os
> from scripts_phangs_ulirg import ToolsULIRG as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_projects/galkey_ulirg.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_projects/key_phangs_ulirg.txt",
>     )
>
> # main
> tl.run_phangs_ulirg(
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

from mycasa_sampling import *
from mycasa_tasks import *
from mycasa_plots import *

##############
# ToolsULIRG #
##############
class ToolsULIRG():
    """
    Class for the PHANGS-(U)LIRG project.
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
        self.dir_ready    = self.dir_proj + self._read_key("dir_ready")
        self.dir_products = self.dir_proj + self._read_key("dir_products")
        self.dir_final    = self.dir_proj + self._read_key("dir_final")
        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_final)

    def _set_input_fits(self):
        """
        """

        this = self.dir_raw + self._read_key("mom0_150pc")
        self.list_mom0_150pc = glob.glob(this.replace("XXX","*"))

    def _set_output_fits(self):
        """
        """

    def _set_input_param(self):
        """
        """

    def _set_output_txt_png(self):
        """
        """

    ####################
    # run_phangs_ulirg #
    ####################

    def run_phangs_ulirg(
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
        if plot_showcase==True:
            self.showcase()

    ############
    # showcase #
    ############

    def showcase(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.list_mom0_150pc[0],taskname)

        print(len(self.list_mom0_150pc))

    #################
    # _one_showcase #
    #################

    def _one_showcase(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.list_mom0_150pc[0],taskname)

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        levels_cont1 = [0.05, 0.1, 0.2, 0.4, 0.8, 0.96]
        width_cont1  = [1.0]
        set_bg_color = "white" # cm.rainbow(0)

        # plot
        myfig_fits2png(
            imcolor=imcolor,
            outfile=outfile,
            imcontour1=imcontour1,
            imsize_as=self.imsize,
            ra_cnt=self.ra_agn_str,
            dec_cnt=self.dec_agn_str,
            levels_cont1=levels_cont1,
            width_cont1=width_cont1,
            set_title=set_title,
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar=label_cbar,
            clim=clim,
            set_bg_color=set_bg_color,
            numann="13co",
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

#####################
# end of ToolsULIRG #
#####################