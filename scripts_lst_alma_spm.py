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
2022-03-19   CASA tools sm and vp for heterogenous array sim
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np
from scipy.stats import gaussian_kde

from mycasa_tasks import *
from mycasa_plots import *
from mycasa_simobs import *

##################
# ToolsLSTSpMSim #
##################
class ToolsLSTSpMSim():
    """
    Class for the LST white paper 2022 project.
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
            self.modname = "ToolsLSTSim."
            self._set_dir()            # directories
            self._set_input_fits()     # input maps
            self._set_output_fits()    # output maps
            self._set_input_param()    # input parameters
            self._set_output_txt_png() # output txt and png

    def _set_dir(self):
        """
        done!
        """

        self.dir_proj     = self._read_key("dir_proj")
        self.dir_raw      = self.dir_proj + self._read_key("dir_raw")
        self.dir_ready    = self.dir_proj + self._read_key("dir_ready")
        self.dir_other    = self.dir_proj + self._read_key("dir_other")
        self.dir_products = self.dir_proj + self._read_key("dir_products")
        self.dir_final    = self.dir_proj + self._read_key("dir_final")

        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_final)

        # simobserve
        self.project_gmaur = self._read_key("project_gmaur")
        self.config_c1     = self.dir_keyfile + self._read_key("config_c1")
        self.config_c9     = self.dir_keyfile + self._read_key("config_c9")
        self.config_c9_lst = self.dir_keyfile + self._read_key("config_c9_lst")
        self.config_c10    = self.dir_keyfile + self._read_key("config_c10")
        self.config_7m     = self.dir_keyfile + self._read_key("config_7m")
        self.config_lst    = self.dir_keyfile + self._read_key("config_lst")

        # phangs-alma pipeline
        self.dir_pipeline = self._read_key("dir_pipeline")

    def _set_input_fits(self):
        """
        """

        self.gmaur_template_file = self._read_key("gmaur_template_file")
        self.gmaur_template_mask = self._read_key("gmaur_template_mask")

    def _set_output_fits(self):
        """
        """

        self.gmaur_template_noshrunk = self.project_gmaur + "_template_shrunk.image"
        self.gmaur_template_shurnk2  = self.project_gmaur + "gmaur_template_shurnk2.image"
        self.gmaur_template_shurnk4  = self.project_gmaur + "gmaur_template_shurnk4.image"

    def _set_input_param(self):
        """
        """

    def _set_output_txt_png(self):
        """
        """

    ####################
    # run_sim_lst_alma #
    ####################

    def run_sim_lst_alma(
        self,
        ############
        # GMAursim #
        ############
        # prepare
        tinteg_GMaursim      = 24,    # 12m total observing time
        observed_freq        = 230.0, # GHz, determine LST and TP beam sizes
        do_template_GMaursim = False, # create template simobserve
        do_simint_GMaursim   = False, # sim C-10 at observed_freq
        ):
        """
        This method runs all the methods which will create figures in the white paper.
        """

        ###########################
        # set GMAursim parameters #
        ###########################
        # observed frequency
        self.observed_freq = observed_freq
        self.incenter      = str(observed_freq)+"GHz"

        # GMaursim_12m from tinteg_GMaursim
        tinteg_12m      = str(float(tinteg_GMaursim))+"h"
        tintegstr_12m   = tinteg_12m.replace(".","p")
        this_target     = self.project_gmaur+"_"+tintegstr_12m
        this_target_lst = self.project_gmaur+"_lst_"+tintegstr_12m

        if do_template_GMaursim==True:
            self.prepare_template_gmaursim()

        if do_simint_GMaursim==True:
            self.simobs_gmaursim(tinteg_12m,tintegstr_12m)

    ###################
    # simobs_gmaursim #
    ###################

    def simobs_gmaursim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.gmaur_template_noshrunk,taskname)

        run_simobserve(
            working_dir = self.dir_ready,
            template    = self.gmaur_template_noshrunk,
            antennalist = self.config_c9,
            project     = self.project_gmaur+"_12m_"+totaltimetint,
            totaltime   = totaltime,
            incenter    = self.incenter,
            pointingspacing = "1arcsec",
            )

        run_simobserve(
            working_dir = self.dir_ready,
            template    = self.gmaur_template_noshrunk,
            antennalist = self.config_c9_lst,
            project     = self.project_gmaur+"_12m_lst_"+totaltimetint,
            totaltime   = totaltime,
            incenter    = self.incenter,
            pointingspacing = "1arcsec",
            )

    #############################
    # prepare_template_gmaursim #
    #############################

    def prepare_template_gmaursim(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name

        gen_cont(
            self.dir_raw,
            self.gmaur_template_file,
            self.gmaur_template_mask,
            self.dir_ready,
            self.gmaur_template_noshrunk,
            self.gmaur_template_shurnk2,
            self.gmaur_template_shurnk4,
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

#########################
# end of ToolsLSTSpMSim #
#########################