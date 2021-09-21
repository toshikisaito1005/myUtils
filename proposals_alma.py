"""
Class for ALMA proposals.

history:
2021-09-21   wrote by TS
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob, csv
import numpy as np

from mycasa_tasks import *
from mycasa_sampling import *
from mycasa_plots import *

myia = aU.createCasaTool(iatool)

###########################
### ToolsDense
###########################
class ProposalsALMA():
    """
    Class for ALMA proposals.
    """

    ############
    # __init__ #
    ############

    def __init__(
        self,
        keyfile_fig = None,
        keyfile_gal = None,
        refresh = False,
        delete_inter = True,
        ):
        # initialize keys
        self.keyfile_gal = keyfile_gal
        self.keyfile_fig = keyfile_fig

        # intialize task
        self.refresh      = refresh
        self.delete_inter = delete_inter
        self.taskname     = None

        self.fig_dpi = 200

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ProposalsALMA."

            # get alma cycle
            self.cycle = self._read_key("this_cycle")
            
            # get directories
            self.dir_proj = self._read_key("dir_proj")
            dir_raw = self.dir_proj + self._read_key("dir_raw") + self.cycle + "/"
            self.dir_ready = self.dir_proj + self._read_key("dir_ready") + self.cycle + "/"
            self.dir_products = self.dir_proj + self._read_key("dir_products") + self.cycle + "/"
            self.dir_final = self.dir_proj + self._read_key("dir_final") + self.cycle + "/"
            self._create_dir(self.dir_ready)
            self._create_dir(self.dir_products)
            self._create_dir(self.dir_final)

            # cycle 8p5
            if self.cycle=="cycle08p5":
                # input data
                self.image_co10_12m7m = dir_raw + self._read_key("image_co10_12m7m")
                self.image_co10_12m = dir_raw + self._read_key("image_co10_12m")
                self.archive_csv = dir_raw + self._read_key("archive_csv")

                # lines
                self.line_key = self.dir_proj + "scripts/keys/key_lines.txt"

                # ngc1068
                self.z = float(self._read_key("z"))

                # output png
                self.png_missingflux = self.dir_products + self._read_key("png_missingflux")

    #################
    # run_cycle_8p5 #
    #################

    def run_cycle_8p5(
        self,
        plot_spw_setup = False,
        ):

        if plot_spw_setup==True:
            self.plot_spw_setup()

    ##################
    # plot_spw_setup #
    ##################

    def plot_spw_setup(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.archive_csv,taskname)

        # read csv
        with open(this_csv) as f:
            reader = csv.reader(f)
            l = [row for row in reader]

        l = np.array(l)

        ### get info
        header    = l[0]
        project   = l[:,0]
        galname   = l[:,1]
        band      = l[:,4]
        freq_info = l[:,6]
        ang_res   = l[:,9]
        pi_name   = l[:,23]
        array     = l[:,11]

        # exclude ToO projects
        mask = []
        for i in range(len(project)):
            this_project = project[i]
            if ".S" in this_project:
                mask.append(i)

        project   = project[mask]
        galname   = galname[mask]
        band      = band[mask]
        freq_info = freq_info[mask]
        ang_res   = ang_res[mask]
        pi_name   = np.array([s.split(",")[0] for s in pi_name[mask]])
        array     = array[mask]

        # get data
        b3,b6 = [],[]
        for i in range(len(band)):
            this_band = band[i]
            if this_band=="3":
                b3.append(i)
            elif this_band=="6":
                b6.append(i)

        data_b3 = np.c_[project[b3], freq_info[b3], ang_res[b3], pi_name[b3], galname[b3], array[b3]]
        data_b6 = np.c_[project[b6], freq_info[b6], ang_res[b6], pi_name[b6], galname[b6], array[b6]]

        # plot

    ###############
    # _create_dir #
    ###############

    def _create_dir(self, this_dir):

        if self.refresh==True:
            print("## refresh " + this_dir)
            os.system("rm -rf " + this_dir)

        if not glob.glob(this_dir):
            print("## create " + this_dir)
            os.makedirs(this_dir)

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
