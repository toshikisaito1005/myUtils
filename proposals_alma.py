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

                # spectral scan setup
                self.line_key = self.dir_proj + "scripts/keys/key_lines.txt"
                l = self._read_key("b3_spw_setup")
                self.b3_spw_setup = [float(s) for s in l.split(",")]
                l = self._read_key("b6_spw_setup")
                self.b6_spw_setup = [float(s) for s in l.split(",")]

                # ngc1068
                self.z = float(self._read_key("z"))

                # output png
                self.png_specscan_b3 = self.dir_products + self._read_key("png_specscan_b3")
                self.png_specscan_b6 = self.dir_products + self._read_key("png_specscan_b6")
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
        with open(self.archive_csv) as f:
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

        # prepare for plot: b3
        list_spw,list_color = [],[]
        for i in range(len(data_b3[:,1])):
            this_freq  = data_b3[i,1]
            this_array = data_b3[i,5]
            this_spws  = this_freq.split(" U ")

            if this_array=="12m":
                this_color = "lightgrey"
            else:
                this_color = "tomato"

            for j in range(len(this_spws)):
                this_spw = this_spws[j].split(",")[0].lstrip("[")
                this_spw = this_spw.rstrip("GHz").split("..")
                this_spw = [float(this_spw[0])*(1+self.z),float(this_spw[1])*(1+self.z)]
                list_spw.append(this_spw)
                list_color.append(this_color)

        list_b3data = np.c_[list_spw,list_color]
        list_b3data = list_b3data[np.argsort(list_b3data[:, 0].astype(np.float64))]

        # plot
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(bottom=0.15,left=0.07,right=0.98,top=0.90)
        gs = gridspec.GridSpec(nrows=3, ncols=1)
        ax1 = plt.subplot(gs[0:1,0:1])
        ax2 = plt.subplot(gs[1:2,0:1])
        ax3 = plt.subplot(gs[2:3,0:1])

        # ax setup
        myax_set(ax2,xlim=[84,116],ylim=[0,100],labelbottom=False,labelleft=False)
        ax2.gca().spines['right'].set_visible(False)

        myax_set(ax3,xlim=[84,116],ylim=[0,len(self.b3_spw_setup)+2],labelbottom=False,labelleft=False)

        # ax2
        for i in range(len(list_b3data)):
            x = [float(list_b3data[i][0]),float(list_b3data[i][1])]
            ax2.plot(x, [i+1,i+1], "-", color=list_b3data[i][2],lw=2)

        # ax3
        for j in range(len(self.b3_spw_setup)):
            x = [self.b3_spw_setup[j]-1.875/2.0, self.b3_spw_setup[j]+1.875/2.0]
            y = [j+1,j+1]
            ax3.plot(x,y,color="deepskyblue",alpha=0.3,lw=3)

        plt.savefig(self.png_specscan_b3, dpi=self.fig_dpi)

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
