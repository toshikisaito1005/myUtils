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

        self.fig_dpi = 500

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
                self.image_cs21 = dir_raw + self._read_key("image_cs21")
                self.archive_csv = dir_raw + self._read_key("archive_csv")
                self.txt_fov_b3 = self._read_key("fov_b3")
                self.txt_fov_b6 = self._read_key("fov_b6")

                # spectral scan setup
                self.line_key = self.dir_proj + "scripts/keys/key_lines.txt"
                l = self._read_key("b3_spw_setup")
                self.b3_spw_setup = [float(s) for s in l.split(",")]
                l = self._read_key("b6_spw_setup")
                self.b6_spw_setup = [float(s) for s in l.split(",")]

                # ngc1068
                self.z       = float(self._read_key("z"))
                self.scale   = float(self._read_key("scale"))
                self.ra_agn  = self._read_key("ra_agn")
                self.dec_agn = self._read_key("dec_agn")

                # output fits
                self.outfits_missingflux = self.dir_ready + self._read_key("outfits_missingflux")
                self.outfits_co10 = self.dir_ready + self._read_key("outfits_co10")

                # output png
                self.png_specscan_b3 = self.dir_products + self._read_key("png_specscan_b3")
                self.png_specscan_b6 = self.dir_products + self._read_key("png_specscan_b6")

                self.png_missingflux = self.dir_products + self._read_key("png_missingflux")
                self.imsize_as       = float(self._read_key("imsize_as"))

                self.png_histogram   = self.dir_products + self._read_key("png_histogram")

                # final products
                self.final_specscan = self.dir_final + self._read_key("final_specscan")
                self.box_specscan = self._read_key("box_specscan")

                self.final_missingflux = self.dir_final + self._read_key("final_missingflux")
                self.box_missingflux = self._read_key("box_missingflux")
                self.box_missingflux2 = self._read_key("box_missingflux2")

    ############################################################################################
    ############################################################################################
    ##############################                                ##############################
    ############################## ALMA cycle 8 supplemental call ##############################
    ##############################                                ##############################
    ############################################################################################
    ############################################################################################

    def run_cycle_8p5(
        self,
        plot_spw_setup   = False,
        plot_missingflux = False,
        combine_figures  = False,
        ):

        if plot_spw_setup==True:
            self.c8p5_plot_spw_setup_b3()
            self.c8p5_plot_spw_setup_b6()

        if plot_missingflux==True:
            self.c8p5_plot_missingflux()

        if combine_figures==True:
            self.c8p5_create_figure_spws()
            self.c8p5_create_figure_missingflux()
            self.c8p5_fov_with_map()

    #####################
    # c8p5_fov_with_map #
    #####################

    def c8p5_fov_with_map(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_cs21,taskname)

    ##################################
    # c8p5_create_figure_missingflux #
    ##################################

    def c8p5_create_figure_missingflux(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.png_missingflux,taskname)

        combine_two_png(
        self.png_missingflux,
        self.png_histogram,
        self.final_missingflux,
        self.box_missingflux,
        self.box_missingflux2,
        )

    #########################
    # c8p5_plot_missingflux #
    #########################

    def c8p5_plot_missingflux(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_co10_12m7m,taskname)

        run_importfits(self.image_co10_12m7m,self.image_co10_12m7m+"_tmp1",defaultaxes=True,
            defaultaxesvalues=["RA","Dec","1GHz","Stokes"])
        run_importfits(self.image_co10_12m,self.image_co10_12m+"_tmp1",defaultaxes=True,
            defaultaxesvalues=["RA","Dec","1GHz","Stokes"])
        run_exportfits(self.image_co10_12m+"_tmp1",self.outfits_co10,True,True,False)

        run_imregrid(self.image_co10_12m7m+"_tmp1",self.image_co10_12m+"_tmp1",
            self.image_co10_12m7m+"_tmp2",delin=True)

        expr = "iif(IM1>0,(IM0-IM1)/IM0*100,0)"
        run_immath_two(self.image_co10_12m7m+"_tmp2",self.image_co10_12m+"_tmp1",
            self.outfits_missingflux+"_tmp1",expr)

        signal_masking(self.outfits_missingflux+"_tmp1",self.outfits_missingflux+"_tmp2",0)

        expr = "iif(IM1>0,IM0,0)"
        run_immath_two(self.outfits_missingflux+"_tmp1",self.outfits_missingflux+"_tmp2",
            self.outfits_missingflux+"_tmp3",expr,delin=True)

        signal_masking(self.image_co10_12m7m+"_tmp2",self.image_co10_12m7m+"_tmp3",0,True)
        signal_masking(self.image_co10_12m+"_tmp1",self.image_co10_12m+"_tmp2",0,True)
        expr = "iif(IM0>0,IM0-IM1,0)"
        run_immath_two(self.image_co10_12m7m+"_tmp3",self.image_co10_12m+"_tmp2",
            self.outfits_missingflux+"_tmp4",expr,delin=True)

        expr = "iif(IM1>0,100,IM0)"
        run_immath_two(self.outfits_missingflux+"_tmp3",self.outfits_missingflux+"_tmp4",
            self.outfits_missingflux+"_tmp5",expr,delin=True)

        imhead(self.outfits_missingflux+"_tmp5",mode="put",hdkey="beammajor",hdvalue="0.8arcsec")
        imhead(self.outfits_missingflux+"_tmp5",mode="put",hdkey="beamminor",hdvalue="0.8arcsec")

        run_exportfits(self.outfits_missingflux+"_tmp5",self.outfits_missingflux,True,True,True)

        # plot map
        scalebar = 500 / self.scale

        myfig_fits2png(
        # general
        self.outfits_missingflux,
        self.png_missingflux,
        imcontour1=self.outfits_co10,
        imsize_as=self.imsize_as,
        ra_cnt=self.ra_agn,
        dec_cnt=self.dec_agn,
        # contour 1
        unit_cont1=None,
        levels_cont1=[0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.96],
        width_cont1=[1.0],
        color_cont1="black",
        # imshow
        fig_dpi=self.fig_dpi,
        set_grid="both",
        set_title="CO(1-0) missing flux map",
        showzero=False,
        showbeam=True,
        color_beam="black",
        scalebar=scalebar,
        label_scalebar="0.5 kpc",
        color_scalebar="black",
        # imshow colorbar
        set_cbar=True,
        clim=[0,100],
        label_cbar="missing flux (%)",
        # annotation
        numann=None,
        textann=True,
        )

        # plt histogram
        data,box = imval_all(self.outfits_missingflux)
        data     = data["data"] * data["mask"]
        data     = data.flatten()
        data     = data[np.where((data>0) & (data<100))]

        p16 = np.percentile(data,16)
        p50 = np.percentile(data,50)
        p84 = np.percentile(data,84)

        data_hist = np.histogram(data, bins=100, range=[0,100], weights=None)
        x = np.delete(data_hist[1],-1)
        y = data_hist[0] / float(np.sum(data_hist[0]))

        # plot
        plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])
        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax,grid=None,xlim=[0,100],ylim=[0,0.03],title="CO(1-0) missing flux histogram",
            xlabel="missing flux (%)",ylabel="count density",adjust=ad)

        width = abs(x[1] - x[0])
        for i in range(100):
            ax.bar(x[np.where((x>=i) & (x<i+1))], y[np.where((x>=i) & (x<i+1))], lw=0,
                color=cm.rainbow(i/100.), alpha=1.0, width=width, align="edge")       

        ax.plot([p50,p50],[0.027,0.027],"o",color="black",lw=0,markersize=15)
        ax.plot([p16,p84],[0.027,0.027],"-",color="black",lw=2)

        ax.text(p50,0.0265,"median = "+str(int(p50))+"%",fontsize=22,ha="center",va="top")

        ax.text(p16,0.0275,"16$^{th}$ pctl. = "+str(int(p16))+"%",
            fontsize=22,ha="center",va="bottom")
        ax.text(p84,0.0275,"84$^{th}$ pctl. = "+str(int(p84))+"%",
            fontsize=22,ha="center",va="bottom")

        plt.savefig(self.png_histogram, dpi=self.fig_dpi)

    ###########################
    # c8p5_create_figure_spws #
    ###########################

    def c8p5_create_figure_spws(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.png_specscan_b6,taskname)

        combine_two_png(
        self.png_specscan_b3,
        self.png_specscan_b6,
        self.final_specscan,
        self.box_specscan,
        self.box_specscan,
        )

    ##########################
    # c8p5_plot_spw_setup_b6 #
    ##########################

    def c8p5_plot_spw_setup_b6(
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
        sci_band = []
        for i in range(len(band)):
            this_band = band[i]
            if this_band=="6":
                sci_band.append(i)

        data = np.c_[project[sci_band], freq_info[sci_band], ang_res[sci_band], pi_name[sci_band], galname[sci_band], array[sci_band]]

        # prepare for plot: sci_band spws
        list_spw,list_color,list_lw = [],[],[]
        for i in range(len(data[:,1])):
            this_freq  = data[i,1]
            this_array = data[i,5]
            this_spws  = this_freq.split(" U ")

            if this_array=="12m":
                this_color = "black"
                this_lw = 1.0
            else:
                this_color = "tomato"
                this_lw = 2.0

            for j in range(len(this_spws)):
                this_spw = this_spws[j].split(",")[0].lstrip("[")
                this_spw = this_spw.rstrip("GHz").split("..")
                this_spw = [float(this_spw[0]),float(this_spw[1])]
                list_spw.append(this_spw)
                list_color.append(this_color)
                list_lw.append(this_lw)

        list_data = np.c_[list_spw,list_color]
        list_data = list_data[np.argsort(list_data[:, 0].astype(np.float64))]

        # prepare for plot: lines
        list_line = np.loadtxt(self.line_key,dtype="str")
        list_linefreq = [float(s[2]) for s in list_line if "b6" in s[0]]
        list_linename = [s[0].split("line_b6_")[1] for s in list_line if "b6" in s[0]]
        list_lineoffset = [float(s[1]) for s in list_line if "b6" in s[0]]

        # plot
        plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(nrows=5, ncols=30)
        ax1 = plt.subplot(gs[0:3,1:30])
        ax2 = plt.subplot(gs[3:5,1:30])

        # ax setup
        myax_set(ax1,xlim=[209,277],ylim=[0.0,10.0],labelbottom=False,labelleft=False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.tick_params("x", length=0, which="major")
        ax1.tick_params("y", length=0, which="major")

        myax_set(ax2,xlim=[209,277],ylim=[-5,100],labelbottom=False,labelleft=False,lw_outline=1.5)
        ax2.tick_params("x", length=0, which="major")
        ax2.tick_params("y", length=0, which="major")

        # ax1
        list_text = []
        for i in range(len(list_linename)):
        	this_name   = list_linename[i]
        	this_freq   = list_linefreq[i] / (1+self.z)
        	this_offset = list_lineoffset[i]
        	ax1.plot([this_freq,this_freq],[1.0,6.0],color="green",lw=2)
        	this_text = ax1.text(this_freq+this_offset,6.0,this_name,
        	    rotation=60,fontsize=15,ha="left",va="bottom")

        # ax2: arcival spw
        for i in range(len(list_data)):
            x = [float(list_data[i][0]),float(list_data[i][1])]
            ax2.plot(x,[i+1,i+1],"-",color=list_data[i][2],lw=list_lw[i])

        # ax2: proposed spw
        for j in range(len(self.b6_spw_setup)):
            x = [self.b6_spw_setup[j]-1.875/2.0, self.b6_spw_setup[j]+1.875/2.0]
            y = [j*2*4/5.+65,j*2*4/5.+65]
            if j>19:
                y = [j*2*4/5.+65-16*2,j*2*4/5.+65-16*2]

            ax2.plot(x,y,color="blue",lw=5)

        # text
        ax1.text(0.50,0.90,"ALMA Band 6 Coverage",
            color="black",weight="bold",transform=ax1.transAxes,fontsize=20,ha="center")
        ax2.text(0.95,0.05,"proposed 7m+TP SPWs",
            color="blue",weight="bold",transform=ax2.transAxes,fontsize=17,ha="right")
        ax2.text(0.95,0.25,"archival 12m SPWs",
            color="black",transform=ax2.transAxes,fontsize=17,ha="right")
        ax2.text(0.95,0.16,"archival 7m SPWs",
            color="tomato",transform=ax2.transAxes,fontsize=17,ha="right")

        # ax1 grid
        width = 0.4
        ax1.plot([211,275],[1,1],lw=2,color="black",zorder=1e9)
        ax1.plot([220,220],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([230,230],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([240,240],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([250,250],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([260,260],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([270,270],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.text(220,1.0-width-0.1,"220",ha="center",va="top",fontsize=15)
        ax1.text(230,1.0-width-0.1,"230",ha="center",va="top",fontsize=15)
        ax1.text(240,1.0-width-0.1,"240",ha="center",va="top",fontsize=15)
        ax1.text(250,1.0-width-0.1,"250",ha="center",va="top",fontsize=15)
        ax1.text(260,1.0-width-0.1,"260",ha="center",va="top",fontsize=15)
        ax1.text(270,1.0-width-0.1,"270",ha="center",va="top",fontsize=15)

        plt.savefig(self.png_specscan_b6, dpi=self.fig_dpi)

    ##########################
    # c8p5_plot_spw_setup_b3 #
    ##########################

    def c8p5_plot_spw_setup_b3(
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
        sci_band = []
        for i in range(len(band)):
            this_band = band[i]
            if this_band=="3":
                sci_band.append(i)

        data = np.c_[project[sci_band], freq_info[sci_band], ang_res[sci_band], pi_name[sci_band], galname[sci_band], array[sci_band]]

        # prepare for plot: sci_band spws
        list_spw,list_color,list_lw = [],[],[]
        for i in range(len(data[:,1])):
            this_freq  = data[i,1]
            this_array = data[i,5]
            this_spws  = this_freq.split(" U ")

            if this_array=="12m":
                this_color = "black"
                this_lw = 1.0
            else:
                this_color = "tomato"
                this_lw = 2.0

            for j in range(len(this_spws)):
                this_spw = this_spws[j].split(",")[0].lstrip("[")
                this_spw = this_spw.rstrip("GHz").split("..")
                this_spw = [float(this_spw[0]),float(this_spw[1])]
                list_spw.append(this_spw)
                list_color.append(this_color)
                list_lw.append(this_lw)

        list_data = np.c_[list_spw,list_color]
        list_data = list_data[np.argsort(list_data[:, 0].astype(np.float64))]

        # prepare for plot: lines
        list_line = np.loadtxt(self.line_key,dtype="str")
        list_linefreq = [float(s[2]) for s in list_line if "b3" in s[0]]
        list_linename = [s[0].split("line_b3_")[1] for s in list_line if "b3" in s[0]]
        list_lineoffset = [float(s[1]) for s in list_line if "b3" in s[0]]

        # plot
        plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(nrows=5, ncols=30)
        ax1 = plt.subplot(gs[0:3,1:30])
        ax2 = plt.subplot(gs[3:5,1:30])

        # ax setup
        myax_set(ax1,xlim=[83,117],ylim=[0.0,10.0],labelbottom=False,labelleft=False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.tick_params("x", length=0, which="major")
        ax1.tick_params("y", length=0, which="major")

        myax_set(ax2,xlim=[83,117],ylim=[-5,100],labelbottom=False,labelleft=False,lw_outline=1.5)
        ax2.tick_params("x", length=0, which="major")
        ax2.tick_params("y", length=0, which="major")

        # ax1
        list_text = []
        for i in range(len(list_linename)):
        	this_name   = list_linename[i]
        	this_freq   = list_linefreq[i] / (1+self.z)
        	this_offset = list_lineoffset[i]
        	ax1.plot([this_freq,this_freq],[1.0,6.0],color="green",lw=2)
        	this_text = ax1.text(this_freq+this_offset,6.0,this_name,
        	    rotation=60,fontsize=15,ha="left",va="bottom")

        # ax2: arcival spw
        for i in range(len(list_data)):
            x = [float(list_data[i][0]),float(list_data[i][1])]
            ax2.plot(x,[i+1,i+1],"-",color=list_data[i][2],lw=list_lw[i])

        # ax2: proposed spw
        for j in range(len(self.b3_spw_setup)):
            x = [self.b3_spw_setup[j]-1.875/2.0, self.b3_spw_setup[j]+1.875/2.0]
            y = [j*2+65,j*2+65]
            ax2.plot(x,y,color="blue",lw=5)

        # text
        ax1.text(0.50,0.90,"ALMA Band 3 Coverage",
            color="black",weight="bold",transform=ax1.transAxes,fontsize=20,ha="center")
        ax2.text(0.95,0.05,"proposed 7m+TP SPWs",
            color="blue",weight="bold",transform=ax2.transAxes,fontsize=17,ha="right")
        ax2.text(0.95,0.25,"archival 12m SPWs",
            color="black",transform=ax2.transAxes,fontsize=17,ha="right")
        ax2.text(0.95,0.16,"archival 7m SPWs",
            color="tomato",transform=ax2.transAxes,fontsize=17,ha="right")

        # ax1 grid
        width = 0.4
        ax1.plot([84,116],[1,1],lw=2,color="black",zorder=1e9)
        ax1.plot([85,85],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([95,95],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([105,105],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([115,115],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.plot([115,115],[1.0-width,1.0+width],lw=2,color="black",zorder=1e9)
        ax1.text(85,1.0-width-0.1,"85",ha="center",va="top",fontsize=15)
        ax1.text(95,1.0-width-0.1,"95",ha="center",va="top",fontsize=15)
        ax1.text(105,1.0-width-0.1,"105",ha="center",va="top",fontsize=15)
        ax1.text(115,1.0-width-0.1,"115",ha="center",va="top",fontsize=15)
        ax1.text(115,1.0-width-0.1,"115",ha="center",va="top",fontsize=15)

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
