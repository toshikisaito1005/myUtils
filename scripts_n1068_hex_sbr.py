"""
Python class for the NGC 1068 outer region spectral scan project.
using CASA.

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:
ALMA Band 3 Nakajima et al.
SFR         Tsai et al. 2012

usage:
> import os
> from scripts_n3110_co import ToolsSBR as tools
> 
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                          To

history:
Date         Action
2021-10-28   created
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

import mycasa_tasks as mytask
reload(mytask)

from mycasa_stacking import cube_stacking
from mycasa_sampling import *
from mycasa_plots import *
from mycasa_pca import *

###########################
### ToolsDense
###########################
class ToolsSBR():
    """
    Class for the NGC 1068 outer region spectral scan project.
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

        self.fig_dpi = 200

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ToolsDense."
            
            # get directories
            self.dir_proj       = self._read_key("dir_proj")
            self.dir_raw        = self.dir_proj + self._read_key("dir_raw")
            self.dir_ready      = self.dir_proj + self._read_key("dir_ready")
            self.dir_other      = self.dir_proj + self._read_key("dir_other")
            self.dir_products   = self.dir_proj + self._read_key("dir_products")
            self.dir_final      = self.dir_proj + self._read_key("dir_final")

            self._create_dir(self.dir_ready)
            self._create_dir(self.dir_products)
            self._create_dir(self.dir_final)

            # input maps
            self.map_av         = self.dir_other + self._read_key("map_av")
            self.maps_mom0      = glob.glob(self.dir_raw + self._read_key("maps_mom0"))
            self.maps_mom0.sort()
            self.maps_emom0     = glob.glob(self.dir_raw + self._read_key("maps_emom0"))
            self.maps_emom0.sort()

            # ngc1068 properties
            self.ra_agn         = float(self._read_key("ra_agn", "gal").split("deg")[0])
            self.dec_agn        = float(self._read_key("dec_agn", "gal").split("deg")[0])
            self.scale_pc       = float(self._read_key("scale", "gal"))
            self.scale_kpc      = self.scale_pc / 1000.

            self.beam           = 2.14859173174056
            self.snr_mom        = 3.0
            self.r_sbr          = 10.0 * self.scale_pc / 1000. # kpc
            self.r_sbr_as       = 10.0
            self.detection_frac = 0.75

            # output maps
            self.outmap_mom0    = self.dir_ready + self._read_key("outmaps_mom0")
            self.outfits_mom0   = self.dir_ready + self._read_key("outfits_maps_mom0")
            self.outmap_emom0   = self.dir_ready + self._read_key("outmaps_emom0")
            self.outfits_emom0  = self.dir_ready + self._read_key("outfits_maps_emom0")

            # output txt and png
            self.table_hex_obs  = self.dir_ready + self._read_key("table_hex_obs")
            self.table_hex_constrain = self.dir_ready + self._read_key("table_hex_constrain")

            self.outpng_corner_slope = self.dir_products + self._read_key("outpng_corner_slope")
            self.outpng_corner_coeff = self.dir_products + self._read_key("outpng_corner_coeff")
            self.outpng_corner_score = self.dir_products + self._read_key("outpng_corner_score")

            self.outpng_hexmap = self.dir_products + self._read_key("outpng_hexmap")

    ###################
    # run_ngc1068_sbr #
    ###################

    def run_ngc1068_sbr(
        self,
        do_prepare     = False,
        do_sampling    = False,
        do_constrain   = False,
        plot_scatters  = False,
        plot_corners   = False,
        plot_showhex   = False,
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        if do_prepare==True:
            self.align_maps()

        if do_sampling==True:
            self.hex_sampling()

        if do_constrain==True:
            self.constrain_table()

        if plot_scatters==True:
            self.plot_scatters()

        if plot_corners==True:
            self.plot_corners()

        if plot_showhex==True:
            self.plot_showhex()

    ################
    # plot_showhex #
    ################

    def plot_showhex(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # read header
        f      = open(self.table_hex_obs)
        header = f.readline()
        header = header.split(" ")[3:]
        header = [s.split("\n")[0] for s in header]
        f.close()

        # import data
        data      = np.loadtxt(self.table_hex_obs)
        len_data  = (len(data[0])-2)/2
        header    = header[:len_data]
        ra        = data[:,0]
        dec       = data[:,1]
        data_mom0 = data[:,2:len_data+2]
        name_mom0 = list(map(str.upper,header))
        name_mom0 = [s.split("10")[0].split("21")[0] for s in name_mom0]
        name_mom0 = [s.replace("13","$^{13}$").replace("18","$^{18}$") for s in name_mom0]
        name_mom0 = [s.replace("HCOP","HCO$^+$") for s in name_mom0]
        name_mom0 = [s.replace("N2HP","N$_2$H$^+$") for s in name_mom0]
        name_mom0 = [s.replace("CH3OH","CH$_3$OH") for s in name_mom0]

        # plot
        for i in range(len(name_mom0)):
            this_mom0 = data_mom0[:,i]
            this_name = name_mom0[i]
            this_outpng = self.outpng_hexmap.replace("???",header[i])

            this_x = ra#[this_mom0>0]
            this_y = dec#[this_mom0>0]
            this_c = this_mom0#[this_mom0>0]

            if len(this_c)>0:
                print("# plot " + this_outpng)
                self._plot_hexmap(this_outpng,this_x,this_y,this_c,this_name)

    ################
    # _plot_hexmap #
    ################

    def _plot_hexmap(
        self,
        outpng,
        x,y,c,
        title,
        title_cbar="(K km s$^{-1}$)",
        ):
        """
        """

        # set plt, ax
        fig = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid=None,
        xlim=[25.0, -25.0],
        ylim=[-25.0, 25.0],
        xlabel="R.A. offset (arcsec)",
        ylabel="Decl. offset (arcsec)",
        adjust=[0.10,0.99,0.10,0.93],
        )
        ax.set_aspect('equal', adjustable='box')

        # plot
        im = ax.scatter(x, y, s=1000, c=c, cmap="gnuplot", marker="h", linewidths=0)

        # cbar
        cbar = plt.colorbar(im)
        cax  = fig.add_axes([0.19, 0.12, 0.025, 0.35])
        fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_tick_params(color="white")
        #cbar.outline.set_edgecolor(fg_color)

        #cbar.set_label(title_cbar)
        #cbar.outline.set_linewidth(2.5)

        # ann
        r_sbr = patches.Circle(xy=(-0,0), radius=self.r_sbr_as,
            fill=False, alpha=1.0, ec="white", ls="dashed", lw=3.5)
        ax.add_patch(r_sbr)

        # text
        ax.text(0.05, 0.9, title, color="white", transform=ax.transAxes, weight="bold", fontsize=24)

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=300)

    ################
    # plot_corners #
    ################

    def plot_corners(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_constrain,taskname)

        # read header
        f      = open(self.table_hex_constrain)
        header = f.readline()
        header = header.split(" ")[4:]
        f.close()

        # read data
        data      = np.loadtxt(self.table_hex_constrain)
        dist_kcp  = data[:,2]
        data_mom0 = data[:,3:]
        name_mom0 = [s.split("\n")[0] for s in header]
        name_mom0 = list(map(str.upper,name_mom0))
        name_mom0 = [s.split("10")[0].split("21")[0] for s in name_mom0]
        name_mom0 = [s.replace("13","$^{13}$").replace("18","$^{18}$") for s in name_mom0]
        name_mom0 = [s.replace("HCOP","HCO$^+$") for s in name_mom0]
        name_mom0 = [s.replace("N2HP","N$_2$H$^+$") for s in name_mom0]
        name_mom0 = [s.replace("CH3OH","CH$_3$OH") for s in name_mom0]

        # measure slope
        l = range(len(name_mom0))
        array_slope = np.zeros([len(name_mom0), len(name_mom0)])
        array_slope = np.where(array_slope==0, np.nan, array_slope)
        for i in itertools.combinations(l, 2):
            x   = np.array(data_mom0[:,i[0]])
            y   = np.array(data_mom0[:,i[1]])
            cut = np.where((x>0) & (y>0))
            x   = np.log10(x[cut])
            y   = np.log10(y[cut])

            # fit
            popt,_  = curve_fit(self._f_lin, x, y, p0=[1.0,0.0], maxfev = 10000)

            array_slope[i[1],i[0]] = popt[0]

        array_coeff = np.zeros([len(name_mom0), len(name_mom0)])
        array_coeff = np.where(array_coeff==0, np.nan, array_coeff)
        # measure coeff
        for i in itertools.combinations(l, 2):
            x   = np.array(data_mom0[:,i[0]])
            y   = np.array(data_mom0[:,i[1]])
            cut = np.where((x>0) & (y>0))
            x   = np.log10(x[cut])
            y   = np.log10(y[cut])

            # measure
            coeff  = np.round(np.corrcoef(x,y)[0,1], 2)

            array_coeff[i[1],i[0]] = coeff

        # mesure best correlation using slope and coeff
        l            = 1-abs(array_slope-1)
        l            = np.nan_to_num(l)
        scaled_slope = np.where(l!=0,(l-np.min(l[l!=0])) / (np.max(l[l!=0])-np.min(l[l!=0])),0)

        l            = abs(array_coeff)
        l            = np.nan_to_num(l)
        scaled_coeff = np.where(l!=0,(l-np.min(l[l!=0])) / (np.max(l[l!=0])-np.min(l[l!=0])),0)

        array_score  = np.sqrt((scaled_slope**2 + scaled_coeff**2)/2.0)
        array_score  = np.where(array_score==0,np.nan,array_score)

        # plot 
        self._plot_corner(
            self.outpng_corner_slope,
            array_slope,
            name_mom0,
            "Slopes of log-log plot",
            "x-axis lines",
            "y-axis lines",
            clim=[0.5,1.5],
            )

        self._plot_corner(
            self.outpng_corner_coeff,
            array_coeff,
            name_mom0,
            "Correlation of log-log plot",
            "x-axis lines",
            "y-axis lines",
            clim=[0.5,1.0],
            )

        self._plot_corner(
            self.outpng_corner_score,
            array_score,
            name_mom0,
            "Correlation score (1=best,0=worst)",
            "x-axis lines",
            "y-axis lines",
            clim=[0.0,1.0],
            cmap="rainbow",
            )

    #################
    # plot_scatters #
    #################

    def plot_scatters(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_constrain,taskname)

        # read header
        f      = open(self.table_hex_constrain)
        header = f.readline()
        header = header.split(" ")[4:]
        f.close()

        # read data
        data      = np.loadtxt(self.table_hex_constrain)
        dist_kcp  = data[:,2]
        data_mom0 = data[:,3:]
        name_mom0 = [s.split("\n")[0] for s in header]

        # get data
        list_mom0 = []
        for i in range(len(name_mom0)):
            this_mom0 = data_mom0[:,i]
            this_name = name_mom0[i]

            if this_name=="n2hp10":
                mom0_n2hp = this_mom0
            else:
                list_mom0.append(this_mom0)

        # plot
        for i in range(len(list_mom0)):
            this_mom0 = list_mom0[i]
            this_name = name_mom0[i]
            cut       = np.where((this_mom0>0) & (mom0_n2hp>0))
            x         = np.log10(this_mom0[cut])
            y         = np.log10(mom0_n2hp[cut])

            xlabel    = "log " + this_name
            ylabel    = "log N$_2$H$^+$"
            output    = self.dir_products + "scatter_n2hp_vs_" + this_name + ".png"

            self._plot_scatters(output,x,y,xlabel=xlabel,ylabel=ylabel)

    ###################
    # constrain_table #
    ###################

    def constrain_table(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # read header
        f = open(self.table_hex_obs)
        header = f.readline()
        header = header.split(" ")[1:]
        f.close()

        # read data
        data       = np.loadtxt(self.table_hex_obs)
        x          = data[:,0]
        y          = data[:,1]
        dist_kpc   = np.sqrt(x**2+y**2) * self.scale_kpc
        len_data   = (len(data[0])-2)/2

        data_mom0  = data[:,2:len_data+2]
        data_emom0 = data[:,len_data+2:]
        name_mom0  = np.array(header[2:len_data+2])
        name_emom0 = np.array(header[len_data+2:])

        n2hp_mom0  = data_mom0[:,np.where(name_mom0=="n2hp10")[0][0]]
        n2hp_emom0 = data_emom0[:,np.where(name_mom0=="n2hp10")[0][0]]

        # constrain data by radius and N2H+ detection
        cut        = np.where((dist_kpc>=self.r_sbr) & (n2hp_mom0>=n2hp_emom0*self.snr_mom))
        x          = x[cut]
        y          = y[cut]
        dist_kpc   = dist_kpc[cut]
        data_mom0  = data_mom0[cut]
        data_emom0 = data_emom0[cut]

        # constrain data by detected pixels
        header = ["x(as) y(as) dist(kpc)"]
        table  = np.c_[x,y,dist_kpc]
        for i in range(len(data_mom0[0])):
            this_name   = name_mom0[i]
            this_mom0   = data_mom0[:,i]
            this_emom0  = data_emom0[:,i]
            this_mom0   = np.where(this_mom0>=this_emom0*self.snr_mom,this_mom0,0)

            detect_rate = len(this_mom0[this_mom0!=0]) / float(len(this_mom0))

            if detect_rate>=self.detection_frac:
                table = np.c_[table,np.array(this_mom0)]
                header.append(this_name)

        header = " ".join(header)
        os.system("rm -rf " + self.table_hex_constrain)
        np.savetxt(self.table_hex_constrain,table,header=header)

    ################
    # hex_sampling #
    ################

    def hex_sampling(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name

        # sampling mom0
        maps_mom0 = glob.glob(self.outfits_mom0.replace("???","*"))
        maps_mom0 = [s for s in maps_mom0 if "err" not in s]
        maps_mom0.sort()

        check_first(maps_mom0[0],taskname)

        header = ["ra(deg)","dec(deg)"]
        for i in range(len(maps_mom0)):
            this_mom0 = maps_mom0[i]
            this_line = this_mom0.split("/")[-1].split("n1068_")[1].split(".fits")[0]
            print("# sampleing " + this_mom0.split("/")[-1])
            x,y,z = hexbin_sampling(
                this_mom0,
                self.ra_agn,
                self.dec_agn,
                beam=self.beam,
                gridsize=27,
                err=False,
                )

            if i==0:
                output_hex = np.c_[x,y]

            output_hex = np.c_[output_hex,z]
            header.append(this_line)

        # sampling emom0
        maps_emom0 = glob.glob(self.outfits_emom0.replace("???","*"))
        maps_emom0.sort()

        for i in range(len(maps_emom0)):
            this_emom0 = maps_emom0[i]
            this_line  = this_emom0.split("/")[-1].split("n1068_")[1].split(".fits")[0]
            print("# sampleing " + this_emom0.split("/")[-1])
            x,y,z = hexbin_sampling(
                this_emom0,
                self.ra_agn,
                self.dec_agn,
                beam=self.beam,
                gridsize=27,
                err=True,
                )

            output_hex = np.c_[output_hex,z]
            header.append(this_line+"(err)")

        header = " ".join(header)
        np.savetxt(self.table_hex_obs,output_hex,header=header)

    ##############
    # align_maps #
    ##############

    def align_maps(self):
        """
        """

        template = "template.image"
        run_importfits(template,taskname)

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(template,taskname)

        # regrid mom0
        for this_map in self.maps_mom0:
            this_output  = self.outmap_mom0.replace("???",this_map.split("/")[-1].split("_")[3])
            this_outfits = self.outfits_mom0.replace("???",this_map.split("/")[-1].split("_")[3])
            run_imregrid(this_map, template, this_output)
            run_exportfits(this_output, this_outfits, True, True, True)

        # regrid emom0
        for this_map in self.maps_emom0:
            this_output  = self.outmap_emom0.replace("???",this_map.split("/")[-1].split("_")[3])
            this_outfits = self.outfits_emom0.replace("???",this_map.split("/")[-1].split("_")[3])
            run_imregrid(this_map, template, this_output)
            run_exportfits(this_output, this_outfits, True, True, True)

        # cleanup
        os.system("rm -rf template")

    ##############
    # _myax_cbar #
    ##############

    def _plot_corner(
        self,
        outpng,
        data,
        list_name,
        title,
        xlabel,
        ylabel,
        clim,
        cmap="bwr",
        ):

        fig = plt.figure(figsize=(10,9))
        gs  = gridspec.GridSpec(nrows=30, ncols=30)
        ax  = plt.subplot(gs[0:30,0:30])
        ax.set_aspect('equal', adjustable='box')
        myax_set(ax,title=title,aspect=1.0,adjust=[0.10,0.99,0.20,0.92],xlabel=xlabel,ylabel=ylabel)

        im = ax.imshow(data, interpolation="none", vmin=clim[0], vmax=clim[1], cmap=cmap)
        
        #self._myax_cbar(fig, ax, im, clim=clim)

        ax.set_xticks(range(len(list_name)))
        ax.set_xticklabels(list_name,rotation=90)
        ax.set_yticks(range(len(list_name)))
        ax.set_yticklabels(list_name)

        # text
        for i in itertools.combinations(range(len(list_name)), 2):
            this_slope = str(np.round(data[i[1],i[0]],2)).ljust(4, '0')
            ax.text(i[0],i[1],this_slope,fontsize=12,
                horizontalalignment="center", verticalalignment="center", weight="bold")

        print("# output = " + outpng)
        fig.savefig(outpng, dpi=fig_dpi)

    ##############
    # _myax_cbar #
    ##############

    def _myax_cbar(
        self,
        fig,
        ax,
        data,
        label=None,
        clim=None,
        ):

        cb = fig.colorbar(data, ax=ax)
        
        if label is not None:
            cb.set_label(label)
        
        if clim is not None:
            cb.set_clim(clim)

        cb.outline.set_linewidth(2.5)

    ##################
    # _plot_scatters #
    ##################

    def _plot_scatters(
        self,
        output,
        x,
        y,
        title=None,
        xlabel=None,
        ylabel=None,
        ):
        """
        """

        # col coeff
        coeff = str( np.round( np.corrcoef(x,y)[0,1],2 ) )

        # fit
        popt,_  = curve_fit(self._f_lin, x, y, p0=[1.0,0.0], maxfev = 10000)
        best_y = self._f_lin(x, popt[0], popt[1])

        # get xlim, ylim
        xlim = [np.median(x)-1.0,np.median(x)+1.0]
        ylim = [np.median(y)-1.0,np.median(y)+1.0]

        print("# plot " + output)
        fig = plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad = [0.215,0.83,0.10,0.90]
        ax1.set_aspect('equal', adjustable='box')
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        # plot
        ax1.scatter(x, y, lw=0, c="gray", s=100)
        ax1.plot(x, best_y, "-", color="tomato", lw=3)

        # text
        if xlim!=None and ylim!=None:
            ax1.plot([ylim[0]-1,ylim[1]+1], [ylim[0]-1,ylim[1]+1], "--", lw=2, color="black")
            ax1.plot([ylim[0]-1,ylim[1]+1], [ylim[0]-1-1.0,ylim[1]+1-1.0], "--", lw=2, color="grey")
            ax1.plot([ylim[0]-1,ylim[1]+1], [ylim[0]-1+1.0,ylim[1]+1+1.0], "--", lw=2, color="grey")

        # text
        ax1.text(0.1, 0.90, "#point $=$ "+str(len(x)), transform=ax1.transAxes)
        ax1.text(0.1, 0.85, "$r$ $=$ "+coeff, transform=ax1.transAxes)
        ax1.text(0.1, 0.80, "slope $=$ "+str(np.round(popt[0],2)), transform=ax1.transAxes)
        ax1.text(0.1, 0.75, "intercept $=$ "+str(np.round(popt[1],2)), transform=ax1.transAxes)

        # save
        plt.savefig(output, dpi=self.fig_dpi)

    ##########
    # _f_lin #
    ##########

    def _f_lin(self, x, a, b):
        """
        """

        return a * x + b

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
