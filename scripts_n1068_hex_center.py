"""
Python class for the NGC 1068 PCA project.

history:
2021-11-10   created
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
class ToolsPCA():
    """
    Class for the NGC 1068 PCA project.
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

        # initialize directories
        self.dir_raw      = None
        self.dir_ready    = None
        self.dir_other    = None
        self.dir_products = None

        self.fig_dpi = 200

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ToolsPCA."
            
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
            self.r_cnd          = 3.0 * self.scale_pc / 1000. # kpc
            self.r_cnd_as       = 3.0
            self.r_sbr          = 10.0 * self.scale_pc / 1000. # kpc
            self.r_sbr_as       = 10.0
            self.gridsize       = int(np.ceil(self.r_sbr_as*2/self.beam)) + 2

            # output maps
            self.outmap_mom0    = self.dir_ready + self._read_key("outmaps_mom0")
            self.outfits_mom0   = self.dir_ready + self._read_key("outfits_maps_mom0")
            self.outmap_emom0   = self.dir_ready + self._read_key("outmaps_emom0")
            self.outfits_emom0  = self.dir_ready + self._read_key("outfits_maps_emom0")

            # output txt and png
            self.table_hex_obs  = self.dir_ready + self._read_key("table_hex_obs")

            self.outpng_pca_mom0 = self.dir_products + self._read_key("outpng_pca_mom0")
            self.outpng_pca_r13co = self.dir_products + self._read_key("outpng_pca_r13co")
            self.outpng_pca_rhcn = self.dir_products + self._read_key("outpng_pca_rhcn")

            self.outpng_mom0 = self.dir_products + self._read_key("outpng_mom0")

    ###################
    # run_ngc1068_pca #
    ###################

    def run_ngc1068_pca(
        self,
        # analysis
        do_prepare       = False,
        do_sampling      = False,
        do_pca           = False,
        plot_hexmap      = False,
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        # analysis
        if do_prepare==True:
            self.align_maps()

        if do_sampling==True:
            self.hex_sampling()

        if do_pca==True:
            self.run_hex_pca() # something wrong! reproduce previous PCA first! see scripts_n1068_dense.py

        if plot_hexmap==True:
            self.plot_hex_mom0()
            self.plot_hex_ratio()

    ##################
    # plot_hex_ratio #
    ##################

    def plot_hex_ratio(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # read header
        f      = open(self.table_hex_obs)
        header = f.readline()
        header = header.split(" ")[3:]
        header = np.array([s.split("\n")[0] for s in header])
        f.close()

        # import data
        data      = np.loadtxt(self.table_hex_obs)
        len_data  = (len(data[0])-2)/2
        header    = header[:len_data]
        ra        = data[:,0]
        dec       = data[:,1]
        data_mom0 = data[:,2:len_data+2]

        data_hcn  = data_mom0[:,np.where(header=="hcn10")[0][0]]

        # plot
        for i in range(len(header)):
            this_c = data_mom0[:,i] / data_hcn
            this_c[np.where(np.isinf(this_c))] = 0
            this_c[np.where(np.isnan(this_c))] = 0
            this_name = header[i]

            this_x = ra[this_c>0]
            this_y = dec[this_c>0]
            this_c = this_c[this_c>0]

            output = self.outpng_mom0.replace("???","r_"+this_name+"_hcn")

            if len(this_c)!=0:
                print("# plot " + output)
                self._plot_hexmap(
                    output,
                    this_x,
                    this_y,
                    this_c,
                    this_name,
                    ann=False,
                    )

    #################
    # plot_hex_mom0 #
    #################

    def plot_hex_mom0(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # read header
        f      = open(self.table_hex_obs)
        header = f.readline()
        header = header.split(" ")[3:]
        header = np.array([s.split("\n")[0] for s in header])
        f.close()

        # import data
        data      = np.loadtxt(self.table_hex_obs)
        len_data  = (len(data[0])-2)/2
        header    = header[:len_data]
        ra        = data[:,0]
        dec       = data[:,1]
        data_mom0 = data[:,2:len_data+2]

        # plot
        for i in range(len(header)):
            this_c = data_mom0[:,i]
            this_name = header[i]

            this_x = ra[this_c>0]
            this_y = dec[this_c>0]
            this_c = this_c[this_c>0]

            output = self.outpng_mom0.replace("???",this_name)

            if len(this_c)!=0:
                print("# plot " + output)
                self._plot_hexmap(
                    output,
                    this_x,
                    this_y,
                    this_c,
                    this_name,
                    ann=False,
                    )

    ###############
    # run_hex_pca #
    ###############

    def run_hex_pca(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # extract line name
        f = open(self.table_hex_obs)
        header = f.readline()
        header = header.split(" ")[1:]
        header = [s.split("\n")[0] for s in header]
        f.close()

        # extract mom0 data
        data = np.loadtxt(self.table_hex_obs)
        x          = data[:,0]
        y          = data[:,1]
        r          = np.sqrt(x**2 + y**2)
        len_data   = (len(data[0])-2)/2

        data_mom0  = data[:,2:len_data+2]
        data_emom0 = data[:,len_data+2:]
        name_mom0  = np.array(header[2:len_data+2])

        mom0_13co  = data_mom0[:,np.where(name_mom0=="13co10")[0][0]]
        emom0_13co = data_emom0[:,np.where(name_mom0=="13co10")[0][0]]

        mom0_hcn   = data_mom0[:,np.where(name_mom0=="hcn10")[0][0]]
        emom0_hcn  = data_emom0[:,np.where(name_mom0=="hcn10")[0][0]]

        # constrain data by detection number of pixels
        list_mom0  = r
        list_r13co = r
        list_rhcn  = r
        list_name  = ["r"]

        for i in range(len(data_mom0[0])):
            this_mom0  = data_mom0[:,i]
            this_emom0 = data_emom0[:,i]
            this_name  = name_mom0[i]
            this_mom0  = np.where(this_mom0>=this_emom0*self.snr_mom,this_mom0,0)

            if len(this_mom0[this_mom0!=0])>=10:
                # save line name
                list_name.append(this_name)

                # save mom0
                list_mom0  = np.c_[list_mom0,this_mom0]

                # save ratio relative to 13co
                ratio      = this_mom0/mom0_13co
                ratio[np.isnan(ratio)] = 0
                ratio[np.isinf(ratio)] = 0
                list_r13co = np.c_[list_r13co,ratio]

                # save ratio relative to hcn
                ratio      = this_mom0/mom0_hcn
                ratio[np.isnan(ratio)] = 0
                ratio[np.isinf(ratio)] = 0
                list_rhcn  = np.c_[list_rhcn,ratio]

        list_mom0  = list_mom0[:,1:]
        list_r13co = list_r13co[:,1:]
        list_rhcn  = list_rhcn[:,1:]
        list_name  = list_name[1:]

        print("# survived lines for PCA analysis are...")
        print(list_name)

        # normalize
        list_name_r13co  = []
        list_name_rhcn   = []
        list_mom0_mean  = r
        list_r13co_mean = r
        list_rhcn_mean  = r
        for i in range(len(list_name)):
            this_name  = list_name[i]
            this_mom0  = list_mom0[:,i]
            this_r13co = list_r13co[:,i]
            this_rhcn  = list_rhcn[:,i]

            thres = 1e9

            mean_mom0   = np.mean(this_mom0[np.where(this_mom0!=thres)])
            mean_r13co  = np.mean(this_r13co[np.where(this_r13co!=thres)])
            mean_rhcn   = np.mean(this_rhcn[np.where(this_rhcn!=thres)])

            std_mom0   = np.std(this_mom0[np.where(this_mom0!=thres)])
            std_r13co  = np.std(this_r13co[np.where(this_r13co!=thres)])
            std_rhcn   = np.std(this_rhcn[np.where(this_rhcn!=thres)])

            list_mom0_mean  = np.c_[list_mom0_mean, np.where(r<=self.r_sbr_as, (this_mom0-mean_mom0)/std_mom0, 0)]
            if this_name!="13co10":
                list_name_r13co.append(this_name)
                list_r13co_mean = np.c_[list_r13co_mean, np.where(r<=self.r_sbr_as, (this_r13co-mean_r13co)/std_r13co, 0)]
            if this_name!="hcn10":
                list_name_rhcn.append(this_name)
                list_rhcn_mean  = np.c_[list_rhcn_mean, np.where(r<=self.r_sbr_as, (this_rhcn-mean_rhcn)/std_rhcn, 0)]

            list_mom0_mean[np.isnan(list_mom0_mean)] = 0
            list_r13co_mean[np.isnan(list_r13co_mean)] = 0
            list_rhcn_mean[np.isnan(list_rhcn_mean)] = 0
            list_mom0_mean[np.isinf(list_mom0_mean)] = 0
            list_r13co_mean[np.isinf(list_r13co_mean)] = 0
            list_rhcn_mean[np.isinf(list_rhcn_mean)] = 0

        list_mom0_mean  = list_mom0_mean[:,1:].T
        list_r13co_mean = list_r13co_mean[:,1:].T
        list_rhcn_mean  = list_rhcn_mean[:,1:].T

        # run pca
        pca_2d_hex(
            x,
            y,
            list_mom0_mean,
            list_name,
            self.outpng_pca_mom0,
            "_150pc",
            self.snr_mom,
            self.beam,
            self.gridsize,
            )

        pca_2d_hex(
            x,
            y,
            list_rhcn_mean,
            list_name_rhcn,
            self.outpng_pca_rhcn,
            "_150pc",
            self.snr_mom,
            self.beam,
            self.gridsize,
            )

        pca_2d_hex(
            x,
            y,
            list_r13co_mean,
            list_name_r13co,
            self.outpng_pca_r13co,
            "_150pc",
            self.snr_mom,
            self.beam,
            self.gridsize,
            )

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
            print("# sampling " + this_mom0.split("/")[-1])
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
            print("# sampling " + this_emom0.split("/")[-1])
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
        run_importfits(self.map_av,template)

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

    ################
    # _plot_hexmap #
    ################

    def _plot_hexmap(
        self,
        outpng,
        x,y,c,
        title,
        title_cbar="(K km s$^{-1}$)",
        cmap="rainbow",
        plot_cbar=True,
        ann=True,
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
        xlim=[29.5, -29.5],
        ylim=[-29.5, 29.5],
        xlabel="R.A. offset (arcsec)",
        ylabel="Decl. offset (arcsec)",
        adjust=[0.10,0.99,0.10,0.93],
        )
        ax.set_aspect('equal', adjustable='box')

        # plot
        im = ax.scatter(x, y, s=690, c=c, cmap=cmap, marker="h", linewidths=0)

        # cbar
        cbar = plt.colorbar(im)
        if plot_cbar==True:
            cax  = fig.add_axes([0.19, 0.12, 0.025, 0.35])
            fig.colorbar(im, cax=cax)

        # text
        ax.text(0.03, 0.93, title, color="black", transform=ax.transAxes, weight="bold", fontsize=24)

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=300)

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
