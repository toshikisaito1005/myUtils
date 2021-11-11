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
            self.map_ionization = self.dir_other + self._read_key("map_ionization")
            self.maps_mom0      = glob.glob(self.dir_raw + self._read_key("maps_mom0"))
            self.maps_mom0.sort()
            self.maps_emom0     = glob.glob(self.dir_raw + self._read_key("maps_emom0"))
            self.maps_emom0.sort()

            # ngc1068 properties
            self.ra_agn         = float(self._read_key("ra_agn", "gal").split("deg")[0])
            self.dec_agn        = float(self._read_key("dec_agn", "gal").split("deg")[0])
            self.scale_pc       = float(self._read_key("scale", "gal"))
            self.scale_kpc      = self.scale_pc / 1000.

            self.beam           = 2.14859173174056 # 150pc in arcsec
            self.snr_mom        = 3.0
            self.r_cnd          = 3.0 * self.scale_pc / 1000. # kpc
            self.r_cnd_as       = 3.0
            self.r_sbr          = 10.0 * self.scale_pc / 1000. # kpc
            self.r_sbr_as       = 10.0
            self.gridsize       = 27 # int(np.ceil(self.r_sbr_as*2/self.beam))

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
        do_and_plot_pca  = False,
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

        if do_and_plot_pca==True:
            self.run_hex_pca(output=self.outpng_pca_mom0,reverse=True)
            self.run_hex_pca(output=self.outpng_pca_rhcn,denom="hcn10",reverse=True)
            self.run_hex_pca(output=self.outpng_pca_r13co,denom="13co10")

        if plot_hexmap==True:
            self.plot_hexmap_mom0()
            self.plot_hexmap_ratio(denom="hcn10")
            self.plot_hexmap_ratio(denom="13co10")

    #####################
    # plot_hexmap_ratio #
    #####################

    def plot_hexmap_ratio(self,denom=None):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # extract line name
        header,data_mom0,_,ra,dec,_ = self._read_table(self.table_hex_obs)
        data_denom = data_mom0[:,np.where(header==denom)[0][0]]

        # plot
        for i in range(len(header)):
            this_c = data_mom0[:,i] / data_denom
            this_c[np.where(np.isinf(this_c))] = 0
            this_c[np.where(np.isnan(this_c))] = 0
            this_name = header[i]

            this_x = ra[this_c>0]
            this_y = dec[this_c>0]
            this_c = this_c[this_c>0]

            output = self.outpng_mom0.replace("???","r_"+this_name+"_"+denom)

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

    ####################
    # plot_hexmap_mom0 #
    ####################

    def plot_hexmap_mom0(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # extract line name
        header,data_mom0,_,ra,dec,_ = self._read_table(self.table_hex_obs)

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

    def run_hex_pca(self,output,denom=None,reverse=False):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # extract line name
        list_name,array_data,array_err,x,y,r = self._read_table(self.table_hex_obs)

        # main
        data, data_name = [], []
        for i in range(len(list_name)):
            this_flux = array_data[:,i]
            this_err  = array_err[:,i]
            this_name = list_name[i]

            if this_name==denom:
                continue

            if denom!=None:
                index      = np.where(list_name==denom)[0][0]
                data_denom = array_data[:,index]
                err_denom  = array_err[:,index]
            else:
                data_denom = None
                err_denom  = None

            # sn cut and zero padding
            this_flux = self._process_hex_for_pca(this_flux,this_err,r,data_denom,err_denom)

            # limiting by #data
            len_data = len(this_flux[this_flux>0])
            if len_data>=10:
                print("# meet " + this_name + " #=" + str(len_data))
                data.append(this_flux.flatten())
                data_name.append(this_name)
            else:
                print("# skip " + this_name + " #=" + str(len_data))

        # run
        os.system("rm -rf " + output)
        pca_2d_hex(
            x,
            y,
            data,
            data_name,
            output,
            "_150pc",
            self.snr_mom,
            self.beam,
            self.gridsize,
            reverse=reverse,
            factor=2,
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
        maps_mom0.append(self.map_av)
        maps_mom0.append(self.map_ionization)

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
                gridsize=self.gridsize,
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
                gridsize=self.gridsize,
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
        os.system("rm -rf template.image")

    ########################
    # _process_hex_for_pca #
    ########################

    def _process_hex_for_pca(
        self,
        this_flux,
        this_err,
        this_r,
        denom_flux=None,
        denom_err=None,
        ):
        """
        """

        # sn cut
        this_thres = abs(this_err * self.snr_mom)
        this_flux  = np.where(this_flux>=this_thres, this_flux, 0)

        if denom_flux!=None:
            denom_flux = np.where(denom_flux>=denom_err*self.snr_mom, denom_flux, 0)
            this_flux  = this_flux / denom_flux
            this_flux[np.isinf(this_flux)] = 0
            this_flux[np.isnan(this_flux)] = 0

        # normalize
        this_flux  = ( this_flux - np.mean(this_flux) ) / np.std(this_flux)

        # zero padding
        this_flux[np.isnan(this_flux)] = 0
        this_flux[np.isinf(this_flux)] = 0

        # extract center by masking
        this_flux = np.where(this_r<=self.r_sbr_as, this_flux, 0)
        this_err  = np.where(this_r<=self.r_sbr_as, this_err, 0)

        return this_flux

    ###############
    # _read_table #
    ###############

    def _read_table(self,txtdata):
        """
        """

        # extract line name
        f = open(txtdata)
        header = f.readline()
        header = header.split(" ")[1:]
        header = [s.split("\n")[0] for s in header]
        f.close()

        # extract mom0 data
        data = np.loadtxt(txtdata)
        x          = data[:,0]
        y          = data[:,1]
        r          = np.sqrt(x**2 + y**2)
        len_data   = (len(data[0])-2)/2

        array_data = data[:,2:len_data+2]
        array_err  = data[:,len_data+2:]
        list_name  = np.array(header[2:len_data+2])

        return list_name, array_data, array_err, x, y, r

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
