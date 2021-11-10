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

            # output maps
            self.outmap_mom0    = self.dir_ready + self._read_key("outmaps_mom0")
            self.outfits_mom0   = self.dir_ready + self._read_key("outfits_maps_mom0")
            self.outmap_emom0   = self.dir_ready + self._read_key("outmaps_emom0")
            self.outfits_emom0  = self.dir_ready + self._read_key("outfits_maps_emom0")

            # output txt and png
            self.table_hex_obs  = self.dir_ready + self._read_key("table_hex_obs")

    ###################
    # run_ngc1068_pca #
    ###################

    def run_ngc1068_pca(
        self,
        # analysis
        do_prepare       = False,
        do_sampling      = False,
        do_pca           = False,
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
            self.run_hex_pca()

    ###############
    # run_hex_pca #
    ###############

    def run_hex_pca(self):
        """
        """

        # extract line name
        with open(txtdata) as f:
            header = f.readline()

        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x          = data[:,0]
        y          = data[:,1]
        r          = np.sqrt(x**2 + y**2) * self.scale_pc / 1000.
        len_data   = (len(data[0])-2)/2

        data_mom0  = data[:,2:len_data+2]
        data_emom0 = data[:,len_data+2:]
        name_mom0  = np.array(header[2:len_data+2])
        name_emom0 = np.array(header[len_data+2:])

        mom0_13co  = data_mom0[:,np.where(name_mom0=="13co10")[0][0]]
        emom0_13co = data_emom0[:,np.where(name_mom0=="13co10")[0][0]]

        mom0_hcn   = data_mom0[:,np.where(name_mom0=="hcn10")[0][0]]
        emom0_hcn  = data_emom0[:,np.where(name_mom0=="hcn10")[0][0]]

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
            cut = np.where(np.sqrt(x**2+y**2)<=self.r_sbr_as)
            x   = x[cut]
            y   = y[cut]
            z   = z[cut]

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
            cut = np.where(np.sqrt(x**2+y**2)<=self.r_sbr_as)
            x   = x[cut]
            y   = y[cut]
            z   = z[cut]

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
