"""
Python class for the NGC 1068 PCA project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:
ALMA Band 3 data 2013.1.00279.S

usage:
> import os
> from scripts_n1068_hex_pca import ToolsPCA as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_projects/galkey_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_projects/key_n1068_hex_pca.txt",
>     )
>
> # main
> tl.run_ngc1068_pca(
>     # analysis
>     do_prepare             = True,ß
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To

history:
2022-04-07   created
Toshiki Saito@NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_sampling import *
from mycasa_lowess import *
from mycasa_tasks import *
from mycasa_plots import *
from mycasa_pca import *

############
# ToolsPCA #
############
class ToolsN6240Contin():
    """
    Class for the NGC 6240 continuum project.
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

        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_final)

    def _set_input_fits(self):
        """
        """

        self.map_b3 = self.dir_raw + self._read_key("map_b3")
        self.map_b4 = self.dir_raw + self._read_key("map_b4")
        self.map_b6 = self.dir_raw + self._read_key("map_b6")
        self.map_b7 = self.dir_raw + self._read_key("map_b7")
        self.map_b8 = self.dir_raw + self._read_key("map_b8")
        self.map_b9 = self.dir_raw + self._read_key("map_b9")

        self.pb_b3  = self.dir_raw + self._read_key("pb_b3")
        self.pb_b4  = self.dir_raw + self._read_key("pb_b4")
        self.pb_b6  = self.dir_raw + self._read_key("pb_b6")
        self.pb_b7  = self.dir_raw + self._read_key("pb_b7")
        self.pb_b8  = self.dir_raw + self._read_key("pb_b8")
        self.pb_b9  = self.dir_raw + self._read_key("pb_b9")

    def _set_output_fits(self):
        """
        """

        self.outfits_b3 = self.dir_raw + self._read_key("outfits_b3")
        self.outfits_b4 = self.dir_raw + self._read_key("outfits_b4")
        self.outfits_b6 = self.dir_raw + self._read_key("outfits_b6")
        self.outfits_b7 = self.dir_raw + self._read_key("outfits_b7")
        self.outfits_b8 = self.dir_raw + self._read_key("outfits_b8")
        self.outfits_b9 = self.dir_raw + self._read_key("outfits_b9")

    def _set_input_param(self):
        """
        """

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
        self.gridsize  = 27 # int(np.ceil(self.r_sbr_as*2/self.beam))

    def _set_output_txt_png(self):
        """
        """

    ######################
    # run_ngc6240_contin #
    ######################

    def run_ngc6240_contin(
        self,
        # analysis
        do_prepare       = False,
        # plot figures in paper
        # calc
        calc_image_stats = True,
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        # analysis
        if do_prepare==True:
            self.align_maps()

        # calc
        if calc_image_stats==True:
            self.calc_image_stats()

    ####################
    # calc_image_stats #
    ####################

    def calc_image_stats(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.map_b3,taskname)

        # create central mask for rms measurement
        inmask = "mask.image"
        run_immath_one(self.map_b3,inmask,"iif(IM0>=0.99,1,0)")
        outmask = "mask.image2"
        run_immath_one(self.map_b3,outmask,"iif(IM0<0.99,1,0)")

        # measure diameter of the mask
        data,_ = imval_all(inmask)
        data   = data["data"] * data["mask"]
        data   = np.array(data.flatten())
        numpix = len(data[data>0])
        pix    = abs(imhead(inmask)["incr"][0]) * 3600 * 180/np.pi
        #
        fov_as = numpix * pix**2

        # measure b3 stats
        this_map = self.map_b3
        run_immath_two(this_map,inmask,this_map+"_in","IM0*IM1")
        run_immath_two(this_map,outmask,this_map+"_out","IM0*IM1")
        data_in,_  = imval_all(this_map+"_in")
        data_in    = data_in["data"] * data_in["mask"]
        data_in    = data_in.flatten()
        data_in    = data_in[data_in!=0]
        data_out,_ = imval_all(this_map+"_out")
        data_out   = data_out["data"] * data_out["mask"]
        data_out   = data_out.flatten()
        data_out   = data_out[data_out!=0]
        #
        b3_beam    = beam_area(this_map)
        b3_max     = np.max(data_in)
        b3_sum     = np.sum(data_in) / b3_beam
        b3_rms     = np.sqrt(np.mean(np.square(data_out)))

        # measure b4 stats
        this_map = self.map_b3
        run_imregrid(inmask,this_map,inmask+"_b4")
        run_imregrid(outmask,this_map,outmask+"_b4")
        #
        run_immath_two(this_map,inmask+"_b4",this_map+"_in","IM0*IM1")
        run_immath_two(this_map,outmask+"_b4",this_map+"_out","IM0*IM1")
        data_in,_  = imval_all(this_map+"_in")
        data_in    = data_in["data"] * data_in["mask"]
        data_in    = data_in.flatten()
        data_in    = data_in[data_in!=0]
        data_out,_ = imval_all(this_map+"_out")
        data_out   = data_out["data"] * data_out["mask"]
        data_out   = data_out.flatten()
        data_out   = data_out[data_out!=0]
        #
        b4_beam    = beam_area(this_map)
        b4_max     = np.max(data_in)
        b4_sum     = np.sum(data_in) / b4_beam
        b4_rms     = np.sqrt(np.mean(np.square(data_out)))

        # measure b6 stats
        this_map = self.map_b6
        run_imregrid(inmask,this_map,inmask+"_b6")
        run_imregrid(outmask,this_map,outmask+"_b6")
        #
        run_immath_two(this_map,inmask+"_b6",this_map+"_in","IM0*IM1")
        run_immath_two(this_map,outmask+"_b6",this_map+"_out","IM0*IM1")
        data_in,_  = imval_all(this_map+"_in")
        data_in    = data_in["data"] * data_in["mask"]
        data_in    = data_in.flatten()
        data_in    = data_in[data_in!=0]
        data_out,_ = imval_all(this_map+"_out")
        data_out   = data_out["data"] * data_out["mask"]
        data_out   = data_out.flatten()
        data_out   = data_out[data_out!=0]
        #
        b6_beam    = beam_area(this_map)
        b6_max     = np.max(data_in)
        b6_sum     = np.sum(data_in) / b6_beam
        b6_rms     = np.sqrt(np.mean(np.square(data_out)))

        # measure b6 stats
        this_map = self.map_b6
        run_imregrid(inmask,this_map,inmask+"_b6")
        run_imregrid(outmask,this_map,outmask+"_b6")
        #
        run_immath_two(this_map,inmask+"_b6",this_map+"_in","IM0*IM1")
        run_immath_two(this_map,outmask+"_b6",this_map+"_out","IM0*IM1")
        data_in,_  = imval_all(this_map+"_in")
        data_in    = data_in["data"] * data_in["mask"]
        data_in    = data_in.flatten()
        data_in    = data_in[data_in!=0]
        data_out,_ = imval_all(this_map+"_out")
        data_out   = data_out["data"] * data_out["mask"]
        data_out   = data_out.flatten()
        data_out   = data_out[data_out!=0]
        #
        b6_beam    = beam_area(this_map)
        b6_max     = np.max(data_in)
        b6_sum     = np.sum(data_in) / b6_beam
        b6_rms     = np.sqrt(np.mean(np.square(data_out)))

        # print calc
        print("fov_as = ",fov_as,"arcsec^2")
        print("#")
        print("B3 max = ",b3_max,"Jy/b")
        print("B3 tot = ",b3_sum,"Jy")
        print("B3 rms = ",b3_rms,"Jy/b")
        print("B3 peak S/N = ",b3_max/b3_rms,"Jy/b")
        print("#")
        print("B4 max = ",b4_max,"Jy/b")
        print("B4 tot = ",b4_sum,"Jy")
        print("B4 rms = ",b4_rms,"Jy/b")
        print("B4 peak S/N = ",b4_max/b4_rms,"Jy/b")

    ##############
    # align_maps #
    ##############

    def align_maps(self):
        """
        """

        template = "template.image"

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.map_b3,taskname)

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

###################
# end of ToolsPCA #
###################
