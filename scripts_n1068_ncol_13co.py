"""
Python class for the NGC 1068 Ncol project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:
ALMA main Band 3 data 
imaging script        all processed by phangs pipeline v2
                      Leroy et al. 2021, ApJS, 255, 19 (https://ui.adsabs.harvard.edu/abs/2021ApJS..255...19L)

usage:
> import os
> from scripts_n1068_hex_pca import ToolsPCA as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_n1068_hex_pca/key_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_n1068_hex_pca/key_figures.txt",
>     )
>
> # main
> tl.run_ngc1068_pca(
>     )
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To

history:
2022-01-30   created
Toshiki Saito@Nichidai/NAOJ
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
class ToolsNcol():
    """
    Class for the NGC 1068 Ncol project.
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
            self.modname = "ToolsNcol."
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

        self.cube_13co10_60pc = self.dir_raw + self._read_key("cube_13co10_60pc")

        self.ecube_13co10_60pc = self.dir_raw + self._read_key("ecube_13co10_60pc")

        self.cube_13co21_60pc = self.dir_raw + self._read_key("cube_13co21_60pc")

        self.ecube_13co21_60pc = self.dir_raw + self._read_key("ecube_13co21_60pc")

    def _set_output_fits(self):
        """
        """

        self.outmaps_13co10 = self.dir_ready + self._read_key("outmaps_13co10")
        self.outmaps_13co21 = self.dir_ready + self._read_key("outmaps_13co21")

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

    def _set_output_txt_png(self):
        """
        """

        # output txt and png
        #self.table_hex_obs = self.dir_products + self._read_key("table_hex_obs")

    ####################
    # run_ngc1068_ncol #
    ####################

    def run_ngc1068_ncol(
        self,
        # analysis
        do_prepare             = False,
        do_sampling            = False,
        do_pca                 = False,
        # plot figures in paper
        plot_hexmap_mom0       = False,
        plot_envmask           = False,
        plot_hexmap_pca        = False,
        plot_hexmap_pca_podium = False,
        plot_median_line_graph = False,
        do_imagemagick         = False,
        # supplement
        plot_supplements       = False,
        do_imagemagick_sub     = False,
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
            self.run_hex_pca(output=self.outpng_pca_mom0,outtxt=self.table_hex_pca_mom0,reverse=True)
            #self.run_hex_pca(output=self.outpng_pca_rhcn,outtxt=self.table_hex_pca_rhcn,denom="hcn10",reverse=True)
            #self.run_hex_pca(output=self.outpng_pca_r13co,outtxt=self.table_hex_pca_r13co,denom="13co10",reverse=True)

        # plot figures in paper
        if plot_hexmap_mom0==True:
            self.plot_hexmap_mom0()

        if plot_envmask==True:
            self.plot_envmask()

        if plot_hexmap_pca==True:
            self.plot_hexmap_pca()

        if plot_hexmap_pca_podium==True:
            self.plot_hexmap_pca_ratio_podium()

        if plot_median_line_graph==True:
            self.plot_max_line_graph(denom="co10",ylim=[-2.2,0.9])
            #self.plot_max_line_graph(denom="hcn10",ylim=[-1.8,2.0])

        if do_imagemagick==True:
            self.immagick_figures()

        # supplement
        if plot_supplements==True:
            self.plot_radial()
            self.plot_hexmap_pca_podium()
            self.plot_hexmap_ratio(denom="13co10")
            self.plot_hexmap_ratio(denom="hcn10")

        if do_imagemagick_sub==True:
            self.immagick_figures_sub()

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

####################
# end of ToolsNcol #
####################
