"""
Python class for the NGC 1068 PCA project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:
ALMA main Band 3 data 2013.1.00279.S
ALMA supplements      2011.0.00061.S
                      2012.1.00657.S
                      2013.1.00060.S
                      2015.1.00960.S
                      2017.1.00586.S
                      2018.1.01506.S
                      2018.1.01684.S
                      2019.1.00130.S
imaging script        all processed by phangs pipeline v2
                      Leroy et al. 2021, ApJS, 255, 19 (https://ui.adsabs.harvard.edu/abs/2021ApJS..255...19L)
ancillary MUSE FITS   Mingozzi et al. 2019, A&A, 622, 146 (https://ui.adsabs.harvard.edu/abs/2019A%26A...622A.146M)
                      SIII/SII ratio map (ionization parameter), ionized gas density, AV maps
                      http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/622/A146

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
>     # analysis
>     do_prepare             = True,
>     do_sampling            = True,
>     do_pca                 = True,
>     # plot
>     plot_hexmap_pca        = True,
>     plot_hexmap_pca_podium = True,
>     plot_radial            = True,
>     do_imagemagick         = True,
>     # supplement
>     plot_hexmap            = True,
>     do_imagemagick_sub     = True,
>     )
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To
2021-12-21   draft_v0_211221.zip     Takano-san,Nakajima-san,Harada-san,Tamura-san

history:
2021-11-10   created
2021-11-14   2DPCA analysis for mom0 maps
2021-11-15   start drafting
2021-11-22   implement LOWESS and bootstrapped standard deviation
2021-12-21   circulate v0 draft to the paper team
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
class ToolsPCA():
    """
    Class for the NGC 1068 PCA project.
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

        self.map_av         = self.dir_other + self._read_key("map_av")
        self.map_ionization = self.dir_other + self._read_key("map_ionization")
        self.maps_mom0      = glob.glob(self.dir_raw + self._read_key("maps_mom0"))
        self.maps_emom0     = glob.glob(self.dir_raw + self._read_key("maps_emom0"))
        self.maps_mom0.sort()
        self.maps_emom0.sort()

    def _set_output_fits(self):
        """
        """

        self.outmap_mom0   = self.dir_ready + self._read_key("outmaps_mom0")
        self.outfits_mom0  = self.dir_ready + self._read_key("outfits_maps_mom0")
        self.outmap_emom0  = self.dir_ready + self._read_key("outmaps_emom0")
        self.outfits_emom0 = self.dir_ready + self._read_key("outfits_maps_emom0")

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

        # output txt and png
        self.table_hex_obs            = self.dir_ready + self._read_key("table_hex_obs")
        self.table_hex_pca_mom0       = self.dir_ready + self._read_key("table_hex_pca_mom0")
        self.table_hex_pca_r13co      = self.dir_ready + self._read_key("table_hex_pca_r13co")
        self.table_hex_pca_rhcn       = self.dir_ready + self._read_key("table_hex_pca_rhcn")

        self.outpng_pca_mom0          = self.dir_products + self._read_key("outpng_pca_mom0")
        self.outpng_pca_r13co         = self.dir_products + self._read_key("outpng_pca_r13co")
        self.outpng_pca_rhcn          = self.dir_products + self._read_key("outpng_pca_rhcn")

        self.outpng_mom0              = self.dir_products + self._read_key("outpng_mom0")
        self.outpng_12co10_oveall     = self.outpng_mom0.replace("???","12co10_overall")

        self.outpng_pca_hexmap        = self.dir_products + self._read_key("outpng_pca_hexmap")
        self.outpng_pca_scatter       = self.dir_products + self._read_key("outpng_pca_scatter")
        self.outpng_pca_hexmap_r13co  = self.dir_products + self._read_key("outpng_pca_hexmap_r13co")
        self.outpng_pca_scatter_r13co = self.dir_products + self._read_key("outpng_pca_scatter_r13co")

        outpng_pca1_mom0_podium       = self.dir_products + self._read_key("outpng_pca1_mom0_podium")
        self.outpng_pca1_mom0_1st     = outpng_pca1_mom0_podium.replace("???","1st")
        self.outpng_pca1_mom0_2nd     = outpng_pca1_mom0_podium.replace("???","2nd")
        self.outpng_pca1_mom0_3rd     = outpng_pca1_mom0_podium.replace("???","3rd")
        self.outpng_pca1_mom0_4th     = outpng_pca1_mom0_podium.replace("???","4th")

        outpng_pca2_mom0_podium       = self.dir_products + self._read_key("outpng_pca2_mom0_podium")
        self.outpng_pca2_mom0_1st     = outpng_pca2_mom0_podium.replace("???","1st")
        self.outpng_pca2_mom0_2nd     = outpng_pca2_mom0_podium.replace("???","2nd")
        self.outpng_pca2_mom0_3rd     = outpng_pca2_mom0_podium.replace("???","3rd")
        self.outpng_pca2_mom0_4th     = outpng_pca2_mom0_podium.replace("???","4th")

        outpng_pca1_ratio_podium      = self.dir_products + self._read_key("outpng_pca1_ratio_podium")
        self.outpng_pca1_ratio_1st    = outpng_pca1_ratio_podium.replace("???","1st")
        self.outpng_pca1_ratio_2nd    = outpng_pca1_ratio_podium.replace("???","2nd")
        self.outpng_pca1_ratio_3rd    = outpng_pca1_ratio_podium.replace("???","3rd")
        self.outpng_pca1_ratio_4th    = outpng_pca1_ratio_podium.replace("???","4th")

        outpng_pca2_ratio_podium      = self.dir_products + self._read_key("outpng_pca2_ratio_podium")
        self.outpng_pca2_ratio_1st    = outpng_pca2_ratio_podium.replace("???","1st")
        self.outpng_pca2_ratio_2nd    = outpng_pca2_ratio_podium.replace("???","2nd")
        self.outpng_pca2_ratio_3rd    = outpng_pca2_ratio_podium.replace("???","3rd")
        self.outpng_pca2_ratio_4th    = outpng_pca2_ratio_podium.replace("???","4th")

        self.outpng_radial1           = self.dir_products + self._read_key("outpng_radial1")
        self.outpng_radial2           = self.dir_products + self._read_key("outpng_radial2")
        self.outpng_radial3           = self.dir_products + self._read_key("outpng_radial3")

        # final
        self.final_overall            = self.dir_final + self._read_key("final_overall")
        self.final_mom0               = self.dir_final + self._read_key("final_mom0")
        self.final_pca_mom0           = self.dir_final + self._read_key("final_pca_mom0")
        self.final_pca_r13co          = self.dir_final + self._read_key("final_pca_r13co")
        self.final_pca_mom0_podium    = self.dir_final + self._read_key("final_pca_mom0_podium")
        self.final_pca_ratio_podium   = self.dir_final + self._read_key("final_pca_ratio_podium")
        self.final_hex_radial         = self.dir_final + self._read_key("final_hex_radial")

        self.box_map                  = self._read_key("box_map")
        self.box_map_noxlabel         = self._read_key("box_map_noxlabel")
        self.box_map_noylabel         = self._read_key("box_map_noylabel")
        self.box_map_noxylabel        = self._read_key("box_map_noxylabel")

    ###################
    # run_ngc1068_pca #
    ###################

    def run_ngc1068_pca(
        self,
        # analysis
        do_prepare             = False,
        do_sampling            = False,
        do_pca                 = False,
        # plot figures in paper
        plot_hexmap_mom0       = False,
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

        if plot_hexmap_pca==True:
            self.plot_hexmap_pca()

        if plot_hexmap_pca_podium==True:
            self.plot_hexmap_pca_ratio_podium()

        if plot_median_line_graph==True:
            self.plot_median_line_graph()

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

    ####################
    # immagick_figures #
    ####################

    def immagick_figures(
        self,
        delin                      = False,
        do_all                     = False,
        do_final_overall           = True,
        do_final_mom0              = False,
        do_final_pca_mom0          = False,
        do_final_pca1_ratio_podium = False,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outpng_pca_hexmap.replace("???","1"),taskname)

        if do_all==True:
            do_final_overall           = True
            do_final_mom0              = True
            do_final_pca_mom0          = True
            do_final_pca1_ratio_podium = True

        if do_final_overall==True:
            print("#####################")
            print("# create final_mom0 #")
            print("#####################")

            immagick_crop(
                self.outpng_12co10_oveall,
                self.final_overall,
                self.box_map,
                delin=delin,
                )

        if do_final_mom0==True:
            print("#####################")
            print("# create final_mom0 #")
            print("#####################")

            # panel 1
            combine_two_png(
                self.outpng_mom0.replace("???","h13cn10"),
                self.outpng_mom0.replace("???","cch10"),
                self.final_mom0+"_tmp1.png",
                self.box_map_noylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.outpng_mom0.replace("???","hcn10"),
                self.outpng_mom0.replace("???","hcop10"),
                self.final_mom0+"_tmp2.png",
                self.box_map_noxylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.final_mom0+"_tmp1.png",
                self.final_mom0+"_tmp2.png",
                self.final_mom0+"_panel1.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )

            # panel 2
            combine_two_png(
                self.outpng_mom0.replace("???","hnc10"),
                self.outpng_mom0.replace("???","hc3n109"),
                self.final_mom0+"_tmp1.png",
                self.box_map_noylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.outpng_mom0.replace("???","n2hp10"),
                self.outpng_mom0.replace("???","ch3oh21"),
                self.final_mom0+"_tmp2.png",
                self.box_map_noxylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.final_mom0+"_tmp1.png",
                self.final_mom0+"_tmp2.png",
                self.final_mom0+"_panel2.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )

            # panel 3
            combine_two_png(
                self.outpng_mom0.replace("???","cs21"),
                self.outpng_mom0.replace("???","hc3n1110"),
                self.final_mom0+"_tmp1.png",
                self.box_map_noylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.outpng_mom0.replace("???","hc3n1211"),
                self.outpng_mom0.replace("???","c18o10"),
                self.final_mom0+"_tmp2.png",
                self.box_map_noxylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.final_mom0+"_tmp1.png",
                self.final_mom0+"_tmp2.png",
                self.final_mom0+"_panel3.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )

            # panel 4
            combine_two_png(
                self.outpng_mom0.replace("???","13co10"),
                self.outpng_mom0.replace("???","cn10l"),
                self.final_mom0+"_tmp1.png",
                self.box_map_noylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.outpng_mom0.replace("???","cn10h"),
                self.outpng_mom0.replace("???","co10"),
                self.final_mom0+"_tmp2.png",
                self.box_map_noxylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.final_mom0+"_tmp1.png",
                self.final_mom0+"_tmp2.png",
                self.final_mom0+"_panel4.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )

            # panel 5
            combine_two_png(
                self.outpng_mom0.replace("???","ci10"),
                self.outpng_mom0.replace("???","siiisii_ratio"),
                self.final_mom0+"_panel5.png",
                self.box_map,
                self.box_map_noxlabel,
                delin=delin,
                )

            # combine
            combine_two_png(
                self.final_mom0+"_panel1.png",
                self.final_mom0+"_panel2.png",
                self.final_mom0+"_panel12.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

            combine_two_png(
                self.final_mom0+"_panel3.png",
                self.final_mom0+"_panel4.png",
                self.final_mom0+"_panel34.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

            combine_two_png(
                self.final_mom0+"_panel12.png",
                self.final_mom0+"_panel34.png",
                self.final_mom0+"_panel1234.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )

            combine_two_png(
                self.final_mom0+"_panel1234.png",
                self.final_mom0+"_panel5.png",
                self.final_mom0,
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                axis="column",
                )
 
        if do_final_pca_mom0==True:
            print("#########################")
            print("# create final_pca_mom0 #")
            print("#########################")

            combine_two_png(
                self.outpng_pca_hexmap.replace("???","1"),
                self.outpng_pca_hexmap.replace("???","2"),
                self.final_pca_mom0+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                self.outpng_pca_scatter,
                self.final_pca_mom0+"_tmp1.png",
                self.final_pca_mom0,
                self.box_map,
                "100000x100000+0+0",
                axis="column",
                delin=delin,
                )
            os.system("rm -rf " + self.final_pca_mom0 + "_tmp1.png")

        if do_final_pca1_ratio_podium==True:
            print("##################################")
            print("# create final_pca1_ratio_podium #")
            print("##################################")

            # pca1 mom0
            combine_two_png(
                self.outpng_pca1_ratio_1st,
                self.outpng_pca1_ratio_2nd,
                self.final_pca_ratio_podium+"_tmp1.png",
                self.box_map_noylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.outpng_pca1_ratio_3rd,
                self.outpng_pca1_ratio_4th,
                self.final_pca_ratio_podium+"_tmp2.png",
                self.box_map_noxylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.final_pca_ratio_podium+"_tmp1.png",
                self.final_pca_ratio_podium+"_tmp2.png",
                self.final_pca_ratio_podium+"_pca1.png",
                "10000x10000+0+0",
                "10000x10000+0+0",
                delin=True,
                )

            combine_two_png(
                self.outpng_pca2_ratio_1st,
                self.outpng_pca2_ratio_2nd,
                self.final_pca_ratio_podium+"_tmp1.png",
                self.box_map,
                self.box_map_noxlabel,
                delin=delin,
                )
            combine_two_png(
                self.outpng_pca2_ratio_3rd,
                self.outpng_pca2_ratio_4th,
                self.final_pca_ratio_podium+"_tmp2.png",
                self.box_map_noxlabel,
                self.box_map_noxlabel,
                delin=delin,
                )
            combine_two_png(
                self.final_pca_ratio_podium+"_tmp1.png",
                self.final_pca_ratio_podium+"_tmp2.png",
                self.final_pca_ratio_podium+"_pca2.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )

            # combine
            combine_two_png(
                self.final_pca_ratio_podium+"_pca1.png",
                self.final_pca_ratio_podium+"_pca2.png",
                self.final_pca_ratio_podium,
                "100000x100000+0+0",
                "100000x100000+0+0",
                axis="column",
                delin=True,
                )

    ########################
    # immagick_figures_sub #
    ########################

    def immagick_figures_sub(
        self,
        delin                     = False,
        do_final_pca1_mom0_podium = False,
        do_final_hex_radial       = False,
        do_final_pca_r13co        = False,
        ):
        """
        """

        if do_final_pca1_mom0_podium==True:
            print("#################################")
            print("# create final_pca1_mom0_podium #")
            print("#################################")

            # pca1 mom0
            combine_two_png(
                self.outpng_pca1_mom0_1st,
                self.outpng_pca1_mom0_2nd,
                self.final_pca_mom0_podium+"_tmp1.png",
                self.box_map_noylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.outpng_pca1_mom0_3rd,
                self.outpng_pca1_mom0_4th,
                self.final_pca_mom0_podium+"_tmp2.png",
                self.box_map_noxylabel,
                self.box_map_noxylabel,
                delin=delin,
                )
            combine_two_png(
                self.final_pca_mom0_podium+"_tmp1.png",
                self.final_pca_mom0_podium+"_tmp2.png",
                self.final_pca_mom0_podium+"_pca1.png",
                "10000x10000+0+0",
                "10000x10000+0+0",
                delin=True,
                )

            # pca2 mom0
            combine_two_png(
                self.outpng_pca2_mom0_1st,
                self.outpng_pca2_mom0_2nd,
                self.final_pca_mom0_podium+"_tmp1.png",
                self.box_map,
                self.box_map_noxlabel,
                delin=delin,
                )
            combine_two_png(
                self.outpng_pca2_mom0_3rd,
                self.outpng_pca2_mom0_4th,
                self.final_pca_mom0_podium+"_tmp2.png",
                self.box_map_noxlabel,
                self.box_map_noxlabel,
                delin=delin,
                )
            combine_two_png(
                self.final_pca_mom0_podium+"_tmp1.png",
                self.final_pca_mom0_podium+"_tmp2.png",
                self.final_pca_mom0_podium+"_pca2.png",
                "100000x100000+0+0",
                "100000x100000+0+0",
                delin=True,
                )

            # combine
            combine_two_png(
                self.final_pca_mom0_podium+"_pca1.png",
                self.final_pca_mom0_podium+"_pca2.png",
                self.final_pca_mom0_podium,
                "100000x100000+0+0",
                "100000x100000+0+0",
                axis="column",
                delin=True,
                )

        if do_final_hex_radial==True:
            print("###########################")
            print("# create final_hex_radial #")
            print("###########################")

            combine_two_png(
                self.outpng_radial1,
                self.outpng_radial2,
                self.final_hex_radial+"_tmp1.png",
                self.box_map,
                self.box_map,
                delin=delin,
                )
            combine_two_png(
                self.final_hex_radial+"_tmp1.png",
                self.outpng_radial3,
                self.final_hex_radial,
                "100000x100000+0+0",
                self.box_map,
                axis="column",
                delin=delin,
                )
            os.system("rm -rf " + self.final_hex_radial + "_tmp1.png")

        if do_final_pca_r13co==True:
            print("##########################")
            print("# create final_pca_r13co #")
            print("##########################")

            combine_three_png(
                self.outpng_pca_hexmap_r13co.replace("???","1"),
                self.outpng_pca_hexmap_r13co.replace("???","2"),
                self.outpng_pca_scatter_r13co,
                self.final_pca_r13co,
                self.box_map,
                self.box_map,
                self.box_map,
                delin=delin,
                )

    ##########################
    # plot_median_line_graph #
    ##########################

    def plot_median_line_graph(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # start
        offset = 10
        angle1 = -15 - offset
        angle2 = -115 + offset
        angle3 = 165 - offset
        angle4 = 65 + offset

        # get table
        header,data_mom0,_,x,y,r = self._read_table(self.table_hex_obs)
        theta_deg = np.degrees(np.arctan2(x, y))
        header    = [s for s in header if not "extinction"]

        # get line data

        for i in range(len(header)):
            this_name  = header[i]
            line_index = np.where(header==this_name)
            data_line  = np.array(data_mom0[:,line_index].flatten())
            data_line[np.isinf(data_line)] = 0
            data_line[np.isnan(data_line)] = 0
            data_line  = np.where(r<=self.r_sbr_as,data_line,0)

            this_cnd = np.where(r<self.r_cnd_as,data_line,0)
            print(len(this_cnd))

        # get line bicone
        #line_zn    = np.where((theta_deg>=angle1)&(theta_deg<angle4)&(r<self.r_sbr_as),data_line,0)
        #line_zs    = np.where((theta_deg>=angle3)&(r<self.r_sbr_as),data_line,0)
        #line_zs    = np.where((theta_deg<angle2)&(r<self.r_sbr_as),data_line,line_zs)

    ################################
    # plot_hexmap_pca_ratio_podium # Figure 4
    ################################

    def plot_hexmap_pca_ratio_podium(self,denom1="co10",denom2="co10"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)
        factor = 1.0

        pc1_name1  = "h13cn10"
        pc1_name2  = "hc3n109"
        pc1_name3  = "hcn10"
        pc1_name4  = "hcop10"

        pc2_name1  = "cn10h"
        pc2_name2  = "cch10"
        pc2_name3  = "hnc10"
        pc2_name4  = "cn10l"

        # extract mom0 data
        header,data_mom0,_,x,y,r = self._read_table(self.table_hex_obs)

        denom1_name  = denom1
        denom1_index = np.where(header==denom1_name)
        denom1_z     = np.array(data_mom0[:,denom1_index].flatten())

        denom2_name  = denom2
        denom2_index = np.where(header==denom2_name)
        denom2_z     = np.array(data_mom0[:,denom2_index].flatten())

        # get PC1 podium
        pc1_index1 = np.where(header==pc1_name1)
        pc1_index2 = np.where(header==pc1_name2)
        pc1_index3 = np.where(header==pc1_name3)
        pc1_index4 = np.where(header==pc1_name4)

        pc1_z1     = np.array(data_mom0[:,pc1_index1].flatten()) / denom1_z
        pc1_z2     = np.array(data_mom0[:,pc1_index2].flatten()) / denom1_z
        pc1_z3     = np.array(data_mom0[:,pc1_index3].flatten()) / denom1_z
        pc1_z4     = np.array(data_mom0[:,pc1_index4].flatten()) / denom1_z

        pc1_z1[np.isinf(pc1_z1)] = 0
        pc1_z2[np.isinf(pc1_z2)] = 0
        pc1_z3[np.isinf(pc1_z3)] = 0
        pc1_z4[np.isinf(pc1_z4)] = 0
        pc1_z1[np.isnan(pc1_z1)] = 0
        pc1_z2[np.isnan(pc1_z2)] = 0
        pc1_z3[np.isnan(pc1_z3)] = 0
        pc1_z4[np.isnan(pc1_z4)] = 0

        pc1_z1     = np.where(r<=self.r_sbr_as,pc1_z1,0)
        pc1_z2     = np.where(r<=self.r_sbr_as,pc1_z2,0)
        pc1_z3     = np.where(r<=self.r_sbr_as,pc1_z3,0)
        pc1_z4     = np.where(r<=self.r_sbr_as,pc1_z4,0)

        pc1_z1     = np.where(pc1_z1>=np.max(pc1_z1)/factor, np.max(pc1_z1)/factor, pc1_z1)
        pc1_z2     = np.where(pc1_z2>=np.max(pc1_z2)/factor, np.max(pc1_z2)/factor, pc1_z2)
        pc1_z3     = np.where(pc1_z3>=np.max(pc1_z3)/factor, np.max(pc1_z3)/factor, pc1_z3)
        pc1_z4     = np.where(pc1_z4>=np.max(pc1_z4)/factor, np.max(pc1_z4)/factor, pc1_z4)

        pc1_z1_sort = pc1_z1.flatten()
        pc1_z2_sort = pc1_z2.flatten()
        pc1_z3_sort = pc1_z3.flatten()
        pc1_z4_sort = pc1_z4.flatten()
        pc1_z1_sort.sort()
        pc1_z2_sort.sort()
        pc1_z3_sort.sort()
        pc1_z4_sort.sort()
        pc1_z1     = np.where(pc1_z1==np.max(pc1_z1), pc1_z1_sort[-2], pc1_z1)
        pc1_z2     = np.where(pc1_z2==np.max(pc1_z2), pc1_z2_sort[-2], pc1_z2)
        pc1_z3     = np.where(pc1_z3==np.max(pc1_z3), pc1_z3_sort[-2], pc1_z3)
        pc1_z4     = np.where(pc1_z4==np.max(pc1_z4), pc1_z4_sort[-2], pc1_z4)

        # get PC2 podium
        pc2_index1 = np.where(header==pc2_name1)
        pc2_index2 = np.where(header==pc2_name2)
        pc2_index3 = np.where(header==pc2_name3)
        pc2_index4 = np.where(header==pc2_name4)

        pc2_z1     = np.array(data_mom0[:,pc2_index1].flatten()) / denom2_z
        pc2_z2     = np.array(data_mom0[:,pc2_index2].flatten()) / denom2_z
        pc2_z3     = np.array(data_mom0[:,pc2_index3].flatten()) / denom2_z
        pc2_z4     = np.array(data_mom0[:,pc2_index4].flatten()) / denom2_z

        pc2_z1[np.isinf(pc2_z1)] = 0
        pc2_z2[np.isinf(pc2_z2)] = 0
        pc2_z3[np.isinf(pc2_z3)] = 0
        pc2_z4[np.isinf(pc2_z4)] = 0
        pc2_z1[np.isnan(pc2_z1)] = 0
        pc2_z2[np.isnan(pc2_z2)] = 0
        pc2_z3[np.isnan(pc2_z3)] = 0
        pc2_z4[np.isnan(pc2_z4)] = 0

        pc2_z1     = np.where(r<=self.r_sbr_as,pc2_z1,0)
        pc2_z2     = np.where(r<=self.r_sbr_as,pc2_z2,0)
        pc2_z3     = np.where(r<=self.r_sbr_as,pc2_z3,0)
        pc2_z4     = np.where(r<=self.r_sbr_as,pc2_z4,0)

        pc2_z1     = np.where(pc2_z1>=np.max(pc2_z1)/factor, np.max(pc2_z1)/factor, pc2_z1)
        pc2_z2     = np.where(pc2_z2>=np.max(pc2_z2)/factor, np.max(pc2_z2)/factor, pc2_z2)
        pc2_z3     = np.where(pc2_z3>=np.max(pc2_z3)/factor, np.max(pc2_z3)/factor, pc2_z3)
        pc2_z4     = np.where(pc2_z4>=np.max(pc2_z4)/factor, np.max(pc2_z4)/factor, pc2_z4)

        pc2_z1_sort = pc2_z1.flatten()
        pc2_z2_sort = pc2_z2.flatten()
        pc2_z3_sort = pc2_z3.flatten()
        pc2_z4_sort = pc2_z4.flatten()
        pc2_z1_sort.sort()
        pc2_z2_sort.sort()
        pc2_z3_sort.sort()
        pc2_z4_sort.sort()
        pc2_z1     = np.where(pc2_z1==np.max(pc2_z1), pc2_z1_sort[-2], pc2_z1)
        pc2_z2     = np.where(pc2_z2==np.max(pc2_z2), pc2_z2_sort[-2], pc2_z2)
        pc2_z3     = np.where(pc2_z3==np.max(pc2_z3), pc2_z3_sort[-2], pc2_z3)
        pc2_z4     = np.where(pc2_z4==np.max(pc2_z4), pc2_z4_sort[-2], pc2_z4)

        if denom1_name=="hcn10":
            denom1_str="HCN(1-0)"
        if denom1_name=="13co10":
            denom1_str="$^{13}$CO(1-0)"
        if denom1_name=="c18o10":
            denom1_str="C$^{18}$O(1-0)"
        if denom1_name=="co10":
            denom1_str="CO(1-0)"

        if denom2_name=="hcn10":
            denom2_str="HCN(1-0)"
        if denom2_name=="13co10":
            denom2_str="$^{13}$CO(1-0)"
        if denom2_name=="c18o10":
            denom2_str="C$^{18}$O(1-0)"
        if denom2_name=="co10":
            denom2_str="CO(1-0)"

        # PC1 podium+1
        self._plot_hexmap(
            self.outpng_pca1_ratio_1st,
            x,
            y,
            pc1_z1,
            "H$^{13}$CN(1-0)/"+denom1_str+" (highest PC1)",
            cmap="Reds",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            )

        self._plot_hexmap(
            self.outpng_pca1_ratio_2nd,
            x,
            y,
            pc1_z2,
            "HC$_3$N(10-9)/"+denom1_str+" (2nd highest PC1)",
            cmap="Reds",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            )

        self._plot_hexmap(
            self.outpng_pca1_ratio_3rd,
            x,
            y,
            pc1_z3,
            "HCN(1-0)/"+denom1_str+" (3rd highest PC1)",
            cmap="Reds",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            )

        self._plot_hexmap(
            self.outpng_pca1_ratio_4th,
            x,
            y,
            pc1_z4,
            "HCO$^+$(1-0)/"+denom1_str+" (4th highest PC1)",
            cmap="Reds",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            )

        # PC2 podium+1
        self._plot_hexmap(
            self.outpng_pca2_ratio_1st,
            x,
            y,
            pc2_z1,
            "CN(1$_{3/2}$-0$_{1/2}$)/"+denom2_str+" (highest PC2)",
            cmap="PuBu",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="",
            )

        self._plot_hexmap(
            self.outpng_pca2_ratio_2nd,
            x,
            y,
            pc2_z2,
            "CCH(1-0)/"+denom2_str+" (2nd highest PC2)",
            cmap="PuBu",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="",
            )

        self._plot_hexmap(
            self.outpng_pca2_ratio_3rd,
            x,
            y,
            pc2_z3,
            "HNC(1-0)/"+denom2_str+" (3rd highest PC2)",
            cmap="PuBu",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="",
            )

        self._plot_hexmap(
            self.outpng_pca2_ratio_4th,
            x,
            y,
            pc2_z4,
            "CN(1$_{1/2}$-0$_{1/2}$)/"+denom2_str+" (4th highest PC2)",
            cmap="PuBu",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="",
            )

    ###################
    # plot_hexmap_pca # Figure 3
    ###################

    def plot_hexmap_pca(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_pca_mom0,taskname)

        # extract mom0 data
        data = np.loadtxt(self.table_hex_pca_mom0)
        x        = data[:,0]
        y        = data[:,1]
        r        = np.sqrt(x**2 + y**2)
        data_pca = data[:,2:]

        ###################
        # plot PC scatter #
        ###################
        print("# plot " + self.outpng_pca_scatter)

        table_hex_pca_mom0_score = self.table_hex_pca_mom0.replace(".txt","_score.txt")
        data_score = np.loadtxt(table_hex_pca_mom0_score,dtype="str")
        score_name = data_score[:,0]
        score_pc1  = data_score[:,1].astype(np.float64)
        score_pc2  = data_score[:,2].astype(np.float64)
        score_pc1  = score_pc1 / np.std(score_pc1)
        score_pc2  = score_pc2 / np.std(score_pc2) * -1

        # set plt, ax
        fig = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid=None,
        xlim=[-3.2,1.8],
        ylim=[-1.6,3.4],
        xlabel="PC1",
        ylabel="PC2",
        adjust=[0.023,0.963,0.10,0.93],
        )
        ax.set_aspect('equal', adjustable='box')

        for i in range(len(score_name)):
            pc1 = score_pc1[i]
            pc2 = score_pc2[i]
            ax.plot([0,pc1],[0,pc2],"-",color="grey",lw=2)

            if score_name[i]=="n2hp10":
                ax.text(pc1,pc2,"N$_2$H$^+$",fontsize=18,ha="center",va="top")
            elif score_name[i]=="hc3n109":
                continue
            elif score_name[i]=="hc3n1110":
                continue
            elif score_name[i]=="hc3n1211":
                ax.text(pc1,pc2-0.15,"HC$_3$Nx3",fontsize=18,ha="left",va="center")
            elif score_name[i]=="h13cn10":
                ax.text(pc1,pc2-0.1,"H$^{13}$CN",fontsize=18,ha="left",va="bottom")
            elif score_name[i]=="cs21":
                ax.text(pc1,pc2,"CS",fontsize=18,ha="center",va="top")
            elif score_name[i]=="hcn10":
                ax.text(pc1,pc2,"HCN",fontsize=18,ha="left",va="center")
            elif score_name[i]=="hcop10":
                ax.text(pc1,pc2,"HCO$^+$",fontsize=18,ha="left",va="center")
            elif score_name[i]=="cn10l":
                ax.text(pc1,pc2,"CN(1-0)l",fontsize=18,ha="left",va="center")
            elif score_name[i]=="hnc10":
                ax.text(pc1,pc2,"HNC",fontsize=18,ha="center",va="bottom")
            elif score_name[i]=="cn10h":
                ax.text(pc1,pc2,"CN(1-0)h",fontsize=18,ha="center",va="bottom")
            elif score_name[i]=="ci10":
                ax.text(pc1,pc2,"[CI]",fontsize=18,ha="center",va="bottom")
            elif score_name[i]=="cch10":
                ax.text(pc1,pc2,"CCH",fontsize=18,ha="right",va="bottom")
            elif score_name[i]=="siiisii_ratio":
                ax.text(pc1,pc2,"[SIII]/[SII] ratio",fontsize=18,ha="center",va="bottom")
            elif score_name[i]=="co10":
                ax.text(pc1,pc2,"CO",fontsize=18,ha="right",va="center")
            elif score_name[i]=="13co10":
                ax.text(pc1,pc2,"$^{13}$CO",fontsize=18,ha="right",va="center")
            elif score_name[i]=="c18o10":
                ax.text(pc1,pc2,"C$^{18}$O",fontsize=18,ha="right",va="center")
            elif score_name[i]=="ch3oh21":
                ax.text(pc1,pc2,"CH$_3$OH",fontsize=18,ha="center",va="top")
            else:
                ax.text(pc1,pc2,score_name[i],fontsize=14)

        ax.text(0.03, 0.93, "(a) PC1 vs. PC2", color="black", transform=ax.transAxes, weight="bold", fontsize=24)

        # save
        os.system("rm -rf " + self.outpng_pca_scatter)
        plt.savefig(self.outpng_pca_scatter, dpi=300)

        #################
        # plot PCA maps #
        #################
        anntexts = [True,False]
        cmaps    = ["Reds","PuBu"]
        headers  = ["b","c"]
        for i in range(len(data_pca[0])):
            this_c    = data_pca[:,i]
            this_x    = x[this_c!=0]
            this_y    = y[this_c!=0]
            this_c    = this_c[this_c!=0]
            this_text = anntexts[i]
            thid_cmap = cmaps[i]

            if abs(np.min(this_c))>abs(np.max(this_c)):
                this_c = this_c * -1

            this_c = np.where(this_c>np.max(this_c)/1.5,np.max(this_c)/1.5,this_c)

            output = self.outpng_pca_hexmap.replace("???",str(i+1))

            print("# plot " + output)
            self._plot_hexmap(
                output,
                this_x,
                this_y,
                this_c,
                "(" + headers[i] + ") PC"+str(i+1),
                cmap=thid_cmap,
                ann=True,
                add_text=this_text,
                lim=13,
                size=3600,
                )

    ####################
    # plot_hexmap_mom0 # Figures 1 and 2
    ####################

    def plot_hexmap_mom0(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # extract line name
        header,data_mom0,data_err,ra,dec,r = self._read_table(self.table_hex_obs)

        # plot
        for i in range(len(header)):
            this_c    = data_mom0[:,i]
            this_cerr = data_err[:,i]
            this_name = header[i]

            cut = np.where(r<=self.r_sbr_as)
            this_x    = ra[cut]
            this_y    = dec[cut]
            this_c    = this_c[cut]
            this_cerr = this_cerr[cut]
            this_c    = np.where(this_c>=this_cerr*self.snr_mom,this_c,0)

            output = self.outpng_mom0.replace("???",this_name)

            if this_name=="co10":
                co10 = data_mom0[:,i]
                print("# plot overall " + self.outpng_12co10_oveall)
                self._plot_hexmap(
                    self.outpng_12co10_oveall,
                    ra,
                    dec,
                    co10,
                    "$^{12}$CO(1-0)",
                    ann      = True,
                    add_text = False,
                    lim      = 28,
                    size     = 780,
                    label    = "(K km s$^{-1}$)",
                    scalebar = "500pc",
                    )  

            if len(this_c[this_c!=0])>=10:
                this_name = this_name.replace("1110","(11-10)")
                this_name = this_name.replace("1211","(12-11)")
                this_name = this_name.replace("10","(1-0)")
                this_name = this_name.replace("21","(2-1)")
                this_name = this_name.replace("(1-0)9","(10-9)")
                this_name = this_name.replace("12","$^{12}$")
                this_name = this_name.replace("13","$^{13}$")
                this_name = this_name.replace("18","$^{18}$")
                this_name = this_name.replace("c3","c$_3$").replace("h3","H$_3$")
                this_name = this_name.replace("ci","[CI]").replace("n2","n$_2$")
                this_name = this_name.replace("c","C").replace("o","O")
                this_name = this_name.replace("n","N").replace("h","H")
                this_name = this_name.replace("p","$^+$").replace("s","S")
                this_name = this_name.replace("SiiiSii_ratiO","[SIII]/[SII] ratio")
                this_name = this_name.replace("(1-0)H","(1-0)h")
                this_name = this_name.replace("11-(1-0)","11-10")
                this_name = this_name.replace("($^{12}$-11)","(12-11)")

                # plot
                print("# plot " + output)
                self._plot_hexmap(
                    output,
                    this_x,
                    this_y,
                    this_c,
                    this_name + " [$N_{pixel}$ = " + str(len(this_c[this_c!=0])) + "]",
                    ann      = True,
                    add_text = False,
                    lim      = 13,
                    size     = 3600,
                    label    = "(K km s$^{-1}$)",
                    )

    ###############
    # run_hex_pca #
    ###############

    def run_hex_pca(
        self,
        output,
        outtxt,
        denom=None,
        reverse=False,
        ):
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
        array_hex_pca, pca_score = pca_2d_hex(
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

        header = "ra(deg) dec(deg) PC1 PC2 ..."
        np.savetxt(outtxt,array_hex_pca,header=header)

        header = "line PC1 PC2 ..."
        np.savetxt(outtxt.replace(".txt","_score.txt"),pca_score,header=header,fmt="%s")

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
        maps_emom0.append(self.map_av)
        maps_emom0.append(self.map_ionization)

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

            if this_line=="extinction":
                z = z * 0
            elif this_line=="siiisii_ratio":
                z = z * 0

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
        ann=False,
        lim=29.5,
        size=690,
        add_text=False,
        label="",
        bgcolor="white",
        scalebar="100pc",
        ):
        """
        """

        # set plt, ax
        fig = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])
        fig.patch.set_facecolor(bgcolor)

        # set ax parameter
        myax_set(
        ax,
        grid=None,
        xlim=[lim, -lim],
        ylim=[-lim, lim],
        xlabel="R.A. offset (arcsec)",
        ylabel="Decl. offset (arcsec)",
        adjust=[0.10,0.99,0.10,0.93],
        )
        ax.set_aspect('equal', adjustable='box')

        # plot
        im = ax.scatter(x, y, s=size, c=c, cmap=cmap, marker="h", linewidths=0, vmin=0)

        # cbar
        cbar = plt.colorbar(im)
        if plot_cbar==True:
            cax  = fig.add_axes([0.19, 0.12, 0.025, 0.35])
            fig.colorbar(im, cax=cax).set_label(label)

        # scale bar
        if scalebar=="100pc":
            bar = 100 / self.scale_pc
            ax.plot([-10,-10+bar],[-10,-10],"-",color="black",lw=4)
            ax.text(-10, -10.5, "100 pc",
                    horizontalalignment="right", verticalalignment="top")
        elif scalebar=="500pc":
            bar = 500 / self.scale_pc
            ax.plot([-22,-22+bar],[-22,-22],"-",color="black",lw=4)
            ax.text(-22, -22.5, "500 pc",
                    horizontalalignment="right", verticalalignment="top")

        # text
        ax.text(0.03, 0.93, title, color="black", transform=ax.transAxes, weight="bold", fontsize=32)

        # ann
        if ann==True:
            theta1      = -10.0 # degree
            theta2      = 70.0 # degree
            fov_diamter = 16.5 # arcsec (12m+7m Band 8)

            fov_diamter = 16.5
            efov1 = patches.Ellipse(xy=(-0,0), width=fov_diamter,
                height=fov_diamter, angle=0, fill=False, edgecolor="black",
                alpha=1.0, lw=3.5)

            ax.add_patch(efov1)

            # plot NGC 1068 AGN and outflow geometry
            x1 = fov_diamter/2.0 * np.cos(np.radians(-1*theta1+90))
            y1 = fov_diamter/2.0 * np.sin(np.radians(-1*theta1+90))
            ax.plot([x1, -x1], [y1, -y1], "--", c="black", lw=3.5)
            x2 = fov_diamter/2.0 * np.cos(np.radians(-1*theta2+90))
            y2 = fov_diamter/2.0 * np.sin(np.radians(-1*theta2+90))
            ax.plot([x2, -x2], [y2, -y2], "--", c="black", lw=3.5)

        # add annotation comment
        if add_text==True:
            ax.plot([0,-7], [0,10], lw=3, c="black")
            ax.text(-10.5, 10.5, "AGN position",
                horizontalalignment="right", verticalalignment="center", weight="bold")

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

    ##########################
    # will be decommissioned #
    ##########################

    def plot_radial(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)

        # get data
        data_cone_co10,r  = self._get_bicone_radial("co10")
        data_cone_hcn10,_ = self._get_bicone_radial("hcn10")
        data_cone_1st,_   = self._get_bicone_radial("cn10h")
        data_cone_2nd,_   = self._get_bicone_radial("hnc10")
        data_cone_3rd,_   = self._get_bicone_radial("cch10")

        data_disk_co10,_  = self._get_bicone_radial("co10",cone="out")
        data_disk_hcn10,_ = self._get_bicone_radial("hcn10",cone="out")
        data_disk_1st,_   = self._get_bicone_radial("cn10h",cone="out")
        data_disk_2nd,_   = self._get_bicone_radial("hnc10",cone="out")
        data_disk_3rd,_   = self._get_bicone_radial("cch10",cone="out")

        # plot
        xlim  = [2.1,9.6]

        rdata = [data_cone_3rd-data_cone_co10,data_disk_3rd-data_disk_co10]
        self._plot_radial(
            self.outpng_radial1,
            r,
            rdata,
            "(a) Radial CCH(1-0)/CO(1-0) ratio",
            size=1000/10,
            ylabel="log Ratio",
            xlim=xlim,
            ylim=[-2.3,0.1],
            ann=1,
            )

        rdata = [data_cone_1st-data_cone_hcn10,data_disk_1st-data_disk_hcn10]
        self._plot_radial(
            self.outpng_radial2,
            r,
            rdata,
            "(b) Radial CN(1$_{3/2}$-0$_{1/2}$)/HCN(1-0) ratio",
            size=1000/10,
            ylabel="log Ratio",
            xlim=xlim,
            ylim=[-0.8,0.6],
            )

        rdata = [data_cone_2nd-data_cone_hcn10,data_disk_2nd-data_disk_hcn10]
        self._plot_radial(
            self.outpng_radial3,
            r,
            rdata,
            "(c) Radial HNC(1-0)/HCN(1-0) ratio",
            size=1000/10,
            ylabel="log Ratio",
            xlim=xlim,
            ylim=[-0.8,-0.1],
            ) 

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

            this_c = np.log10(this_c)
            this_c[np.where(np.isinf(this_c))] = 0
            this_c[np.where(np.isnan(this_c))] = 0

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

    def plot_hexmap_pca_podium(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_obs,taskname)
        factor = 2.0

        # extract mom0 data
        header,data_mom0,_,x,y,r = self._read_table(self.table_hex_obs)

        # get PC1 podium
        pc1_name1  = "h13cn10"
        pc1_name2  = "hc3n109"
        pc1_name3  = "hcn10"
        pc1_name4  = "13co10"

        pc1_index1 = np.where(header==pc1_name1)
        pc1_index2 = np.where(header==pc1_name2)
        pc1_index3 = np.where(header==pc1_name3)
        pc1_index4 = np.where(header==pc1_name4)

        pc1_z1     = np.array(data_mom0[:,pc1_index1].flatten())
        pc1_z2     = np.array(data_mom0[:,pc1_index2].flatten())
        pc1_z3     = np.array(data_mom0[:,pc1_index3].flatten())
        pc1_z4     = np.array(data_mom0[:,pc1_index4].flatten())

        pc1_z1     = np.where(r<=self.r_sbr_as,pc1_z1,0)
        pc1_z2     = np.where(r<=self.r_sbr_as,pc1_z2,0)
        pc1_z3     = np.where(r<=self.r_sbr_as,pc1_z3,0)
        pc1_z4     = np.where(r<=self.r_sbr_as,pc1_z4,0)

        pc1_z1     = np.where(pc1_z1>=np.max(pc1_z1)/factor, np.max(pc1_z1)/factor, pc1_z1)
        pc1_z2     = np.where(pc1_z2>=np.max(pc1_z2)/factor, np.max(pc1_z2)/factor, pc1_z2)
        pc1_z3     = np.where(pc1_z3>=np.max(pc1_z3)/factor, np.max(pc1_z3)/factor, pc1_z3)
        pc1_z4     = np.where(pc1_z4>=np.max(pc1_z4)/factor, np.max(pc1_z4)/factor, pc1_z4)

        # get PC1 podium
        pc2_name1  = "cn10h"
        pc2_name2  = "cch10"
        pc2_name3  = "hnc10"
        pc2_name4  = "c18o10"

        pc2_index1 = np.where(header==pc2_name1)
        pc2_index2 = np.where(header==pc2_name2)
        pc2_index3 = np.where(header==pc2_name3)
        pc2_index4 = np.where(header==pc2_name4)

        pc2_z1     = np.array(data_mom0[:,pc2_index1].flatten())
        pc2_z2     = np.array(data_mom0[:,pc2_index2].flatten())
        pc2_z3     = np.array(data_mom0[:,pc2_index3].flatten())
        pc2_z4     = np.array(data_mom0[:,pc2_index4].flatten())

        pc2_z1     = np.where(r<=self.r_sbr_as,pc2_z1,0)
        pc2_z2     = np.where(r<=self.r_sbr_as,pc2_z2,0)
        pc2_z3     = np.where(r<=self.r_sbr_as,pc2_z3,0)
        pc2_z4     = np.where(r<=self.r_sbr_as,pc2_z4,0)

        pc2_z1     = np.where(pc2_z1>=np.max(pc2_z1)/factor, np.max(pc2_z1)/factor, pc2_z1)
        pc2_z2     = np.where(pc2_z2>=np.max(pc2_z2)/factor, np.max(pc2_z2)/factor, pc2_z2)
        pc2_z3     = np.where(pc2_z3>=np.max(pc2_z3)/factor, np.max(pc2_z3)/factor, pc2_z3)
        pc2_z4     = np.where(pc2_z4>=np.max(pc2_z4)/factor, np.max(pc2_z4)/factor, pc2_z4)

        # PC1 podium+1
        self._plot_hexmap(
            self.outpng_pca1_mom0_1st,
            x,
            y,
            pc1_z1,
            "H$^{13}$CN(1-0) (highest PC1)",
            cmap="Reds",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="(K km s$^{-1}$)",
            )

        self._plot_hexmap(
            self.outpng_pca1_mom0_2nd,
            x,
            y,
            pc1_z2,
            "HC$_3$N(10-9) (2nd highest PC1)",
            cmap="Reds",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="(K km s$^{-1}$)",
            )

        self._plot_hexmap(
            self.outpng_pca1_mom0_3rd,
            x,
            y,
            pc1_z3,
            "HCN(1-0) (3rd highest PC1)",
            cmap="Reds",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="(K km s$^{-1}$)",
            )

        self._plot_hexmap(
            self.outpng_pca1_mom0_4th,
            x,
            y,
            pc1_z4,
            "$^{13}$CO(1-0) (lowest PC1)",
            cmap="Greys",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="(K km s$^{-1}$)",
            )

        # PC2 podium+1
        self._plot_hexmap(
            self.outpng_pca2_mom0_1st,
            x,
            y,
            pc2_z1,
            "CN(1-0)h (highest PC2)",
            cmap="PuBu",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="(K km s$^{-1}$)",
            )

        self._plot_hexmap(
            self.outpng_pca2_mom0_2nd,
            x,
            y,
            pc2_z2,
            "CCH(1-0) (2nd highest PC2)",
            cmap="PuBu",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="(K km s$^{-1}$)",
            )

        self._plot_hexmap(
            self.outpng_pca2_mom0_3rd,
            x,
            y,
            pc2_z3,
            "HNC(1-0) (3rd highest PC2)",
            cmap="PuBu",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="(K km s$^{-1}$)",
            )

        self._plot_hexmap(
            self.outpng_pca2_mom0_4th,
            x,
            y,
            pc2_z4,
            "C$^{18}$O(1-0) (lowest PC2)",
            cmap="Greys",
            ann=True,
            add_text=False,
            lim=13,
            size=3600,
            label="(K km s$^{-1}$)",
            )

    def plot_hexmap_pca_r13co(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.table_hex_pca_r13co,taskname)

        # extract mom0 data
        data = np.loadtxt(self.table_hex_pca_r13co)
        x        = data[:,0]
        y        = data[:,1]
        r        = np.sqrt(x**2 + y**2)
        data_pca = data[:,2:]

        ###################
        # plot PC scatter #
        ###################
        print("# plot " + self.outpng_pca_scatter_r13co)

        table_hex_pca_score = self.table_hex_pca_r13co.replace(".txt","_score.txt")
        data_score = np.loadtxt(table_hex_pca_score,dtype="str")
        score_name = data_score[:,0]
        score_pc1  = data_score[:,1].astype(np.float64)
        score_pc2  = data_score[:,2].astype(np.float64)
        score_pc1  = score_pc1 / np.std(score_pc1)
        score_pc2  = score_pc2 / np.std(score_pc2)

        # set plt, ax
        fig = plt.figure(figsize=(13,10))
        plt.rcParams["font.size"] = 16
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax = plt.subplot(gs[0:10,0:10])

        # set ax parameter
        myax_set(
        ax,
        grid=None,
        xlim=[-3.2,1.8],
        ylim=[-1.6,3.4],
        xlabel="PC1",
        ylabel="PC2",
        adjust=[0.023,0.963,0.10,0.93],
        )
        ax.set_aspect('equal', adjustable='box')

        for i in range(len(score_name)):
            pc1 = score_pc1[i]
            pc2 = score_pc2[i]
            ax.plot([0,pc1],[0,pc2],"-",color="grey",lw=2)

            """
            if score_name[i]=="n2hp10":
                ax.text(pc1,pc2,"N$_2$H$^+$",fontsize=18,ha="center",va="top")
            elif score_name[i]=="hc3n109":
                continue
            elif score_name[i]=="hc3n1110":
                continue
            elif score_name[i]=="hc3n1211":
                ax.text(pc1,pc2-0.15,"HC$_3$Nx3",fontsize=18,ha="left",va="center")
            elif score_name[i]=="h13cn10":
                ax.text(pc1,pc2-0.1,"H$^{13}$CN",fontsize=18,ha="left",va="bottom")
            elif score_name[i]=="cs21":
                ax.text(pc1,pc2,"CS",fontsize=18,ha="center",va="top")
            elif score_name[i]=="hcn10":
                ax.text(pc1,pc2,"HCN",fontsize=18,ha="left",va="center")
            elif score_name[i]=="hcop10":
                ax.text(pc1,pc2,"HCO$^+$",fontsize=18,ha="left",va="center")
            elif score_name[i]=="cn10l":
                ax.text(pc1,pc2,"CN(1-0)l",fontsize=18,ha="left",va="center")
            elif score_name[i]=="hnc10":
                ax.text(pc1,pc2,"HNC",fontsize=18,ha="center",va="bottom")
            elif score_name[i]=="cn10h":
                ax.text(pc1,pc2,"CN(1-0)h",fontsize=18,ha="center",va="bottom")
            elif score_name[i]=="ci10":
                ax.text(pc1,pc2,"[CI]",fontsize=18,ha="center",va="bottom")
            elif score_name[i]=="cch10":
                ax.text(pc1,pc2,"CCH",fontsize=18,ha="right",va="bottom")
            elif score_name[i]=="siiisii_ratio":
                ax.text(pc1,pc2,"[SIII]/[SII] ratio",fontsize=18,ha="center",va="bottom")
            elif score_name[i]=="co10":
                ax.text(pc1,pc2,"CO",fontsize=18,ha="right",va="center")
            elif score_name[i]=="13co10":
                ax.text(pc1,pc2,"$^{13}$CO",fontsize=18,ha="right",va="center")
            elif score_name[i]=="c18o10":
                ax.text(pc1,pc2,"C$^{18}$O",fontsize=18,ha="right",va="center")
            elif score_name[i]=="ch3oh21":
                ax.text(pc1,pc2,"CH$_3$OH",fontsize=18,ha="center",va="top")
            else:
            """
            ax.text(pc1,pc2,score_name[i],fontsize=14)

        ax.text(0.03, 0.93, "PC1 vs. PC2", color="black", transform=ax.transAxes, weight="bold", fontsize=24)

        # save
        os.system("rm -rf " + self.outpng_pca_scatter_r13co)
        plt.savefig(self.outpng_pca_scatter_r13co, dpi=300)

        #################
        # plot PCA maps #
        #################
        anntexts = [True,False]
        cmaps    = ["Reds","PuBu"]
        for i in range(len(data_pca[0])):
            this_c    = data_pca[:,i]
            this_x    = x[this_c!=0]
            this_y    = y[this_c!=0]
            this_c    = this_c[this_c!=0]
            this_text = anntexts[i]
            thid_cmap = cmaps[i]

            if abs(np.min(this_c))>abs(np.max(this_c)):
                this_c = this_c * -1

            this_c = np.where(this_c>np.max(this_c)/1.5,np.max(this_c)/1.5,this_c)

            output = self.outpng_pca_hexmap_r13co.replace("???",str(i+1))

            print("# plot " + output)
            self._plot_hexmap(
                output,
                this_x,
                this_y,
                this_c,
                "PC"+str(i+1)+" (intensity normalized to $^{13}$CO)",
                cmap=thid_cmap,
                ann=True,
                add_text=this_text,
                lim=13,
                size=3600,
                )

    def _plot_radial(
        self,
        outpng,
        r,clist,
        title,
        size=100,
        ylabel=None,
        xlim=None,
        ylim=None,
        ann=None,
        fvalue=1/3.0,
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
        grid="both",
        xlim=xlim,
        ylim=ylim,
        xlabel="Distance (arcsec)",
        ylabel=ylabel,
        adjust=[0.176,0.813,0.10,0.93],
        )

        zorder = 1e9
        # plot data
        for i in range(len(clist)):
            this_c = clist[i]
            this_zorder = zorder / float(i+1)

            # constrain data
            this_c[np.isnan(this_c)] = -100
            this_c[np.isinf(this_c)] = -100
            cut    = np.where((this_c!=-100) & (this_c!=0))
            this_c = this_c[cut]
            this_r = r[cut]

            # plot data
            if i==0:
                color = "tomato"
            elif i==1:
                color = "deepskyblue"

            # plot LOWESS
            order  = np.argsort(this_r)
            y_sm, y_std = lowess(this_r, this_c, f=fvalue)
            ax.plot(this_r[order], y_sm[order], color=color, lw=8, zorder=this_zorder)
            plt.fill_between(this_r[order], y_sm[order] - y_std[order],
                y_sm[order] + y_std[order], lw=0, color=color, alpha=0.5, zorder=this_zorder)

        # text
        ax.text(0.03, 0.93, title, color="black", transform=ax.transAxes, weight="bold", fontsize=24)

        # annotation
        if ann==1:
            ax.text(0.97, 0.93, "Bicone", color="tomato", transform=ax.transAxes, fontsize=24, ha="right", weight="bold")
            ax.text(0.97, 0.87, "Disk", color="deepskyblue", transform=ax.transAxes, fontsize=24, ha="right", weight="bold")
        if ann==2:
            ax.text(0.97, 0.93, "CN(1$_{3/2}$-0$_{1/2}$)/HCN(1-0)", color=cm.rainbow(0/float(len(clist)-1)), transform=ax.transAxes, fontsize=24, ha="right")
            ax.text(0.97, 0.87, "HNC(1-0)/HCN(1-0)", color=cm.rainbow(1/float(len(clist)-1)), transform=ax.transAxes, fontsize=24, ha="right")
            ax.text(0.97, 0.81, "CCH(1-0)/HCN(1-0)", color=cm.rainbow(2/float(len(clist)-1)), transform=ax.transAxes, fontsize=24, ha="right")

        # save
        os.system("rm -rf " + outpng)
        plt.savefig(outpng, dpi=300)

    def _get_bicone_radial(
        self,
        name,
        cone="in",
        ):
        """
        """

        offset = 10
        angle1 = -15 - offset
        angle2 = -115 + offset
        angle3 = 165 - offset
        angle4 = 65 + offset

        # get table
        header,data_mom0,_,x,y,r = self._read_table(self.table_hex_obs)
        theta_deg  =  np.degrees(np.arctan2(x, y))

        # get line data
        line_index = np.where(header==name)
        data_line  = np.array(data_mom0[:,line_index].flatten())
        data_line[np.isinf(data_line)] = 0
        data_line[np.isnan(data_line)] = 0
        data_line  = np.where(r<=self.r_sbr_as,data_line,0)

        # get line bicone
        line_zc    = np.where(r<1,data_line,0)
        line_zn    = np.where((theta_deg>=angle1)&(theta_deg<angle4)&(r<self.r_sbr_as),data_line,0)
        line_zs    = np.where((theta_deg>=angle3)&(r<self.r_sbr_as),data_line,0)
        line_zs    = np.where((theta_deg<angle2)&(r<self.r_sbr_as),data_line,line_zs)

        if cone=="in":
            data_line = np.log10(line_zn + line_zs + line_zc)
        elif cone=="out":
            data_line = np.log10(data_line - line_zn - line_zs)

        return data_line, r

###################
# end of ToolsPCA #
###################
