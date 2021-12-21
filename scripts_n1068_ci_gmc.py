"""
Python class for the NGC 1068 CI-GMC project.

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:

usage:
> import os
> from scripts_n1068_ci_gmc import ToolsCIGMC as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_n1068_ci_gmc/key_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_n1068_ci_gmc/key_figures.txt",
>     )
>
> # main
> tl.run_ngc1068_cigmc(
>     # analysis
>     do_prepare             = True,
>     # plot
>     # supplement
>     )
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To

references:
https://github.com/PhangsTeam/pycprops
https://qiita.com/Shinji_Fujita/items/7463038f70401040aedc

history:
2021-12-05   created (in Sinkansen!)
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_tasks import *
from mycasa_plots import *

############
# ToolsPCA #
############
class ToolsCIGMC():
    """
    Class for the NGC 1068 CI-GMC project.
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

        self.cube_cn10h   = self.dir_raw + self._read_key("cube_cn10h")
        self.cube_hcop10  = self.dir_raw + self._read_key("cube_hcop10")
        self.cube_hcn10   = self.dir_raw + self._read_key("cube_hcn10")
        self.cube_co10    = self.dir_raw + self._read_key("cube_co10")
        self.cube_ci10    = self.dir_raw + self._read_key("cube_ci10")

        self.ncube_cn10h  = self.dir_raw + self._read_key("ncube_cn10h")
        self.ncube_hcop10 = self.dir_raw + self._read_key("ncube_hcop10")
        self.ncube_hcn10  = self.dir_raw + self._read_key("ncube_hcn10")
        self.ncube_co10   = self.dir_raw + self._read_key("ncube_co10")
        self.ncube_ci10   = self.dir_raw + self._read_key("ncube_ci10")

        self.mom0_cn10h   = self.dir_raw + self._read_key("mom0_cn10h")
        self.mom0_hcop10  = self.dir_raw + self._read_key("mom0_hcop10")
        self.mom0_hcn10   = self.dir_raw + self._read_key("mom0_hcn10")
        self.mom0_co10    = self.dir_raw + self._read_key("mom0_co10")
        self.mom0_ci10    = self.dir_raw + self._read_key("mom0_ci10")

    def _set_output_fits(self):
        """
        """

        print("TBE.")

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

        # output fits
        self.cprops_hcn10 = self.dir_ready + self._read_key("cprops_hcn10")
        self.cprops_co10  = self.dir_ready + self._read_key("cprops_co10")
        self.cprops_ci10  = self.dir_ready + self._read_key("cprops_ci10")

        # output txt and png
        print("TBE.")

        # final
        print("TBE.")

    #####################
    # run_ngc1068_cigmc #
    #####################

    def run_ngc1068_cigmc(
        self,
        # analysis
        do_prepare        = False,
        do_cprops         = False,
        # plot figures in paper
        plot_stats_cprops = False,
        # supplement
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        # analysis
        if do_prepare==True:
            self.do_align()

        if do_cprops==True:
            print("### conda activate cprops")
            print("### python mypython_cprops.py")
            print("")
            print("import os, sys, pycprops")
            print("import astropy.units as u")
            print("from astropy.io import fits")
            print("import numpy as np")
            print("import matplotlib.pyplot as plt")
            print("")
            print("# https://qiita.com/Shinji_Fujita/items/7463038f70401040aedc")
            print("")
            print("# input")
            print("cube_hcn10 = '../data_ready/ngc1068_b3_12m_hcn10_0p8as.regrid.fits'")
            print("cube_co10  = '../data_ready/ngc1068_b3_12m+7m_co10_0p8as.regrid.fits'")
            print("cube_ci10  = '../data_ready/ngc1068_b8_12m+7m_ci10.fits'")
            print("d          = 13.97 * 1e6 * u.pc")
            print("")
            print("# output")
            print("cprops_hcn10 = '../data_ready/ngc1068_hcn10_cprops.fits'")
            print("cprops_co10  = '../data_ready/ngc1068_co10_cprops.fits'")
            print("cprops_ci10  = '../data_ready/ngc1068_ci10_cprops.fits'")
            print("")
            print("# run (repeat by lines)")
            print("cubefile = cube_hcn10")
            print("outfile  = cprops_hcn10")
            print("mask     = cubefile")
            print("")
            print("pycprops.fits2props(")
            print("    cubefile,")
            print("    mask_file=mask,")
            print("    distance=d,")
            print("    asgnname=cubefile[:-5]+'.asgn.fits',")
            print("    propsname=cubefile[:-5]+'.props.fits',")
            print("    )")
            print("os.system('mv ' + cubefile[:-5]+'.props.fits' + ' ' + outfile)")

        if plot_stats_cprops==True:
            self.plot_stats_cprops()

    ####################
    # immagick_figures #
    ####################

    def immagick_figures(
        self,
        delin=False,
        ):
        """
        """

        """
        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outpng_pca_hexmap.replace("???","1"),taskname)

        print("#########################")
        print("# create final_pca_mom0 #")
        print("#########################")

        combine_three_png(
            self.outpng_pca_scatter,
            self.outpng_pca_hexmap.replace("???","1"),
            self.outpng_pca_hexmap.replace("???","2"),
            self.final_pca_mom0,
            self.box_map,
            self.box_map,
            self.box_map,
            delin=delin,
            )
        """

    #####################
    # plot_stats_cprops #
    #####################

    def plot_stats_cprops(
        self,
        delin=False,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cprops_hcn10,taskname)

        # import fits table
        f = pyfits.open(self.cprops_hcn10)
        tb_hcn10 = f[1].data
        f = pyfits.open(self.cprops_co10)
        tb_co10 = f[1].data
        f = pyfits.open(self.cprops_ci10)
        tb_ci10 = f[1].data

        # extract tag
        cnum_hcn10 = tb_hcn10["CLOUDNUM"]
        cnum_co10  = tb_co10["CLOUDNUM"]
        cnum_ci10  = tb_ci10["CLOUDNUM"]

        ra_hcn10 = tb_hcn10["XCTR_DEG"]
        ra_co10  = tb_co10["XCTR_DEG"]
        ra_ci10  = tb_ci10["XCTR_DEG"]

        dec_hcn10 = tb_hcn10["YCTR_DEG"]
        dec_co10  = tb_co10["YCTR_DEG"]
        dec_ci10  = tb_ci10["YCTR_DEG"]

        v_hcn10 = tb_hcn10["VCTR_KMS"]
        v_co10  = tb_co10["VCTR_KMS"]
        v_ci10  = tb_ci10["VCTR_KMS"]

        r_pc_hcn10 = tb_hcn10["RAD_PC"]
        r_pc_co10  = tb_co10["RAD_PC"]
        r_pc_ci10  = tb_ci10["RAD_PC"]

        sigma_hcn10 = tb_hcn10["SIGV_KMS"]
        sigma_co10  = tb_co10["SIGV_KMS"]
        sigma_ci10  = tb_ci10["SIGV_KMS"]

        flux_hcn10 = tb_hcn10["FLUX_KKMS_PC2"]
        flux_co10  = tb_co10["FLUX_KKMS_PC2"]
        flux_ci10  = tb_ci10["FLUX_KKMS_PC2"]

        mvir_hcn10 = tb_hcn10["MVIR_MSUN"]
        mvir_co10  = tb_co10["MVIR_MSUN"]
        mvir_ci10  = tb_ci10["MVIR_MSUN"]

        snr_hcn10 = tb_hcn10["S2N"]
        snr_co10  = tb_co10["S2N"]
        snr_ci10  = tb_ci10["S2N"]

        # plot

    ############
    # do_align #
    ############

    def do_align(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cube_ci10,taskname)

        self.cn10h_ready  = self.dir_ready + self._read_key("cube_cn10h")[:-5] + ".regrid.fits"
        self.hcop10_ready = self.dir_ready + self._read_key("cube_hcop10")[:-5] + ".regrid.fits"
        self.hcn10_ready  = self.dir_ready + self._read_key("cube_hcn10")[:-5] + ".regrid.fits"
        self.co10_ready   = self.dir_ready + self._read_key("cube_co10")[:-5] + ".regrid.fits"
        self.ci10_ready   = self.dir_ready + self._read_key("cube_ci10")
        template          = "template.image"

        # get restfreq
        restf_cn10h  = imhead(self.cube_cn10h,mode="list")["restfreq"][0]
        restf_hcop10 = imhead(self.cube_hcop10,mode="list")["restfreq"][0]
        restf_hcn10  = imhead(self.cube_hcn10,mode="list")["restfreq"][0]
        restf_co10   = imhead(self.cube_co10,mode="list")["restfreq"][0]
        restf_ci10   = imhead(self.cube_ci10,mode="list")["restfreq"][0]

        # regrid to ci10 cube 98,71,179,151
        run_importfits(self.cube_ci10,template+"2")
        imsubimage(template+"2",template,box="98,71,179,151")
        run_imregrid(self.cube_cn10h,template,self.cn10h_ready+".image",axes=[0,1])
        run_imregrid(self.cube_hcop10,template,self.hcop10_ready+".image",axes=[0,1])
        run_imregrid(self.cube_hcn10,template,self.hcn10_ready+".image",axes=[0,1])
        run_imregrid(self.cube_co10,template,self.co10_ready+".image",axes=[0,1])
        os.system("rm -rf " + template + " " + template + "2")

        # to fits
        run_exportfits(self.cn10h_ready+".image",self.cn10h_ready+"2",delin=True,velocity=True)
        run_exportfits(self.hcop10_ready+".image",self.hcop10_ready+"2",delin=True,velocity=True)
        run_exportfits(self.hcn10_ready+".image",self.hcn10_ready+"2",delin=True,velocity=True)
        run_exportfits(self.co10_ready+".image",self.co10_ready+"2",delin=True,velocity=True)
        os.system("cp -r " + self.cube_ci10 + " " + self.ci10_ready + "2")

        # change header to CPROPS format
        hdu = fits.open(self.cn10h_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.cn10h_ready, overwrite=True)
        os.system("rm -rf " + self.cn10h_ready + "2")
        
        hdu = fits.open(self.hcop10_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.hcop10_ready, overwrite=True)
        os.system("rm -rf " + self.hcop10_ready + "2")

        hdu = fits.open(self.hcn10_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.hcn10_ready, overwrite=True)
        os.system("rm -rf " + self.hcn10_ready + "2")

        hdu = fits.open(self.co10_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_co10
        fits.PrimaryHDU(d, h).writeto(self.co10_ready, overwrite=True)
        os.system("rm -rf " + self.co10_ready + "2")

        hdu = fits.open(self.ci10_ready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_ci10
        fits.PrimaryHDU(d, h).writeto(self.ci10_ready, overwrite=True)
        os.system("rm -rf " + self.ci10_ready + "2")

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

#####################
# end of ToolsCIGMC #
#####################
