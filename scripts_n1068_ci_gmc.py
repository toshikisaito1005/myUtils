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
        self.cprops_cn10h  = self.dir_ready + self._read_key("cprops_cn10h")
        self.cprops_hcop10 = self.dir_ready + self._read_key("cprops_hcop10")
        self.cprops_hcn10  = self.dir_ready + self._read_key("cprops_hcn10")
        self.cprops_co10   = self.dir_ready + self._read_key("cprops_co10")
        self.cprops_ci10   = self.dir_ready + self._read_key("cprops_ci10")

        # output txt and png
        self.outpng_cprops_cn10h  = self.dir_products + self._read_key("outpng_cprops_cn10h")
        self.outpng_cprops_hcop10 = self.dir_products + self._read_key("outpng_cprops_hcop10")
        self.outpng_cprops_hcn10  = self.dir_products + self._read_key("outpng_cprops_hcn10")
        self.outpng_cprops_co10   = self.dir_products + self._read_key("outpng_cprops_co10")
        self.outpng_cprops_ci10   = self.dir_products + self._read_key("outpng_cprops_ci10")

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
            print("cube_cn10h  = '../data_ready/ngc1068_b3_12m_cn10h_0p8as.regrid.fits'")
            print("cube_hcop10 = '../data_ready/ngc1068_b3_12m_hcop10_0p8as.regrid.fits'")
            print("cube_hcn10  = '../data_ready/ngc1068_b3_12m_hcn10_0p8as.regrid.fits'")
            print("cube_co10   = '../data_ready/ngc1068_b3_12m+7m_co10_0p8as.regrid.fits'")
            print("cube_ci10   = '../data_ready/ngc1068_b8_12m+7m_ci10.fits'")
            print("d           = 13.97 * 1e6 * u.pc")
            print("cubes       = [cube_cn10h,cube_hcop10,cube_hcn10,cube_co10,cube_ci10]")
            print("")
            print("# output")
            print("cprops_cn10h  = '../data_ready/ngc1068_cn10h_cprops.fits'")
            print("cprops_hcop10 = '../data_ready/ngc1068_hcop10_cprops.fits'")
            print("cprops_hcn10  = '../data_ready/ngc1068_hcn10_cprops.fits'")
            print("cprops_co10   = '../data_ready/ngc1068_co10_cprops.fits'")
            print("cprops_ci10   = '../data_ready/ngc1068_ci10_cprops.fits'")
            print("outfiles = [cprops_cn10h,cprops_hcop10,cprops_hcn10,cprops_co10,cprops_ci10]")
            print("")
            print("# run (repeat by lines)")
            print("cubefile = cube_hcn10")
            print("outfile  = cprops_hcn10")
            print("mask     = cubefile")
            print("")
            print("# run")
            print("for i in range(len(cubes)):")
            print("    cubefile = cubes[i]")
            print("    outfile  = outfiles[i]")
            print("    mask     = cubefile")
            print("    pycprops.fits2props(")
            print("        cubefile,")
            print("        mask_file=mask,")
            print("        distance=d,")
            print("        asgnname=cubefile[:-5]+'.asgn.fits',")
            print("        propsname=cubefile[:-5]+'.props.fits',")
            print("        )")
            print("    os.system('mv ' + cubefile[:-5]+'.props.fits' + ' ' + outfile)")

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
        f = pyfits.open(self.cprops_cn10h)
        tb_cn10h = f[1].data

        f = pyfits.open(self.cprops_hcop10)
        tb_hcop10 = f[1].data

        f = pyfits.open(self.cprops_hcn10)
        tb_hcn10 = f[1].data

        f = pyfits.open(self.cprops_co10)
        tb_co10 = f[1].data

        f = pyfits.open(self.cprops_ci10)
        tb_ci10 = f[1].data


        # extract tag
        self._plot_cprops_map(self.mom0_cn10h,tb_cn10h,"CN(1-0)h",self.outpng_cprops_cn10h)
        self._plot_cprops_map(self.mom0_hcn10,tb_hcn10,"HCN(1-0)",self.outpng_cprops_hcn10)

    ####################
    # _plot_cprops_map #
    ####################

    def _plot_cprops_map(
        self,
        imagename,
        this_tb,
        linename,
        outpng,
        ):
        """
        # CLOUDNUM
        # XCTR_DEG
        # YCTR_DEG
        # VCTR_KMS
        # RAD_PC
        # SIGV_KMS
        # FLUX_KKMS_PC2
        # MVIR_MSUN
        # S2N
        """

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        myfig_fits2png(
            imagename,
            outpng,
            imsize_as = 26.0,
            ra_cnt    = str(self.ra_agn) + "deg",
            dec_cnt   = str(self.dec_agn) + "deg",
            numann    = "ci-gmc",
            txtfiles  = this_tb,
            set_title = linename + " Cloud Catalog",
            scalebar  = scalebar,
            label_scalebar = label_scalebar,
            )

    ###################
    # _plot_all_param #
    ###################

    def _plot_all_param(
        self,
        this_tb,
        linename,
        outpng_header,
        snr=4,
        ):
        """
        # CLOUDNUM
        # XCTR_DEG
        # YCTR_DEG
        # VCTR_KMS
        # RAD_PC
        # SIGV_KMS
        # FLUX_KKMS_PC2
        # MVIR_MSUN
        # S2N
        """

        radlimit = 72.*0.8/2.

        # FLUX_KKMS_PC2
        params = ["FLUX_KKMS_PC2","RAD_PC","SIGV_KMS"]
        footers = ["flux","radius","disp"]
        for i in range(len(params)):
            this_param  = params[i]
            this_footer = footers[i]
            cut         = np.where((this_tb["S2N"]>=snr) & (this_tb["RAD_PC"]>=radlimit) & (this_tb["RAD_PC"]<=radlimit*10))
            this_x      = (this_tb["XCTR_DEG"][cut] - self.ra_agn) * 3600.  
            this_y      = (this_tb["YCTR_DEG"][cut] - self.dec_agn) * 3600.  
            this_c      = this_tb[this_param][cut]
            this_outpng = outpng_header.replace(".png","_"+this_footer+".png")

            if this_param=="FLUX_KKMS_PC2":
                this_c = np.log(this_c)

            self._plot_cpropsmap(
                this_outpng,
                this_x,
                this_y,
                this_c,
                linename + " (" + this_param + ")",
                title_cbar="(K km s$^{-1}$)",
                cmap="rainbow",
                add_text=False,
                label="",
                )

    ############
    # do_align #
    ############

    def do_align(
        self,
        ):
        """
        add ncube alignment
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.cube_ci10,taskname)

        self.cn10h_ready  = self.dir_ready + self._read_key("cube_cn10h")[:-5] + ".regrid.fits"
        self.hcop10_ready = self.dir_ready + self._read_key("cube_hcop10")[:-5] + ".regrid.fits"
        self.hcn10_ready  = self.dir_ready + self._read_key("cube_hcn10")[:-5] + ".regrid.fits"
        self.co10_ready   = self.dir_ready + self._read_key("cube_co10")[:-5] + ".regrid.fits"
        self.ci10_ready   = self.dir_ready + self._read_key("cube_ci10")
        template          = "template.image"

        self.cn10h_nready  = self.dir_ready + self._read_key("ncube_cn10h")[:-5] + ".regrid.fits"
        self.hcop10_nready = self.dir_ready + self._read_key("ncube_hcop10")[:-5] + ".regrid.fits"
        self.hcn10_nready  = self.dir_ready + self._read_key("ncube_hcn10")[:-5] + ".regrid.fits"
        self.co10_nready   = self.dir_ready + self._read_key("ncube_co10")[:-5] + ".regrid.fits"
        self.ci10_nready   = self.dir_ready + self._read_key("ncube_ci10")

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

        run_imregrid(self.ncube_cn10h,template,self.cn10h_nready+".image",axes=[0,1])
        run_imregrid(self.ncube_hcop10,template,self.hcop10_nready+".image",axes=[0,1])
        run_imregrid(self.ncube_hcn10,template,self.hcn10_nready+".image",axes=[0,1])
        run_imregrid(self.ncube_co10,template,self.co10_nready+".image",axes=[0,1])
        os.system("rm -rf " + template + " " + template + "2")

        # to fits
        run_exportfits(self.cn10h_nready+".image",self.cn10h_nready+"2",delin=True,velocity=True)
        run_exportfits(self.hcop10_nready+".image",self.hcop10_nready+"2",delin=True,velocity=True)
        run_exportfits(self.hcn10_nready+".image",self.hcn10_nready+"2",delin=True,velocity=True)
        run_exportfits(self.co10_nready+".image",self.co10_nready+"2",delin=True,velocity=True)
        os.system("cp -r " + self.ncube_ci10 + " " + self.ci10_nready + "2")

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

        #
        hdu = fits.open(self.cn10h_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.cn10h_nready, overwrite=True)
        os.system("rm -rf " + self.cn10h_nready + "2")
        
        hdu = fits.open(self.hcop10_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.hcop10_nready, overwrite=True)
        os.system("rm -rf " + self.hcop10_nready + "2")

        hdu = fits.open(self.hcn10_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_hcn10
        fits.PrimaryHDU(d, h).writeto(self.hcn10_nready, overwrite=True)
        os.system("rm -rf " + self.hcn10_nready + "2")

        hdu = fits.open(self.co10_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_co10
        fits.PrimaryHDU(d, h).writeto(self.co10_nready, overwrite=True)
        os.system("rm -rf " + self.co10_nready + "2")

        hdu = fits.open(self.ci10_nready+"2")[0]
        d, h = hdu.data, hdu.header
        h["CTYPE3"] = "VELOCITY"
        h["RESTFREQ"] = restf_ci10
        fits.PrimaryHDU(d, h).writeto(self.ci10_nready, overwrite=True)
        os.system("rm -rf " + self.ci10_nready + "2")

    ###################
    # _plot_cpropsmap #
    ###################

    def _plot_cpropsmap(
        self,
        outpng,
        x,y,c,
        title,
        title_cbar="(K km s$^{-1}$)",
        cmap="rainbow",
        plot_cbar=True,
        ann=True,
        lim=13.0,
        size=100,
        add_text=False,
        label="",
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
        xlim=[lim, -lim],
        ylim=[-lim, lim],
        xlabel="R.A. offset (arcsec)",
        ylabel="Decl. offset (arcsec)",
        adjust=[0.10,0.99,0.10,0.93],
        )
        ax.set_aspect('equal', adjustable='box')

        # plot
        im = ax.scatter(x, y, s=size, c=c, cmap=cmap, marker="o", linewidths=0)

        # cbar
        cbar = plt.colorbar(im)
        if plot_cbar==True:
            cax  = fig.add_axes([0.19, 0.12, 0.025, 0.35])
            fig.colorbar(im, cax=cax).set_label(label)

        # scale bar
        bar = 100 / self.scale_pc
        ax.plot([-10,-10+bar],[-10,-10],"-",color="black",lw=4)
        ax.text(-10, -10.5, "100 pc",
                horizontalalignment="right", verticalalignment="top")

        # text
        ax.text(0.03, 0.93, title, color="black", transform=ax.transAxes, weight="bold", fontsize=24)

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

#####################
# end of ToolsCIGMC #
#####################
