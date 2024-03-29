"""
Python class for the PHANGS-(U)LIRG project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:


usage:
> import os
> from scripts_phangs_ulirg import ToolsULIRG as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_projects/galkey_ulirg.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_projects/key_phangs_ulirg.txt",
>     )
>
> # main
> tl.run_phangs_ulirg(
>     do_all           = True,
>     # analysis
>     do_prepare       = True,
>     # plot
>     plot_showcase    = True,
>     # supplement
>     )
>
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                To

history:
2017-11-01   created
2022-07-28   constructed align_cubes
2022-07-29   constructed multismooth
Toshiki Saito@NAOJ
"""

import os, sys, glob
import numpy as np
from scipy.stats import gaussian_kde

from mycasa_sampling import *
from mycasa_tasks import *
from mycasa_plots import *

##############
# ToolsULIRG #
##############
class ToolsULIRG():
    """
    Class for the PHANGS-(U)LIRG project.
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
            self.modname = "ToolsR21."
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
        self.dir_products = self.dir_proj + self._read_key("dir_products")
        self.dir_final    = self.dir_proj + self._read_key("dir_final")
        self._create_dir(self.dir_ready)
        self._create_dir(self.dir_products)
        self._create_dir(self.dir_final)

    def _set_input_fits(self):
        """
        """

        # (U)LIRG data
        this = self.dir_raw + self._read_key("mom0_150pc")
        self.list_mom0_150pc = glob.glob(this.replace("XXX","*"))
        self.list_mom0_150pc.sort()

        this = self.dir_raw + self._read_key("emom0_150pc")
        self.list_emom0_150pc = glob.glob(this.replace("XXX","*"))
        self.list_emom0_150pc.sort()

        this = self.dir_raw + self._read_key("mom1_150pc")
        self.list_mom1_150pc = glob.glob(this.replace("XXX","*"))
        self.list_mom1_150pc.sort()

        this = self.dir_raw + self._read_key("emom1_150pc")
        self.list_emom1_150pc = glob.glob(this.replace("XXX","*"))
        self.list_emom1_150pc.sort()

        this = self.dir_raw + self._read_key("mom2_150pc")
        self.list_mom2_150pc = glob.glob(this.replace("XXX","*"))
        self.list_mom2_150pc.sort()

        this = self.dir_raw + self._read_key("emom2_150pc")
        self.list_emom2_150pc = glob.glob(this.replace("XXX","*"))
        self.list_emom2_150pc.sort()

        # PHANGS catalog
        self.phangs_catalog = self.dir_raw + self._read_key("Sun22_phangs_catalog")
        self.Sun22_phangs_150pc = self.dir_raw + self._read_key("Sun22_phangs_150pc")

        this = self.dir_raw + self._read_key("mom0_phangs_150pc")
        self.list_mom0_phangs_150pc = glob.glob(this.replace("XXX","*"))
        self.list_mom0_phangs_150pc.sort()

        this = self.dir_raw + self._read_key("emom0_phangs_150pc")
        self.list_emom0_phangs_150pc = glob.glob(this.replace("XXX","*"))
        self.list_emom0_phangs_150pc.sort()

        this = self.dir_raw + self._read_key("mom2_phangs_150pc")
        self.list_mom2_phangs_150pc = glob.glob(this.replace("XXX","*"))
        self.list_mom2_phangs_150pc.sort()

        this = self.dir_raw + self._read_key("emom2_phangs_150pc")
        self.list_emom2_phangs_150pc = glob.glob(this.replace("XXX","*"))
        self.list_emom2_phangs_150pc.sort()

    def _set_output_fits(self):
        """
        """

    def _set_input_param(self):
        """
        """

        self.snr = 4.0

    def _set_output_txt_png(self):
        """
        """

        self.outpng_mom0_vs_mom2  = self.dir_products + self._read_key("outpng_mom0_vs_mom2")
        self.outpng_pturb_vs_avir = self.dir_products + self._read_key("outpng_pturb_vs_avir")

    ####################
    # run_phangs_ulirg #
    ####################

    def run_phangs_ulirg(
        self,
        do_all        = False,
        # analysis
        do_prepare    = False,
        # plot figures in paper
        plot_showcase = False,
        plot_scatter  = False,
        # supplement
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        if do_all==True:
            do_prepare = True

        # analysis
        if do_prepare==True:
            self.align_cubes()

        # plot figures in paper
        if plot_showcase==True:
            self.showcase()

        if plot_scatter==True:
            self.plot_mom0_vs_mom2()

    #####################
    # plot_mom0_vs_mom2 #
    #####################

    def plot_mom0_vs_mom2(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.list_mom0_150pc[0],taskname)

        x_lirg = []
        y_lirg = []
        for i in range(len(self.list_mom0_150pc)):
            this_mom0,_  = imval_all(self.list_mom0_150pc[i])
            this_mom2,_  = imval_all(self.list_mom2_150pc[i])
            this_emom0,_ = imval_all(self.list_emom0_150pc[i])
            this_emom2,_ = imval_all(self.list_emom2_150pc[i])
            name         = self.list_mom0_150pc[i].split("/")[-1].split("_")[0]

            cut = np.where((this_mom0["data"]>=this_emom0["data"]*self.snr) & (this_mom2["data"]>=this_emom2["data"]*self.snr))
            this_mom0 = this_mom0["data"][cut]
            this_mom2 = this_mom2["data"][cut]

            x_lirg.append(np.log10(np.mean(this_mom0)))
            y_lirg.append(np.log10(np.mean(this_mom2)))
            print(name, str(np.round(np.log10(np.mean(this_mom0)),2)), str(np.round(np.log10(np.mean(this_mom2)),2)))

        x_phangs = []
        y_phangs = []
        data = np.loadtxt(self.Sun22_phangs_150pc, dtype="unicode")
        list_galname = np.unique(data[:,0])
        list_name    = data[:,0]
        list_mom0    = data[:,7].astype(float)
        list_mom2    = data[:,8].astype(float)

        for i in range(len(list_galname)):
            this_name = list_galname[i]
            this_mom0 = list_mom0[list_name==this_name]
            this_mom2 = list_mom2[list_name==this_name]

            x_phangs.append(np.log10(np.mean(this_mom0)))
            y_phangs.append(np.log10(np.mean(this_mom2)))
            print(this_name, str(np.round(np.log10(np.mean(this_mom0)),2)), str(np.round(np.log10(np.mean(this_mom2)),2)))

        ########
        # plot #
        ########
        fig = plt.figure(figsize=(15,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.10,0.95,0.10,0.95] # [0.215,0.83,0.10,0.90]
        myax_set(
            ax1,
            None,
            [0.8,3.5],
            [0.3,2.2],
            None,
            "<$\Sigma$$_{\mathrm{H_2,150pc}}$> ($M_{\odot}$ pc$^{-2}$)",
            "<$\sigma$$_{\mathrm{v,150pc}}$> (km s$^{-1}$)",
            adjust=ad,
            )

        ax1.scatter(x_lirg, y_lirg, c="tomato", lw=0, s=40, zorder=1e9)
        ax1.scatter(x_phangs, y_phangs, c="deepskyblue", lw=0, s=40, zorder=1e9)

        os.system("rm -rf " + self.outpng_mom0_vs_mom2)
        plt.savefig(self.outpng_mom0_vs_mom2, dpi=self.fig_dpi)

        ########
        # plot #
        ########
        x_lirg   = 10**np.array(x_lirg)
        y_lirg   = 10**np.array(y_lirg)
        x_phangs = 10**np.array(x_phangs)
        y_phangs = 10**np.array(y_phangs)
        x2_lirg   = np.log10( 61.3 * x_lirg * y_lirg**2 * (75/40.)**-1 )
        y2_lirg   = np.log10( 5.77 * y_lirg**2 * x_lirg**-1 * (75/40.)**-1 )
        x2_phangs = np.log10( 61.3 * x_phangs * y_phangs**2 * (75/40.)**-1 )
        y2_phangs = np.log10( 5.77 * y_phangs**2 * x_phangs**-1 * (75/40.)**-1 )

        fig = plt.figure(figsize=(15,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad  = [0.10,0.95,0.10,0.95] # [0.215,0.83,0.10,0.90]
        myax_set(
            ax1,
            None,
            [3.5,9.0],
            [0.0,1.6],
            None,
            "log$_{10}$ <$P_{\mathrm{turb,150pc}}/k_{\mathrm{B}}$> (K cm$^{-3}$) $\propto$ log$_{10}$ <$\Sigma\sigma^2$>",
            r"log$_{10}$ <$\alpha_{\mathrm{vir,150pc}}$> $\propto$ log$_{10}$ <$\sigma^2/\Sigma$>",
            adjust=ad,
            )

        ax1.scatter(x2_lirg, y2_lirg, c="tomato", lw=0, s=40, zorder=1e9)
        ax1.scatter(x2_phangs, y2_phangs, c="deepskyblue", lw=0, s=40, zorder=1e9)

        os.system("rm -rf " + self.outpng_pturb_vs_avir)
        plt.savefig(self.outpng_pturb_vs_avir, dpi=self.fig_dpi)

    ############
    # showcase #
    ############

    def showcase(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.list_mom0_150pc[0],taskname)

        # (U)LIRG
        for i in range(len(self.list_mom0_150pc)):
            this_mom0    = self.list_mom0_150pc[i]
            this_mom1    = self.list_mom1_150pc[i]
            this_emom0   = self.list_emom0_150pc[i]
            this_outfile = this_mom0.replace("data_raw","products_png").replace(".fits",".png")
            self._one_showcase(
                this_mom1,
                this_mom0,
                this_emom0,
                "(km s$^{-1}$)",
                this_outfile,
                color="rainbow",
                )

        """
        # PHANGS
        galaxy_cat = np.loadtxt(self.phangs_catalog, dtype="unicode")

        for i in range(len(self.list_mom0_phangs_150pc)):
            this_mom0  = self.list_mom0_phangs_150pc[i]
            this_emom0 = self.list_emom0_phangs_150pc[i]
            this_name  = this_mom0.split("/")[-1].split("_")[0]
            index      = np.where(np.array(galaxy_cat[:,0])==this_name.replace("ngc","NGC").replace("ic","IC").replace("a","A"))[0]

            if index:
                dist         = galaxy_cat[index,3].astype(float)[0]
                this_outfile = this_mom0.replace("data_raw","products_png").replace("phangs_v4p0_release/","phangs_").replace(".fits",".png")
                self._one_showcase(
                    this_mom0,
                    this_mom0,
                    this_emom0,
                    "(K km s$^{-1}$)",
                    this_outfile,
                    color="Blues",
                    dist=dist,
                    )
        """

    #################
    # _one_showcase #
    #################

    def _one_showcase(
        self,
        imcolor,
        imcontour1,
        imcolornoise,
        label_cbar,
        outfile,
        color="Reds",
        dist=None,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.list_mom0_150pc[0],taskname)

        # get header
        header = imhead(imcolor,mode="list")
        beam   = header["beammajor"]["value"]
        ra     = str(header["crval1"] * 180 / np.pi)
        dec    = str(header["crval2"] * 180 / np.pi)
        title  = imcolor.split("/")[-1].split("_")[0]
        label_scalebar = "500 pc"

        clim = None

        if title=="arp220":
            clim=[4950, 5600]
        elif title=="eso297g011":
            clim=[4960, 5150]
        elif title=="eso319":
            clim=[4640, 4920]
        elif title=="ic4518w":
            clim=[4400, 4800]
        elif title=="iras13120":
            clim=[8800,9200]
        elif title=="ngc6240":
            clim=[6800,7400]
        elif title=="ngc3110":
            clim=[4800, 5200]
        elif title=="ngc7130":
            clim=[4700,4850]

        if dist==None:
            beam_pc = 150
            imsize = beam * 20000. / beam_pc # 20kpc size in arcsec
        else:
            beam_pc = dist * 1000000. * np.tan(np.radians(beam/3600.))
            imsize = beam * 20000. / beam_pc # 20kpc size in arcsec

        scalebar = beam * 500. / beam_pc

        print(title, dist, beam, beam_pc)

        # achieved s/n ratio
        mom0,_  = imval_all(imcolor)
        emom0,_ = imval_all(imcolornoise)
        mom0    = mom0["data"].flatten()
        emom0   = emom0["data"].flatten()
        emom0   = emom0[mom0>0]

        rms = np.median(emom0)
        rms_norm = rms / np.nanmax(mom0)

        levels_cont1 = rms_norm * np.array([2,4,8,16,32,64,128,256]) # [0.05, 0.1, 0.2, 0.4, 0.8, 0.96]
        width_cont1  = [0.5]
        set_bg_color = "white" # cm.rainbow(0)

        # plot
        myfig_fits2png(
            imcolor=imcolor,
            outfile=outfile,
            imcontour1=imcontour1,
            imsize_as=imsize,
            ra_cnt=ra,
            dec_cnt=dec,
            levels_cont1=levels_cont1,
            width_cont1=width_cont1,
            set_title=title,
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar=label_cbar,
            set_grid=None,
            set_cmap=color,
            clim=clim,
            set_bg_color=set_bg_color,
            )

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
# end of ToolsULIRG #
#####################