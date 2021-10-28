"""
Class for the NGC 1068 CI outflow project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:
ALMA [CI](1-0)        2017.1.00586.S (PI: Takano Shuro, 12m+7m, 3-pointing)
                      processed at ADC:/home02/saitots/scripts/phangs_imaging_scripts/keys_ngc1068_b8/
                      products at ADC:/lfs02/saitots/proj_n1068_b8/derived/ngc1068_b8/
ALMA CO(1-0)          2018.1.01684.S (PI: Tosaki Tomoka, 12m+7m)
                      processed at ADC:/home02/saitots/scripts/phangs_imaging_scripts/keys_ngc1068_b3/
                      products at ADC:/lfs02/saitots/proj_n1068_b3/derived/ngc1068_b3/
imaging script        all processed by phangs pipeline v2
                      Leroy et al. 2021, ApJS, 255, 19 (https://arxiv.org/abs/2104.07739)
ancillary MUSE FITS   Mingozzi et al. 2019, A&A, 622, 146 (https://ui.adsabs.harvard.edu/abs/2019A%26A...622A.146M)
                      SIII/SII ratio map (ionization parameter), ionized gas density, AV maps
                      http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/622/A146
ancillary HST FITS    https://hla.stsci.edu/
ancillary VLA FITS    https://archive.nrao.edu/archive/archiveimage.html

usage:
> import os
> from scripts_n1068_outflow import ToolsOutflow as tools
> 
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_n1068_outflow/key_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_n1068_outflow/key_figures.txt",
>     )
> 
> tl.run_ci_outflow(
>     do_prepare    = True, # align maps to the [CI](1-0) grid, and mask by [CI] FoVs
>     do_ratio_map  = True,
>     do_compare_7m = True,
>     plot_scatters = True,
>     plot_showcase = True,
>     plot_channel  = True,
>     )
> 
> os.system("rm -rf *.last")

paper drafts:
Date         Filename                          To
2021-06-04   draft_v0_210604.zip               Takano,Nakajima,Harada

history:
2021-04-22   start project, write README
2021-05-17   start to create paper-ready figures
2021-05-26   start writing draft
2021-06-04   circulate v0 draft to the paper team
2021-06-25   move to ADC due to issues with new laptop
2021-08-06   created
2021-10-24   circulate v0.2 draft to the paper team
2021-10-26   refactored align_maps
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_tasks import *
from mycasa_sampling import *
from mycasa_plots import *

###########################
### ToolsDense
###########################
class ToolsOutflow():
    """
    Class for the NGC 1068 CI outflow project.
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

        self.fig_dpi = 200

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ToolsOutflow."
            
            # get directories
            self.dir_proj = self._read_key("dir_proj")
            self.dir_raw_co = self.dir_proj + self._read_key("dir_raw_co")
            self.dir_raw_ci = self.dir_proj + self._read_key("dir_raw_ci")
            self.dir_hst = self.dir_proj + self._read_key("dir_hst")
            self.dir_vla = self.dir_proj + self._read_key("dir_vla")
            self.dir_magnum = self.dir_proj + self._read_key("dir_magnum")
            self.dir_7m = self.dir_proj + "data_raw/b8_7m_ci10_indiv_cube/"

            self.dir_ready = self.dir_proj + self._read_key("dir_ready")
            self.dir_products = self.dir_proj + self._read_key("dir_products")
            self.dir_chan = self.dir_proj + self._read_key("dir_chan")
            self.dir_final = self.dir_proj + self._read_key("dir_final")

            self._create_dir(self.dir_ready)
            self._create_dir(self.dir_products)
            self._create_dir(self.dir_chan)
            self._create_dir(self.dir_final)

            # input maps
            self.image_co10 = self.dir_raw_co + self._read_key("image_co10")
            self.image_ci10 = self.dir_raw_ci + self._read_key("image_ci10")
            self.image_eco10 = self.dir_raw_co + self._read_key("image_eco10")
            self.image_eci10 = self.dir_raw_ci + self._read_key("image_eci10")
            self.image_siiisii = self.dir_magnum + self._read_key("image_siiisii")
            self.image_av = self.dir_magnum + self._read_key("image_av")
            self.image_oiii = self.dir_hst + self._read_key("hst_oiii")
            self.image_vla = self.dir_vla + self._read_key("vla_radio")

            self.cube_co10 = self.dir_raw_co + self._read_key("cube_co10")
            self.cube_ci10 = self.dir_raw_ci + self._read_key("cube_ci10")
            self.ncube_co10 = self.dir_raw_co + self._read_key("ncube_co10")
            self.ncube_ci10 = self.dir_raw_ci + self._read_key("ncube_ci10")

            # output maps
            self.out_map_co10 = self.dir_ready + self._read_key("out_map_co10")
            self.out_map_ci10 = self.dir_ready + self._read_key("out_map_ci10")
            self.out_map_eco10 = self.dir_ready + self._read_key("out_map_eco10")
            self.out_map_eci10 = self.dir_ready + self._read_key("out_map_eci10")
            self.out_map_cico = self.dir_ready + self._read_key("out_map_cico")
            self.out_cube_co10 = self.dir_ready + self._read_key("out_cube_co10")
            self.out_cube_ci10 = self.dir_ready + self._read_key("out_cube_ci10")
            self.out_cube_cico = self.dir_ready + self._read_key("out_cube_cico")
            self.out_ncube_co10 = self.dir_ready + self._read_key("out_ncube_co10")
            self.out_ncube_ci10 = self.dir_ready + self._read_key("out_ncube_ci10")
            self.out_map_av = self.dir_ready + self._read_key("out_map_av")
            self.out_map_oiii = self.dir_ready + self._read_key("out_map_oiii")
            self.out_map_radio = self.dir_ready + self._read_key("out_map_radio")
            self.out_map_siiisii = self.dir_ready + self._read_key("out_map_siiisii")

            self.outfits_map_co10 = self.out_map_co10.replace(".image",".fits")
            self.outfits_map_ci10 = self.out_map_ci10.replace(".image",".fits")
            self.outfits_map_eco10 = self.out_map_eco10.replace(".image",".fits")
            self.outfits_map_eci10 = self.out_map_eci10.replace(".image",".fits")
            self.outfits_map_cico = self.out_map_cico.replace(".image",".fits")
            self.outfits_cube_co10 = self.out_cube_co10.replace(".cube","_cube.fits")
            self.outfits_cube_ci10 = self.out_cube_ci10.replace(".cube","_cube.fits")
            self.outfits_cube_cico = self.out_cube_cico.replace(".cube","_cube.fits")
            self.outfits_ncube_co10 =self.out_ncube_co10.replace(".cube.noise","_cube_err.fits")
            self.outfits_ncube_ci10 =self.out_ncube_ci10.replace(".cube.noise","_cube_err.fits")
            self.outfits_map_av = self.out_map_av.replace(".image",".fits")
            self.outfits_map_oiii = self.out_map_oiii.replace(".image",".fits")
            self.outfits_map_radio = self.out_map_radio.replace(".image",".fits")
            self.outfits_map_siiisii = self.out_map_siiisii.replace(".image",".fits")

            self.outfits_cube_ci10_rebin = \
                self.out_cube_ci10.replace(".cube","_cube_rebin.fits")
            self.outfits_cube_cico_rebin = \
                self.out_cube_cico.replace(".cube","_cube_rebin.fits")
            self.outfits_ci10_outflow_mom0 = self.dir_ready + \
                self._read_key("out_ci10_outflow_mom0").replace(".image",".fits")
            self.outfits_ci10_outflow_mom1 = self.dir_ready + \
                self._read_key("out_ci10_outflow_mom1").replace(".image",".fits")

            # get ngc1068 properties
            self.ra_agn_str = self._read_key("ra_agn", "gal")
            self.ra_agn     = float(self.ra_agn_str.replace("deg",""))

            self.dec_agn_str = self._read_key("dec_agn", "gal")
            self.dec_agn     = float(self.dec_agn_str.replace("deg",""))

            self.pa        = float(self._read_key("pa", "gal"))
            self.incl      = float(self._read_key("incl", "gal"))
            self.scale_pc  = float(self._read_key("scale", "gal"))
            self.scale_kpc = float(self._read_key("scale", "gal")) / 1000.
            
            # input parameters
            self.fov_radius = float(self._read_key("fov_radius_as")) * self.scale_pc

            self.snr_cube  = float(self._read_key("snr_cube"))
            self.snr_ratio = float(self._read_key("snr_ratio"))
            self.snr_chan  = float(self._read_key("snr_chan"))

            l = self._read_key("imrebin_factor")
            self.imrebin_factor = [int(s) for s in l.split(",")]

            self.r_cnd =  float(self._read_key("r_cnd_as")) * self.scale_kpc
            self.r_sbr = float(self._read_key("r_sbr_as")) * self.scale_kpc

            l = self._read_key("chans_num")
            self.chans_num = [int(s) for s in l.split(",")]
            self.chans_text = self._read_key("chans_text").split(",")
            self.chans_color = self._read_key("chans_color").split(",")

            # model parameters
            self.model_length       = float(self._read_key("model_length"))
            self.model_pa           = float(self._read_key("model_pa"))
            self.model_incl         = float(self._read_key("model_incl"))
            self.model_theta_in     = float(self._read_key("model_theta_in"))
            self.model_theta_out    = float(self._read_key("model_theta_out"))
            self.model_maxvel_const = float(self._read_key("model_maxvel_const"))
            self.model_maxvel_decv  = float(self._read_key("model_maxvel_decv"))
            self.model_maxvel_best  = float(self._read_key("model_maxvel_best"))
            self.model_cnd_rout     = float(self._read_key("model_cnd_rout"))
            l = self._read_key("model_chanlist")
            self.model_chanlist     = [float(s) for s in l.split(",")]

            self.model_nbins        = 300 # 300
            self.model_disk_width   = 100
            self.model_pa_disk      = 286-270 # degree
            self.model_incl_disk    = 41 # degree
            self.model_r_turn       = 140 # pc
            self.model_velindex     = 0.35 # decomission

            # output txt and png
            self.outtxt_slopes_7m = self.dir_products + self._read_key("txt_slopes")
            self.outpng_slopes_7m = self.dir_products + self._read_key("png_slopes")

            self.outpng_ci_vs_co = self.dir_products + self._read_key("png_ci_vs_co")
            self.outpng_cico_vs_siiisii = \
                self.dir_products + self._read_key("png_ci_vs_co")

            self.imsize_as       = float(self._read_key("imsize_as"))
            self.outpng_map_ci   = self.dir_products + self._read_key("png_map_ci")
            self.outpng_map_co   = self.dir_products + self._read_key("png_map_co")
            self.outpng_map_cico = self.dir_products + self._read_key("png_map_cico")

            self.outpng_outflow_mom0 = \
                self.dir_products + self._read_key("png_outflow_mom0")
            self.outpng_outflow_mom1 = \
                self.dir_products + self._read_key("png_outflow_mom1")
            self.outpng_outflow_chans = \
                self.dir_products + self._read_key("png_outflow_chans")

            self.png_outflow_model = \
                self.dir_chan + self._read_key("png_outflow_model")

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

    ##################
    # run_ci_outflow #
    ##################

    def run_ci_outflow(
        self,
        do_prepare       = False, # refactored
        do_ratio_map     = False, # refactored
        plot_scatters    = False, # refactored
        plot_showcase    = False, # refactored
        plot_channel     = False, # refactored
        do_modeling      = False, # refactored (need to validate)
        # appendix
        plot_outflow_mom = False,
        # additional analysis
        do_compare_7m    = False,
        ):

        if do_prepare==True:
            self.align_maps()
            self.ci_fov_masking()

        if do_ratio_map==True:
            self.ratio_map()
            self.ratio_cube()

        if plot_scatters==True:
            self.plot_ci_vs_co()
            self.plot_cico_vs_siiisii()

        if plot_showcase==True:
            self.showcase()

        if plot_channel==True:
            self.get_outflow_channels()

        if do_modeling==True:
            self.bicone_modeling()

        # appendix
        if plot_outflow_mom==True:
            self.get_outflow_moments()
            self.plot_outflow_moments()

        # additional
        if do_compare_7m==True:
            self.compare_7m_cubes()

    ###################
    # bicone_modeling #
    ###################

    def bicone_modeling(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_cube_cico,taskname)

        ################
        # create model #
        ################
        angle_list = np.linspace(self.model_theta_out,self.model_theta_in,41)

        # constant velocity bicone
        cone_cnst = [[],[],[],[]]
        for i in range(len(angle_list)):
            this_angle = angle_list[i]

            ## geometry (X=R.A., Z=decl., Y=depth)
            nx,ny,nz,nv,sx,sy,sz,sv = self._create_3d_bicone_rel(
                self.model_length, self.model_nbins, this_angle, self.model_pa,
                self.model_incl, self.model_pa_disk, self.model_incl_disk,
                self.model_disk_width*1.5, self.scale_pc,
                clipoutflow = False,
                velmax      = self.model_maxvel_const,
                velmodel    = "const",
                velindex    = self.model_velindex,
                clipcnd     = self.model_cnd_rout,
                r_turn      = self.model_r_turn,
                )
            ## combine
            cone_cnst[0].extend(nx.flatten())
            cone_cnst[1].extend(ny.flatten())
            cone_cnst[2].extend(nz.flatten())
            cone_cnst[3].extend(nv.flatten())
            cone_cnst[0].extend(sx.flatten())
            cone_cnst[1].extend(sy.flatten())
            cone_cnst[2].extend(sz.flatten())
            cone_cnst[3].extend(sv.flatten())

        # decelerating velocity bicone
        cone_decv = [[],[],[],[]]
        for i in range(len(angle_list)):
            this_angle = angle_list[i]
            ## geometry (X=R.A., Z=decl., Y=depth)
            nx,ny,nz,nv,sx,sy,sz,sv = self._create_3d_bicone_rel(
                self.model_length, self.model_nbins, this_angle, self.model_pa,
                self.model_incl, self.model_pa_disk, self.model_incl_disk,
                self.model_disk_width*1.5, self.scale_pc,
                clipoutflow = False,
                velmax      = self.model_maxvel_decv,
                velmodel    = "decelerate",
                velindex    = self.model_velindex,
                clipcnd     = self.model_cnd_rout,
                r_turn      = self.model_r_turn,
                )
            ## combine
            cone_decv[0].extend(nx.flatten())
            cone_decv[1].extend(ny.flatten())
            cone_decv[2].extend(nz.flatten())
            cone_decv[3].extend(nv.flatten())
            cone_decv[0].extend(sx.flatten())
            cone_decv[1].extend(sy.flatten())
            cone_decv[2].extend(sz.flatten())
            cone_decv[3].extend(sv.flatten())

        # decelerating, truncated velocity bicone i.e., best
        cone_best = [[],[],[],[]]
        for i in range(len(angle_list)):
            this_angle = angle_list[i]
            ## geometry (X=R.A., Z=decl., Y=depth)
            nx,ny,nz,nv,sx,sy,sz,sv = self._create_3d_bicone_rel(
                self.model_length, self.model_nbins, this_angle, self.model_pa,
                self.model_incl, self.model_pa_disk, self.model_incl_disk,
                self.model_disk_width*1.5, self.scale_pc,
                clipoutflow = True,
                velmax      = self.model_maxvel_best,
                velmodel    = "decelerate",
                velindex    = self.model_velindex,
                clipcnd     = self.model_cnd_rout,
                r_turn      = self.model_r_turn,
                )
            ## combine
            cone_best[0].extend(nx.flatten())
            cone_best[1].extend(ny.flatten())
            cone_best[2].extend(nz.flatten())
            cone_best[3].extend(nv.flatten())
            cone_best[0].extend(sx.flatten())
            cone_best[1].extend(sy.flatten())
            cone_best[2].extend(sz.flatten())
            cone_best[3].extend(sv.flatten())

        ###############
        # plot models #
        ###############
        chanwdith_GHz = 0.004

        size = 300
        ### cnst bicone
        for i in range(len(self.model_chanlist)):
            ## preparation
            # parameter
            this_vel     = self.model_chanlist[i]
            this_vel_str = str(this_vel).replace("-","m").split(".")[0]

            # velocity range of this channel
            this_map = self._velrange_thischan(this_vel, chanwdith_GHz, cone_cnst)

            # outputname
            outputpng = self.png_outflow_model.replace("thisvel",this_vel_str)
            outputpng = outputpng.replace("thismodel","cnst")

            ## plot
            plt.figure(figsize=(13,10))
            plt.subplots_adjust(bottom=0.10, left=0.2145, right=0.83, top=0.90)
            gs = gridspec.GridSpec(nrows=10, ncols=10)
            ax = plt.subplot(gs[0:10,0:10])
            self._ax_conemodel(ax, this_vel)
            ax.scatter(-1*this_map[0], this_map[1], c="darkred", lw=0, s=size)
            plt.savefig(outputpng, dpi=fig_dpi, transparent=False)

    #################
    # _ax_conemodel #
    #################

    def _ax_conemodel(
        ax,
        this_vel,
        ):
        """
        """

        ax_general_setup(
            ax,
            grid   = "both",
            title  = None,
            xlim   = [9, -9],
            ylim   = [-9, 9],
            xlabel = "x-offset (arcsec)",
            ylabel = "y-offset (arcsec)",
            )
        
        fov_diamter = 16.5
        fov = patches.Ellipse(
            xy        = (-0, 0),
            width     = fov_diamter,
            height    = fov_diamter,
            angle     = 0,
            fill      = False,
            edgecolor = "black",
            alpha     = 1.0,
            lw        = 3.5,
            )

        ax.add_patch(fov)
        # annotation 2: outflow outlines
        theta1 = -10.0 # degree
        x1 = fov_diamter/2.0 * np.cos(np.radians(-1*theta1+90))
        y1 = fov_diamter/2.0 * np.sin(np.radians(-1*theta1+90))
        ax.plot([x1, -x1], [y1, -y1], "--", c="black", lw=3.5)
        theta2 = 70.0 # degree
        x1 = fov_diamter/2.0 * np.cos(np.radians(-1*theta2+90))
        y1 = fov_diamter/2.0 * np.sin(np.radians(-1*theta2+90))
        ax.plot([x1, -x1], [y1, -y1], "--", c="black", lw=3.5)
        # text
        comment = str(this_vel) + " km s$^{-1}$"
        if this_vel>0:
            color = "red"
        else:
            color = "blue"
        t = ax.text(0.04, 0.96, comment, horizontalalignment="left", verticalalignment="top", color = color, weight="bold",
        transform = ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))
        #
        ax.set_xlabel("x-offset (arcsec)")
        ax.set_ylabel("y-offset (arcsec)")

    ######################
    # _velrange_thischan #
    ######################

    def _velrange_thischan(
        self,
        this_vel,
        chanwdith_GHz,
        data,
        ):

        data          = np.array(data)
        z             = 0.00379
        obs_freq_ci   = 492.16065100 / (1+z) # GHz
        # velocity range of this channel
        chanwidth_kms = chanwdith_GHz / obs_freq_ci * 299792.458 # km/s
        vupp          = this_vel + chanwidth_kms/2. #* 2
        vlow          = this_vel - chanwidth_kms/2. #* 2
        x             = data[0][data[3]<=vupp]
        y             = data[1][data[3]<=vupp]
        z             = data[2][data[3]<=vupp]
        v             = data[3][data[3]<=vupp]
        x             = x[v>=vlow]
        y             = y[v>=vlow]
        z             = z[v>=vlow]
        v             = v[v>=vlow]

        return np.array([x, y, z, v])

    #########################
    # _create_3d_bicone_rel #
    #########################

    def _create_3d_bicone_rel(
        self,
        length,
        nbins,
        angle,
        pa,
        incl,
        pa_disk,
        incl_disk,
        width_disk,
        scale,
        clipoutflow = False,
        velmax      = 0,
        velmodel    = "const", # const or decelerate
        velindex    = 0.3,
        clipcnd     = None,
        r_turn      = 140, # parsec
        ):
        """
        """

        ### radian to degree
        angle     = np.radians(angle)
        pa        = np.radians(pa)
        incl      = np.radians(incl)
        pa_disk   = np.radians(pa_disk)
        incl_disk = np.radians(incl_disk)

        ### preparation
        theta = np.linspace(0, 2*np.pi, nbins)
        r     = np.linspace(0, length, nbins)
        t,R   = np.meshgrid(theta, r)

        ### northern outer cone
        ## generate edge-on nothern cone (-X=R.A., Y=decl., Z=depth)
        x0 = np.tan(angle)*R*np.cos(t)
        y0 = R
        z0 = np.tan(angle)*R*np.sin(t)
        r0 = np.sqrt(x0**2 + y0**2 + z0**2)
        ## incine it based on inc
        x1 = x0
        y1 = y0*np.cos(incl) - z0*np.sin(incl)
        z1 = y0*np.sin(incl) + z0*np.cos(incl)
        ## rotate it based on pa
        x2 = x1*np.cos(pa) - y1*np.sin(pa)
        y2 = x1*np.sin(pa) + y1*np.cos(pa)
        z2 = z1
        ## LoS velocity
        t2 = -np.degrees(np.arccos(x2/r0))
        p2 = -np.degrees(np.arctan2(y2, z2))
        v2 = np.sin(np.radians(t2)) * np.cos(np.radians(p2)) * velmax
        ## mask
        # incine it based on inc
        X1 = x0
        Y1 = y0*np.cos(incl+incl_disk) - z0*np.sin(incl+incl_disk)
        Z1 = y0*np.sin(incl+incl_disk) + z0*np.cos(incl+incl_disk)
        # rotate it based on pa
        X2 = X1*np.cos(pa-pa_disk) - Y1*np.sin(pa-pa_disk)
        Y2 = X1*np.sin(pa-pa_disk) + Y1*np.cos(pa-pa_disk)
        Z2 = Z1
        if velmodel=="decelerate":
            ka = velmax/r_turn
            kb = velmax/r_turn/3.0
            v2 = np.where(r0>r_turn,velmax-kb*(r0-r_turn),ka*r0)
            v2 = np.sin(np.radians(t2)) * np.cos(np.radians(p2)) * v2
        # mask CND and outflow
        if clipoutflow==True: # width_disk/2.
            ax = 0.35265396141693
            ay = 0
            az = 2
            aa = -width_disk#/2.0
            x2[ax*X2+ay*Y2+az*Z2+aa>=0] = 0
            y2[ax*X2+ay*Y2+az*Z2+aa>=0] = 0
            z2[ax*X2+ay*Y2+az*Z2+aa>=0] = 0
            v2[ax*X2+ay*Y2+az*Z2+aa>=0] = 0
            nX = x2 / scale
            nY = y2 / scale
            nZ = z2 / scale
            nV = v2
        else:
            nX = x2 / scale
            nY = y2 / scale
            nZ = z2 / scale
            nV = v2

        nX[np.isnan(nX)] = 0
        nY[np.isnan(nY)] = 0
        nZ[np.isnan(nZ)] = 0
        nV[np.isnan(nV)] = 0

        if clipcnd!=None:
            nR = np.sqrt(nX**2 + nY**2 + nZ**2) * scale
            nX[abs(nR)<clipcnd] = 0
            nY[abs(nR)<clipcnd] = 0
            nZ[abs(nR)<clipcnd] = 0
            nV[abs(nR)<clipcnd] = 0

        ### southern outer cone
        ## generate edge-on nothern cone (-X=R.A., Y=decl., Z=depth)
        x0 = np.tan(angle)*R*np.cos(t)
        y0 = -R
        z0 = np.tan(angle)*R*np.sin(t)
        r0 = np.sqrt(x0**2 + y0**2 + z0**2)
        ## incine it based on inc
        x1 = x0
        y1 = y0*np.cos(incl) - z0*np.sin(incl)
        z1 = y0*np.sin(incl) + z0*np.cos(incl)
        ## rotate it based on pa
        x2 = x1*np.cos(pa) - y1*np.sin(pa)
        y2 = x1*np.sin(pa) + y1*np.cos(pa)
        z2 = z1
        ## LoS velocity
        t2 = -np.degrees(np.arccos(x2/r0))
        p2 = -np.degrees(np.arctan2(y2, z2))
        v2 = np.sin(np.radians(t2)) * np.cos(np.radians(p2)) * velmax
        ## mask
        # incine it based on inc
        X1 = x0
        Y1 = y0*np.cos(incl+incl_disk) - z0*np.sin(incl+incl_disk)
        Z1 = y0*np.sin(incl+incl_disk) + z0*np.cos(incl+incl_disk)
        # rotate it based on pa
        X2 = X1*np.cos(pa-pa_disk) - Y1*np.sin(pa-pa_disk)
        Y2 = X1*np.sin(pa-pa_disk) + Y1*np.cos(pa-pa_disk)
        Z2 = Z1
        if velmodel=="decelerate":
            ka = velmax/r_turn
            kb = velmax/r_turn/3.0
            v2 = np.where(r0>r_turn,velmax-kb*(r0-r_turn),ka*r0)
            v2 = np.sin(np.radians(t2)) * np.cos(np.radians(p2)) * v2
        # mask CND and outflow
        if clipoutflow==True:
            ax = 0.35265396141693
            ay = 0
            az = 2
            aa = width_disk#/2.0
            x2[ax*X2+ay*Y2+az*Z2+aa<=0] = 0
            y2[ax*X2+ay*Y2+az*Z2+aa<=0] = 0
            z2[ax*X2+ay*Y2+az*Z2+aa<=0] = 0
            v2[ax*X2+ay*Y2+az*Z2+aa<=0] = 0
            sX = x2 / scale
            sY = y2 / scale
            sZ = z2 / scale
            sV = v2
        else:
            sX = x2 / scale
            sY = y2 / scale
            sZ = z2 / scale
            sV = v2
        
        sX[np.isnan(sX)] = 0
        sY[np.isnan(sY)] = 0
        sZ[np.isnan(sZ)] = 0
        sV[np.isnan(sV)] = 0
        
        if clipcnd!=None:
            sR = np.sqrt(sX**2 + sY**2 + sZ**2) * scale
            sX[abs(sR)<clipcnd] = 0
            sY[abs(sR)<clipcnd] = 0
            sZ[abs(sR)<clipcnd] = 0
            sV[abs(sR)<clipcnd] = 0

        return nX, nY, nZ, nV, sX, sY, sZ, sV

    ########################
    # get_outflow_channels #
    ########################

    def get_outflow_channels(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_cube_cico,taskname)
        check_first(self.outfits_cube_ci10)
        check_first(self.outfits_cube_co10)

        for i in range(len(self.chans_num)):
            this_chan = str(self.chans_num[i])
            this_chan_str =  this_chan.zfill(3)
            this_text = self.chans_text[i]
            this_c    = self.chans_color[i]
            this_cico = self.dir_ready+"cube_cico_ch"+this_chan_str+".fits"
            this_ci10 = self.dir_ready+"cube_ci10_ch"+this_chan_str+".fits"
            this_co10 = self.dir_ready+"cube_co10_ch"+this_chan_str+".fits"
            this_outpng = self.outpng_outflow_chans.replace("?",str(this_chan))

            # extract a channel of the ratio cube
            self._extract_one_chan(self.outfits_cube_cico,
                this_cico,this_chan,self.imrebin_factor)

            # extract a channel of the ci cube
            self._extract_one_chan(self.outfits_cube_ci10,
                this_ci10,this_chan,self.imrebin_factor)

            # extract a channel of the co cube
            self._extract_one_chan(self.outfits_cube_co10,
                this_co10,this_chan,self.imrebin_factor)

            # plot
            scalebar = 100. / self.scale_pc
            label_scalebar = "100 pc"

            if i==len(self.chans_num)-1:
                title = "(a) Observed [CI]/CO ratio"
            else:
                title = None

            myfig_fits2png(
                imcolor=this_cico,
                outfile=this_outpng,
                imcontour1=this_ci10,
                imsize_as=self.imsize_as,
                ra_cnt=self.ra_agn_str,
                dec_cnt=self.dec_agn_str,
                unit_cont1=0.05,
                levels_cont1=[-5.0,5.0,10.0,20.0],
                set_cmap="jet",
                clim=[0,1],
                set_title=title,
                #set_bg_color=cm.rainbow(0),
                scalebar=scalebar,
                label_scalebar=label_scalebar,
                set_cbar=True,
                label_cbar="(K km s$^{-1}$)",
                numann=1,
                textann=False,
                comment=this_text,
                comment_color=this_c,
                )

            # cleanup
            os.system("rm -rf "+this_cico+" "+this_ci10+" "+this_co10)

    #####################
    # _extract_one_chan #
    #####################

    def _extract_one_chan(
        self,
        cubeimage,
        imagename,
        this_chan,
        factor,
        ):
        """
        """

        os.system("rm -rf " + imagename + "*")
        imsubimage(
            imagename = cubeimage,
            outfile   = imagename + "_tmp1",
            chans     = this_chan,
            )
        imrebin(
            imagename = imagename + "_tmp1",
            outfile   = imagename + "_tmp2",
            factor    = factor,
            )
        run_exportfits(
            imagename  = imagename + "_tmp2",
            fitsimage  = imagename,
            dropdeg    = True,
            dropstokes = True,
            delin      = True,
            )
        os.system("rm -rf " + imagename + "_*")

    ########################
    # plot_outflow_moments #
    ########################

    def plot_outflow_moments(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_ci10_outflow_mom0,taskname)
        check_first(self.outfits_ci10_outflow_mom1)

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        myfig_fits2png(
            imcolor=self.outfits_ci10_outflow_mom0,
            outfile=self.outpng_outflow_mom0,
            imcontour1=self.outfits_ci10_outflow_mom0,
            imsize_as=self.imsize_as,
            ra_cnt=self.ra_agn_str,
            dec_cnt=self.dec_agn_str,
            levels_cont1=[0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            width_cont1=[1.0],
            set_title="[CI] Outflow Integrated Intensity",
            colorlog=True,
            set_bg_color=cm.rainbow(0),
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar="(K km s$^{-1}$)",
            numann=1,
            textann=False,
            )

        myfig_fits2png(
            imcolor=self.outfits_ci10_outflow_mom1,
            outfile=self.outpng_outflow_mom1,
            imcontour1=self.outfits_ci10_outflow_mom0,
            imsize_as=self.imsize_as,
            ra_cnt=self.ra_agn_str,
            dec_cnt=self.dec_agn_str,
            levels_cont1=[0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            width_cont1=[1.0],
            set_title="[CI] Outflow Velocity Field",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar="(km s$^{-1}$)",
            numann=1,
            textann=False,
            )

    #######################
    # get_outflow_moments #
    #######################

    def get_outflow_moments(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_cube_cico,taskname)
        check_first(self.outfits_cube_ci10)
        check_first(self.outfits_ncube_ci10)

        # rebin cico ratio cube
        run_imrebin(
            self.outfits_cube_cico,
            self.outfits_cube_cico_rebin + "_tmp1",
            factor=self.imrebin_factor,
            )

        run_exportfits(
            self.outfits_cube_cico_rebin + "_tmp1",
            self.outfits_cube_cico_rebin,
            dropdeg    = True,
            dropstokes = True,
            delin      = True,
            )

        # mask and rebin ci cube
        run_immath_two(
            self.outfits_ncube_ci10,
            self.outfits_cube_ci10,
            self.outfits_cube_ci10_rebin + "_tmp1",
            "iif(IM1>=abs(IM0*"+str(self.snr_chan)+"),IM1,0.0)",
            )

        run_imrebin(
            self.outfits_cube_ci10_rebin + "_tmp1",
            self.outfits_cube_ci10_rebin + "_tmp2",
            factor=self.imrebin_factor,
            delin=True,
            )

        run_exportfits(
            self.outfits_cube_ci10_rebin + "_tmp2",
            self.outfits_cube_ci10_rebin + "_tmp3",
            dropdeg    = True,
            dropstokes = True,
            delin      = True,
            )

        # mask high ratio region of the ci cube
        run_immath_two(
            self.outfits_cube_cico_rebin,
            self.outfits_cube_ci10_rebin + "_tmp3",
            self.outfits_cube_ci10_rebin + "_tmp4",
            "iif(IM0>=1.0,IM1,0.0)",
            delin=False,
            )
        os.system("rm -rf " + self.outfits_cube_ci10_rebin + "_tmp3")

        run_exportfits(
            self.outfits_cube_ci10_rebin + "_tmp4",
            self.outfits_cube_ci10_rebin,
            dropdeg    = True,
            dropstokes = True,
            delin      = True,
            )

        # outflow moment map creation
        run_immoments(
            self.outfits_cube_ci10_rebin,
            self.outfits_cube_ci10_rebin,
            self.outfits_ci10_outflow_mom0 + "_tmp1",
            mom=0,
            rms=1.0,
            snr=1.0,
            )
        run_immoments(
            self.outfits_cube_ci10_rebin,
            self.outfits_cube_ci10_rebin,
            self.outfits_ci10_outflow_mom1 + "_tmp1",
            mom=1,
            rms=1.0,
            snr=1.0,
            )

        # exportfits
        run_exportfits(
            self.outfits_ci10_outflow_mom0 + "_tmp1",
            self.outfits_ci10_outflow_mom0,
            dropdeg=True,
            dropstokes=True,
            delin=True,
            )
        run_exportfits(
            self.outfits_ci10_outflow_mom1 + "_tmp1",
            self.outfits_ci10_outflow_mom1,
            dropdeg=True,
            dropstokes=True,
            delin=True,
            )

    ############
    # showcase #
    ############

    def showcase(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_map_co10,taskname)

        scalebar = 100. / self.scale_pc
        label_scalebar = "100 pc"

        myfig_fits2png(
            imcolor=self.outfits_map_ci10,
            outfile=self.outpng_map_ci,
            imcontour1=self.outfits_map_ci10,
            imsize_as=self.imsize_as,
            ra_cnt=self.ra_agn_str,
            dec_cnt=self.dec_agn_str,
            levels_cont1=[0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            width_cont1=[1.0],
            set_title="(a) [CI] $^3P_1$-$^3P_0$ Integrated Intensity",
            colorlog=True,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar="(K km s$^{-1}$)",
            numann=1,
            textann=True,
            )

        myfig_fits2png(
            imcolor=self.outfits_map_co10,
            outfile=self.outpng_map_co,
            imcontour1=self.outfits_map_ci10,
            imsize_as=self.imsize_as,
            ra_cnt=self.ra_agn_str,
            dec_cnt=self.dec_agn_str,
            levels_cont1=[0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            width_cont1=[1.0],
            set_title="(b) $^{12}CO(1-0) $Integrated Intensity",
            colorlog=True,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar="(K km s$^{-1}$)",
            numann=1,
            textann=False,
            )

        myfig_fits2png(
            imcolor=self.outfits_map_cico,
            outfile=self.outpng_map_cico,
            imcontour1=self.outfits_map_ci10,
            imsize_as=self.imsize_as,
            ra_cnt=self.ra_agn_str,
            dec_cnt=self.dec_agn_str,
            levels_cont1=[0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0.96],
            width_cont1=[1.0],
            set_title="(d) [CI]/CO Ratio",
            colorlog=True,
            clim=[0.1,2],
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar="Ratio",
            numann=1,
            textann=False,
            )

    ########################
    # plot_cico_vs_siiisii #
    ########################

    def plot_cico_vs_siiisii(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_map_co10,taskname)

      	# get coords data
        data_co, box = imval_all(self.outfits_map_co10)
        data_coords = imval(self.outfits_map_co10,box=box)["coords"]

        ra_deg = data_coords[:,:,0] * 180/np.pi
        ra_deg  = ra_deg.flatten()

        dec_deg = data_coords[:,:,1] * 180/np.pi
        dec_deg = dec_deg.flatten()

        dist_pc, theta = get_reldist_pc(ra_deg, dec_deg, self.ra_agn,
        	self.dec_agn, self.scale_pc, self.pa, self.incl)

        # co10
        data_co = imval(self.outfits_map_co10,box=box)
        data_co = data_co["data"] * data_co["mask"]
        data_co = data_co.flatten()
        data_co[np.isnan(data_co)] = 0

        # ci10
        data_ci = imval(self.outfits_map_ci10,box=box)
        data_ci = data_ci["data"] * data_ci["mask"]
        data_ci = data_ci.flatten()
        data_ci[np.isnan(data_ci)] = 0

        # sii_sii
        data_siii_sii = imval(self.outfits_map_siiisii,box=box)
        data_siii_sii = data_siii_sii["data"] * data_siii_sii["mask"]
        data_siii_sii = data_siii_sii.flatten()
        data_siii_sii[np.isnan(data_siii_sii)] = 0

        """
        ### plot scatter ci vs co
        # prepare
        cut = np.where((data_co>0) & (data_ci>0))
        co = np.log10(data_co[cut])
        ci = np.log10(data_ci[cut])
        r = dist_pc[cut]
        t = theta[cut]

        cut_c1 = np.where( (r<=self.fov_radius) & (t>=0) & (t<=60) )
        cut_c2 = np.where( (r<=self.fov_radius) & (t<=-120) & (t>=-180) )
        cut_o1 = np.where( (r<=self.fov_radius) & (t>=60) & (t<=180) )
        cut_o2 = np.where( (r<=self.fov_radius) & (t<=00) & (t>=-120) )
        cut_sb = np.where( (r>self.fov_radius) )

        # split data
        co_c1, ci_c1, r_c1 = co[cut_c1], ci[cut_c1], r[cut_c1]
        co_c2, ci_c2, r_c2 = co[cut_c2], ci[cut_c2], r[cut_c2]
        co_o1, ci_o1, _ = co[cut_o1], ci[cut_o1], r[cut_o1]
        co_o2, ci_o2, _ = co[cut_o2], ci[cut_o2], r[cut_o2]
        co_sb, ci_sb, _ = co[cut_sb], ci[cut_sb], r[cut_sb]

        co_cone = np.r_[co_c1, co_c2]
        ci_cone = np.r_[ci_c1, ci_c2]
        r_cone  = np.r_[r_c1,  r_c2]

        co_outcone = np.r_[co_o1, co_o2]
        ci_outcone = np.r_[ci_o1, ci_o2]

        # plot
        self._plot_scatters(
        	self.outpng_cico_vs_siiisii,
            co_sb, ci_sb, co_outcone, ci_outcone, co_cone, ci_cone, r_cone,
            "log $L'_{CO(1-0)}$ (K km s$^{-1}$ pc$^2$)",
            "log $L'_{[CI](1-0)}$ (K km s$^{-1}$ pc$^2$)",
            "(c) log $L'_{[CI](1-0)}$ vs. log $L'_{CO(1-0)}$",
            [-1,3.5], [-0.1,3.5],
            plot_line = False,
            )
        """

    #################
    # plot_ci_vs_co #
    #################

    def plot_ci_vs_co(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_map_co10,taskname)

      	# get coords data
        data_co, box = imval_all(self.outfits_map_co10)
        data_coords = imval(self.outfits_map_co10,box=box)["coords"]

        ra_deg = data_coords[:,:,0] * 180/np.pi
        ra_deg  = ra_deg.flatten()

        dec_deg = data_coords[:,:,1] * 180/np.pi
        dec_deg = dec_deg.flatten()

        dist_pc, theta = get_reldist_pc(ra_deg, dec_deg, self.ra_agn,
        	self.dec_agn, self.scale_pc, self.pa, self.incl)

        # co10
        data_co = imval(self.outfits_map_co10,box=box)
        data_co = data_co["data"] * data_co["mask"]
        data_co = data_co.flatten()
        data_co[np.isnan(data_co)] = 0

        # ci10
        data_ci = imval(self.outfits_map_ci10,box=box)
        data_ci = data_ci["data"] * data_ci["mask"]
        data_ci = data_ci.flatten()
        data_ci[np.isnan(data_ci)] = 0

        # sii_sii
        data_siii_sii = imval(self.outfits_map_siiisii,box=box)
        data_siii_sii = data_siii_sii["data"] * data_siii_sii["mask"]
        data_siii_sii = data_siii_sii.flatten()
        data_siii_sii[np.isnan(data_siii_sii)] = 0

        ### plot scatter ci vs co
        # prepare
        cut = np.where((data_co>0) & (data_ci>0))
        co = np.log10(data_co[cut])
        ci = np.log10(data_ci[cut])
        r = dist_pc[cut]
        t = theta[cut]

        cut_c1 = np.where( (r<=self.fov_radius) & (t>=0) & (t<=60) )
        cut_c2 = np.where( (r<=self.fov_radius) & (t<=-120) & (t>=-180) )
        cut_o1 = np.where( (r<=self.fov_radius) & (t>=60) & (t<=180) )
        cut_o2 = np.where( (r<=self.fov_radius) & (t<=00) & (t>=-120) )
        cut_sb = np.where( (r>self.fov_radius) )

        # split data
        co_c1, ci_c1, r_c1 = co[cut_c1], ci[cut_c1], r[cut_c1]
        co_c2, ci_c2, r_c2 = co[cut_c2], ci[cut_c2], r[cut_c2]
        co_o1, ci_o1, _ = co[cut_o1], ci[cut_o1], r[cut_o1]
        co_o2, ci_o2, _ = co[cut_o2], ci[cut_o2], r[cut_o2]
        co_sb, ci_sb, _ = co[cut_sb], ci[cut_sb], r[cut_sb]

        co_cone = np.r_[co_c1, co_c2]
        ci_cone = np.r_[ci_c1, ci_c2]
        r_cone  = np.r_[r_c1,  r_c2]

        co_outcone = np.r_[co_o1, co_o2]
        ci_outcone = np.r_[ci_o1, ci_o2]

        # plot
        self._plot_scatters(
        	self.outpng_ci_vs_co,
            co_sb, ci_sb, co_outcone, ci_outcone, co_cone, ci_cone, r_cone,
            "log $L'_{CO(1-0)}$ (K km s$^{-1}$ pc$^2$)",
            "log $L'_{[CI](1-0)}$ (K km s$^{-1}$ pc$^2$)",
            "(c) log $L'_{[CI](1-0)}$ vs. log $L'_{CO(1-0)}$",
            [-1,3.5], [-0.1,3.5],
            plot_line = True,
            )

    def _plot_scatters(
        self,
        output,
        x1, y1,
        x2, y2,
        x3, y3, r3,
        xlabel, ylabel, title,
        xlim, ylim,
        plot_line = True,
        ):

        fig = plt.figure(figsize=(13,10))
        gs = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        ad = [0.215,0.83,0.10,0.90]
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        # plot
        ax1.scatter(x1, y1, lw=0, c="gray", s=20)
        ax1.scatter(x2, y2, lw=0, c="black", s=40)
        cs = ax1.scatter(x3, y3, lw=0, c=r3, cmap="rainbow_r", s=40)

        # colorbar
        cax = fig.add_axes([0.71, 0.15, 0.03, 0.3]) # fig.add_axes([0.75, 0.57, 0.03, 0.3])
        cbar = plt.colorbar(cs, cax=cax)
        cbar.set_label("Distance (pc)")
        cbar.set_ticks([100,200,300,400,500])

        # plot line
        if plot_line==True:
            ax1.plot([-1.0,8.0], [-1.0,8.0], "k-", lw=2)
            ax1.plot([-1,8.0], [-1.30103,7.69897], "k-", lw=2)
            ax1.plot([-1.0,8.0], [-2.0,7.0], "k-", lw=2)
            ax1.text(3.42,3.41,"1:1",rotation=51.34,horizontalalignment="right")
            ax1.text(3.42,3.12,"1:0.5",rotation=51.34,horizontalalignment="right")
            ax1.text(3.42,2.43,"1:0.1",rotation=51.34,horizontalalignment="right")

        # text
        ax1.text(0.05,0.92,"FoV-1 Cone (colorized)",transform=ax1.transAxes)
        ax1.text(0.05,0.87,"FoV-1 Non-cone (black)",transform=ax1.transAxes)
        ax1.text(0.05,0.82,"FoV-2 (grey)",transform=ax1.transAxes)
        ax1.text(0.05,0.77,"FoV-3 (grey)",transform=ax1.transAxes)

        plt.savefig(output, dpi=self.fig_dpi)

    ##################
    # _plot_7m_cubes #
    ##################

    def _plot_7m_cubes(
        self,
        ):
        """ not tested yet
        """

        data = np.loadtxt(self.outtxt_slopes_7m)

        plt.figure(figsize=(10,10))
        plt.subplots_adjust(bottom=0.10, left=0.19, right=0.99, top=0.90)
        gs = gridspec.GridSpec(nrows=11, ncols=11)
        ax = plt.subplot(gs[0:11,0:11])
        ax.hist(data)
        plt.savefig(self.outpng_slopes_7m)

    ####################
    # compare_7m_cubes #
    ####################

    def compare_7m_cubes(
        self,
        inputbeam_as=4.5,
        round_beam_as=6.0,
        cut_flux=0.8,
        ):
        """ not tested yet
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        this_image = glob.glob(self.dir_7m + this_target + "*.fits")[0]
        check_first(this_image,taskname)

        #
        targets   = ["ngc1068_b8_1","ngc1068_b8_2","ngc1068_b8_3"]
        list_data = []

        for i in range(len(targets)):
            this_target = targets[i]
            cubes       = glob.glob(self.dir_7m + this_target + "*.fits")
            cubes.sort()

            for j in range(len(cubes)):
                this_cube        = cubes[j]
                this_cube_smooth = this_cube.replace(".fits",".smooth")
                this_cube_regrid = this_cube.replace(".fits",".regrid")
                print(this_cube.split("/")[-1])

                # regrid template
                if j==0:
                    template = this_cube.replace(".fits",".template")
                    run_importfits(this_cube, template)
                    shape = imhead(template,mode="list")["shape"]
                    box   = "0,0," + str(shape[0]-1) + "," + str(shape[1]-1)

                run_roundsmooth(
                    this_cube,
                    this_cube_smooth,
                    targetbeam=round_beam_as,
                    inputbeam=inputbeam_as,
                    )

                run_imregrid(
                    this_cube_smooth,
                    template,
                    this_cube_regrid,
                    delin=True,
                    )

                data = imval(this_cube_regrid,box=box)["data"]
                list_data.append(data.tolist())
                os.system("rm -rf " + this_cube_regrid)

            os.system("rm -rf " + template)

            # start comparison
            print("# start comparison")
            len_cubes = np.linspace(0,len(cubes)-1,len(cubes))
            comb = itertools.permutations(len_cubes, 2)
            list_slopes = []

            for j in comb:
                print(int(j[0]), int(j[1]))
                data_x = np.array(list_data[int(j[0])])
                data_y = np.array(list_data[int(j[1])])

                # cut data
                cut    = np.where((data_x>=cut_flux) & (data_y>=cut_flux))
                data_x = np.log10(data_x[cut])
                data_y = np.log10(data_y[cut])
                sigma  = 1./np.sqrt(data_x**2+data_y**2)

                popt,_  = curve_fit(f_lin2, data_x, data_y,
                	sigma=sigma, p0=[1.0], maxfev = 10000)
                best_y = func_lin(data_x, popt[0])
                list_slopes.append(popt[0])

            os.system("rm -rf " + self.outtxt_slopes_7m)
            np.savetxt(self.outtxt_slopes_7m, list_slopes)

        self._plot_7m_cubes()

    ##############
    # ratio_cube #
    ##############

    def ratio_cube(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_cube_ci10,taskname)

        run_immath_two(
            imagename1 = self.outfits_cube_ci10,
            imagename2 = self.outfits_cube_co10,
            expr       = "IM0/IM1",
            outfile    = self.out_cube_cico + "_tmp1",
            )

        run_immath_two(
            imagename1 = self.outfits_ncube_co10,
            imagename2 = self.outfits_cube_co10,
            expr       = "iif(IM1>0,IM0/IM1,1e4)",
            outfile    = self.out_cube_cico + "_tmp2",
            )

        run_immath_two(
            imagename1 = self.outfits_ncube_ci10,
            imagename2 = self.outfits_cube_ci10,
            expr       = "iif(IM1>0,IM0/IM1,1e4)",
            outfile    = self.out_cube_cico + "_tmp3",
            )

        run_immath_three(
            imagename1 = self.out_cube_cico + "_tmp2",
            imagename2 = self.out_cube_cico + "_tmp3",
            imagename3 = self.out_cube_cico + "_tmp1",
            expr       = "abs(IM2 * sqrt(IM0*IM0+IM1*IM1))",
            outfile    = self.out_cube_cico + "_tmp4",
            )

        run_immath_two(
            imagename1 = self.out_cube_cico + "_tmp1",
            imagename2 = self.out_cube_cico + "_tmp4",
            expr       = "iif(IM0>IM1*" + str(self.snr_cube) + ",IM0,0)",
            outfile    = self.out_cube_cico + "_tmp5",
            )

        run_immath_two(
            imagename1 = self.out_cube_cico + "_tmp1",
            imagename2 = self.out_cube_cico + "_tmp4",
            expr       = "iif(IM0>IM1*" + str(self.snr_cube) + ",0,1)",
            outfile    = self.out_cube_cico + "_tmp6",
            )

        run_immath_one(
            imagename  = self.out_cube_cico + "_tmp3",
            expr       = "iif(IM0<" + str(1.0/self.snr_ratio) + ",1,0)",
            outfile    = self.out_cube_cico + "_tmp7",
            )

        run_immath_two(
            imagename1 = self.out_cube_cico + "_tmp6",
            imagename2 = self.out_cube_cico + "_tmp7",
            expr       = "IM0*IM1",
            outfile    = self.out_cube_cico + "_tmp8",
            )

        run_immath_two(
            imagename1 = self.out_cube_cico + "_tmp5",
            imagename2 = self.out_cube_cico + "_tmp8",
            expr       = "iif(IM0>0,IM0,IM1*10)",
            outfile    = self.out_cube_cico,
            )

        run_exportfits(self.out_cube_cico,self.outfits_cube_cico)

        os.system("rm -rf " + self.out_cube_cico + "*")

    #############
    # ratio_map #
    #############

    def ratio_map(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_map_co10.replace(".fits",".masked")+".fits",taskname)

        run_importfits(
            fitsimage = self.outfits_map_co10.replace(".fits",".masked")+".fits",
            imagename = self.outfits_map_co10.replace(".fits",".masked"),
            defaultaxes = True,
            defaultaxesvalues = ["RA","Dec","1GHz","I"],
            )

        run_immath_two(
            imagename1 = self.outfits_map_ci10,
            imagename2 = self.outfits_map_co10.replace(".fits",".masked"),
            expr       = "iif(IM1>0,IM0/IM1,0)",
            chans      = "0",
            outfile    = self.out_map_cico,
            )

        run_exportfits(
            imagename  = self.out_map_cico,
            fitsimage  = self.outfits_map_cico,
            dropdeg    = True,
            dropstokes = True,
            delin      = True,
            )

        os.system("rm -rf " + self.outfits_map_co10.replace(".fits",".masked"))

    ##################
    # ci_fov_masking #
    ##################

    def ci_fov_masking(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_co10,taskname)

        # create ci fov & detection mask
        mask = "fov.mask"
        run_importfits(
            fitsimage         = self.outfits_map_ci10,
            imagename         = mask + "_tmp1",
            defaultaxes       = True,
            defaultaxesvalues = ["RA","Dec","1GHz","I"],
            )
        thres = "iif(IM0>=0, 1, 0)"
        signal_masking(mask+"_tmp1", mask, thres, delin=True)

        # masking
        expr = "IM0*IM1"

        outfile_co10 = self.outfits_map_co10.replace(".fits",".masked")
        run_immath_two(self.outfits_map_co10, mask, outfile_co10, expr, "0")
        run_exportfits(outfile_co10,outfile_co10+".fits",True,True)

        outfile_siiisii = self.outfits_map_siiisii.replace(".fits",".masked")
        run_immath_two(self.outfits_map_siiisii, mask, outfile_siiisii, expr, "0")
        run_exportfits(outfile_siiisii,outfile_siiisii+".fits",True,True)

        outfile_av = self.outfits_map_av.replace(".fits",".masked")
        run_immath_two(self.outfits_map_av, mask, outfile_av, expr, "0")
        run_exportfits(outfile_av,outfile_av+".fits",True,True)

        os.system("rm -rf " + mask)
        os.system("rm -rf " + outfile_co10)
        os.system("rm -rf " + outfile_siiisii)
        os.system("rm -rf " + outfile_av)

    ##############
    # align_maps #
    ##############

    def align_maps(
        self,
        ):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.image_co10,taskname)

        ##############################
        # MUSE SIII/SII map (MAGNUM) #
        ##############################
        run_importfits(self.image_siiisii,self.out_map_siiisii)

        #############################
        # 2D co10 (align to MAGNUM) #
        #############################
        run_importfits(self.image_co10,self.out_map_co10)
        run_importfits(self.image_eco10,self.out_map_eco10)

        # align
        template = self.out_map_siiisii

        expr = "iif( IM0>IM1*" + str(self.snr_cube) + ", IM0, 0 )"
        outfile1 = self.out_map_co10 + "_tmp1"
        run_immath_two(self.out_map_co10, self.out_map_eco10, outfile1, expr)
        self._align_one_map(outfile1, template, self.outfits_map_co10)

        expr = "iif( IM0>IM1*" + str(self.snr_cube) + ", IM1, 0 )"
        outfile2 = self.out_map_eco10 + "_tmp1"
        run_immath_two(outfile1, self.out_map_eco10, outfile2, expr)
        self._align_one_map(outfile2, template, self.outfits_map_eco10)

        os.system("rm -rf " + outfile1)
        os.system("rm -rf " + outfile2)
        os.system("rm -rf " + self.out_map_co10)
        os.system("rm -rf " + self.out_map_eco10)

        #############################
        # 2D ci10 (align to MAGNUM) #
        #############################
        run_importfits(self.image_ci10,self.out_map_ci10)
        run_importfits(self.image_eci10,self.out_map_eci10)

        # align
        template = self.out_map_siiisii

        expr = "iif( IM0>IM1*" + str(self.snr_cube) + ", IM0, 0 )"
        outfile1 = self.out_map_ci10 + "_tmp1"
        run_immath_two(self.out_map_ci10, self.out_map_eci10, outfile1, expr)
        self._align_one_map(outfile1, template, self.outfits_map_ci10)

        expr = "iif( IM0>IM1*" + str(self.snr_cube) + ", IM1, 0 )"
        outfile2 = self.out_map_eci10 + "_tmp1"
        run_immath_two(outfile1, self.out_map_eci10, outfile2, expr)
        self._align_one_map(outfile2, template, self.outfits_map_eci10)

        os.system("rm -rf " + outfile1)
        os.system("rm -rf " + outfile2)
        os.system("rm -rf " + self.out_map_ci10)
        os.system("rm -rf " + self.out_map_eci10)

        #############################
        # 3D ci10 (align to MAGNUM) #
        #############################
        run_importfits(self.cube_ci10,self.out_cube_ci10)
        run_importfits(self.ncube_ci10,self.out_ncube_ci10)

        # align
        template = self.outfits_map_co10
        self._align_one_map(self.out_cube_ci10, template, self.outfits_cube_ci10, axes=[0,1])

        template = self.outfits_cube_ci10
        self._align_one_map(self.out_ncube_ci10, template, self.outfits_ncube_ci10)

        os.system("rm -rf " + self.out_cube_ci10)
        os.system("rm -rf " + self.out_ncube_ci10)

        #############################
        # 3D co10 (align to MAGNUM) #
        #############################
        run_importfits(self.cube_co10,self.out_cube_co10)
        run_importfits(self.ncube_co10,self.out_ncube_co10)

        # align
        template = self.outfits_cube_ci10
        self._align_one_map(self.out_cube_co10, template, self.outfits_cube_co10)
        self._align_one_map(self.out_ncube_co10, template, self.outfits_ncube_co10)

        os.system("rm -rf " + self.out_cube_co10)
        os.system("rm -rf " + self.out_ncube_co10)

        #################
        # other 2D maps #
        #################
        # import to casa
        run_importfits(self.image_av,self.out_map_av)
        run_importfits(self.image_oiii,self.out_map_oiii)
        run_importfits(self.image_vla,self.out_map_radio)

        # add beam
        imhead(self.out_map_siiisii,mode="put",hdkey="beammajor",hdvalue="0.8arcsec")
        imhead(self.out_map_siiisii,mode="put",hdkey="beamminor",hdvalue="0.8arcsec")
        imhead(self.out_map_av,mode="put",hdkey="beammajor",hdvalue="0.8arcsec")
        imhead(self.out_map_av,mode="put",hdkey="beamminor",hdvalue="0.8arcsec")
        imhead(self.out_map_oiii,mode="put",hdkey="beammajor",hdvalue="0.2arcsec")
        imhead(self.out_map_oiii,mode="put",hdkey="beamminor",hdvalue="0.2arcsec")

        # export
        run_exportfits(self.out_map_siiisii,self.outfits_map_siiisii,True,True,True)
        run_exportfits(self.out_map_av,self.outfits_map_av,True,True,True)
        run_exportfits(self.out_map_oiii,self.outfits_map_oiii,True,True,True)
        run_exportfits(self.out_map_radio,self.outfits_map_radio,True,True,True)

    ##################
    # _align_one_map #
    ##################

    def _align_one_map(self, imagename, template, outfits, beam="0.8arcsec", axes=-1):
        """
        """
        delim  = False
        deltmp = False

        # make sure casa image
        if imagename[-5:]==".fits":
            run_importfits(
                fitsimage   = imagename,
                imagename   = imagename.replace(".fits",".image"),
                defaultaxes = False,
                )
            imagename = imagename.replace(".fits",".image")
            delim     = True

        # make sure casa image
        if template[-5:]==".fits":
            run_importfits(
                fitsimage   = template,
                imagename   = template.replace(".fits",".image"),
                )
            template = template.replace(".fits",".image")
            deltmp   = True

        # make sure ICRS
        relabelimage(imagename, j2000_to_icrs=True)
        relabelimage(template, j2000_to_icrs=True)

        # regrid
        run_imregrid(
            imagename = imagename,
            template  = template,
            outfile   = imagename + ".regrid",
            )

        # exportfits
        run_exportfits(
            imagename = imagename + ".regrid",
            fitsimage = outfits,
            dropdeg    = True,
            dropstokes = True,
            delin      = True,
            )

        # delete
        os.system("rm -rf " + imagename + ".regrid")

        if delim==True:
            os.system("rm -rf " + imagename)

        if deltmp==True:
            os.system("rm -rf " + template)

