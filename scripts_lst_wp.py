"""
Python class for the NGC 1068 PCA project

requirements:
CASA Version 5.4.0-70, ananlysisUtils, astropy

data:

usage:
> import os
> from scripts_sim_lst_wp import ToolsLSTSim as tools
>
> # key
> tl = tools(
>     refresh     = False,
>     keyfile_gal = "/home02/saitots/myUtils/keys_sim_lst_wp/key_ngc1068.txt",
>     keyfile_fig = "/home02/saitots/myUtils/keys_sim_lst_wp/key_figures.txt",
>     )
>
> # main
> tl.run_sim_lst_alma(
>     # analysis
>     do_prepare             = True,
>     )
>
> os.system("rm -rf *.last")

white paper drafts:
Date         Filename                To
2022-??-??

history:
2022-03-08   created
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob
import numpy as np

from mycasa_tasks import *
from mycasa_plots import *
from mycasa_simobs import *

###############
# ToolsLSTSim #
###############
class ToolsLSTSim():
    """
    Class for the LST white paper 2022 project.
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
        self.dir_keyfile = "/".join(self.keyfile_gal.split("/")[:-1]) + "/"

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

        # simobserve
        self.project_torus = self._read_key("project_torus")
        self.project_n1097 = self._read_key("project_n1097")
        self.config_c1     = self.dir_keyfile + self._read_key("config_c1")
        self.config_c10    = self.dir_keyfile + self._read_key("config_c10")
        self.config_7m     = self.dir_keyfile + self._read_key("config_7m")
        self.config_7m_lst = self.dir_keyfile + self._read_key("config_7m_lst")

        # phangs-alma pipeline
        self.dir_pipeline = self._read_key("dir_pipeline")

    def _set_input_fits(self):
        """
        """

        self.n1068_template_file = self._read_key("n1068_template_file")
        self.n1068_template_mask = self._read_key("n1068_template_mask")

        self.n1097_template_file = self._read_key("n1097_template_file")
        self.n1097_template_mask = self._read_key("n1097_template_mask")

    def _set_output_fits(self):
        """
        """

        self.n1097_template_in_jypix        = self.project_n1097 + "_template_jypixel.image"
        self.n1097_template_clipped         = self.project_n1097 + "_template_clipped.image"
        self.n1097_template_mask_imported   = self.project_n1097 + "_template_mask.image"
        self.n1097_template_rotated         = self.project_n1097 + "_template_rotated.image"
        self.n1097_template_shrunk          = self.project_n1097 + "_template_shrunk.image"
        self.n1097_template_fullspec        = self.project_n1097 + "_template_fullspec.image"
        self.n1097_template_fullspec_div3   = self.project_n1097 + "_template_fullspec_div3.image"
        self.n1097_template_fullspec_div5   = self.project_n1097 + "_template_fullspec_div5.image"
        self.n1097_template_fullspec_div10  = self.project_n1097 + "_template_fullspec_div10.image"
        self.n1097_template_fullspec_div30  = self.project_n1097 + "_template_fullspec_div30.image"
        self.n1097_template_fullspec_div100 = self.project_n1097 + "_template_fullspec_div100.image"
        self.n1097_template_withcont        = self.project_n1097 + "_template_withcont.image"
        self.n1097_template_withcont_div3   = self.project_n1097 + "_template_withcont_div3.image"
        self.n1097_template_withcont_div5   = self.project_n1097 + "_template_withcont_div5.image"
        self.n1097_template_withcont_div10  = self.project_n1097 + "_template_withcont_div10.image"
        self.n1097_template_withcont_div30  = self.project_n1097 + "_template_withcont_div30.image"
        self.n1097_template_withcont_div100 = self.project_n1097 + "_template_withcont_div100.image"
        self.n1097_sdnoise_image            = self.project_n1097 + "_aca_tp_noise_nocont.image"
        self.n1097_sdimage_fullspec         = self.project_n1097 + "_aca_tp_nocont.image"
        self.n1097_lstnoise_image           = self.project_n1097 + "_lst_noise_nocont.image"
        self.n1097_lstimage_fullspec        = self.project_n1097 + "_lst_nocont.image"

        self.n1097_feather_tp_7m            = self.dir_ready + "outputs/" + self._read_key("n1097_feather_tp_7m")
        self.n1097_feather_lst_7m           = self.dir_ready + "outputs/" + self._read_key("n1097_feather_lst_7m")

        self.torus_template_file            = self._read_key("torus_template_file")

    def _set_input_param(self):
        """
        """

        # sim properties
        self.singledish_noise = 0.102 # Jy/beam at final res
        self.singledish_res   = "28.37arcsec" # resolution
        self.image_rot_n1068sim = "0deg"
        self.image_rot_n1097sim = "35deg"

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

        self.outpng_config_12m   = self.dir_products + self._read_key("outpng_config_12m")
        self.outpng_config_7m    = self.dir_products + self._read_key("outpng_config_7m")
        self.outpng_uv_alma_lst1 = self.dir_products + self._read_key("outpng_uv_alma_lst1")

    ####################
    # run_sim_lst_alma #
    ####################

    def run_sim_lst_alma(
        self,
        ##############
        # ngc1097sim #
        ##############
        # prepare
        tinteg_n1097sim        = 48,    # 7m total observing time
        observed_freq          = 492.16065100, # GHz, determine LST and TP beam sizes
        do_template_n1097sim   = False, # create "wide" template cube for mapping simobserve
        # ACA-alone
        do_simACA_n1097sim     = False, # sim ACA band 8 for big ngc1097sim
        do_imaging_n1097sim    = False, # imaging sim ms
        # SD-alone
        dryrun_simSD           = False, # just output SD mapping parameters
        do_simTP_n1097sim      = False, # sim ACA TP alone; after do_imaging_n1097sim
        do_simLST_n1097sim     = False, # sim LST alone; after do_imaging_n1097sim
        do_feather             = False,
        do_tp2vis              = False, # not implemented yet
        # LST-connected 7m array
        do_simACA_LST_n1097sim = False,
        #
        ############
        # torussim #
        ############
        # prepare
        tinteg_torussim        = 24,
        do_template_torussim   = False, # create "compact" template cube for long-baseline simobserve
        do_simint_torussim     = False, # sim C-10 band 9
        do_imaging_torussim    = False, # imaging sim ms
        #
        ########
        # plot #
        ########
        # plot
        plot_config            = False,
        # calc
        calc_collectingarea    = False,
        ):
        """
        This method runs all the methods which will create figures in the white paper.
        """

        #############################
        # set ngc1097sim parameters #
        #############################
        # observed frequency
        self.observed_freq = observed_freq
        self.incenter      = str(observed_freq)+"GHz"

        # n1097sim_7m from tinteg_n1097sim
        tinteg      = str(float(tinteg_n1097sim))+"h"
        tintegstr   = tinteg.replace(".","p")
        this_target = self.project_n1097+"_"+tintegstr

        # determine LST and TP beam sizes
        lst_beam    = str(12.979 * 115.27120 / self.observed_freq)+"arcsec"
        tp_beam     = str(50.6   * 115.27120 / self.observed_freq)+"arcsec"
        lst30m_beam = str(21.631 * 115.27120 / self.observed_freq)+"arcsec"

        # define products
        cube_tp  = self.dir_ready+"outputs/"+self.n1097_sdimage_fullspec.replace(".image","_"+tintegstr+"7m.image")
        cube_lst = self.dir_ready+"outputs/"+self.n1097_lstimage_fullspec.replace(".image","_"+tintegstr+"7m.image")
        cube_7m  = self.dir_ready+"outputs/postprocess/"+this_target+"/"+this_target+"_7m_ci10_pbcorr_trimmed.image"

        ##################
        # run ngc1097sim #
        ##################
        if do_template_n1097sim==True:
            self.prepare_template_n1097sim()

        if do_simACA_n1097sim==True:
            self.simaca_n1097sim(tinteg,tintegstr)

        if do_imaging_n1097sim==True:
            self.phangs_pipeline_imaging(
                this_proj=self.project_n1097,
                this_array="7m",
                this_target=this_target,
                )

        if do_simTP_n1097sim==True:
            self.simtp_n1097sim(tp_beam,tintegstr,dryrun_simSD)

        if do_simLST_n1097sim==True:
            self.simlst_n1097sim(lst_beam,tp_beam,tintegstr,dryrun_simSD)
            self.simlst_n1097sim(lst30m_beam,tp_beam,tintegstr,True)

        if do_feather==True:
            self.do_feather(cube_7m,cube_tp,self.n1097_feather_tp_7m,-1)
            self.do_feather(cube_7m,cube_lst,self.n1097_feather_lst_7m,-1)

        if do_simACA_LST_n1097sim==True:
            self.do_simaca_lst_n1097sim(tinteg,tintegstr)

        ###########################
        # set torussim parameters #
        ###########################
        tinteg      = str(float(tinteg_torussim))+"h"
        tintegstr   = tinteg.replace(".","p")
        this_target = self.project_torus+"_"+tintegstr

        ################
        # run torussim #
        ################
        if do_template_torussim==True:
            self.prepare_template_torussim()

        if do_simint_torussim==True:
            self.sim12m_torussim(tinteg,tintegstr)

        if do_imaging_torussim==True:
            # stage instead of pipeline
            msname  = self.project_torus + "_12m_" + tintegstr + "."+self.config_c10.split("/")[-1].split(".cfg")[0]+".noisy.ms"
            ms_from = self.dir_ready + "ms/" + self.project_torus + "_12m_" + tintegstr + "/" + msname
            dir_to  = self.dir_ready + "outputs/imaging/" + this_target + "/"
            ms_to   = dir_to + this_target + "_12m_cont.ms"
            os.system("rm -rf " + ms_to)
            os.system("rm -rf " + dir_to)
            os.makedirs(dir_to)
            os.system("cp -r " + ms_from + " " + ms_to)

            # run
            self.phangs_pipeline_imaging(
                this_proj=self.project_torus,
                this_array="12m",
                this_target=this_target,
                do_cont=True,
                )

        ########
        # plot #
        ########
        # plot
        if plot_config==True:
            self.plot_config()

        # calc
        if calc_collectingarea==True:
            self.calc_collectingarea()

    ###################
    # sim12m_torussim #
    ###################

    def sim12m_torussim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.torus_template_file,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.torus_template_file,
            antennalist=self.config_c10,
            project=self.project_torus+"_12m_"+totaltimetint,
            totaltime=totaltime,
            incenter="693.9640232GHz",
            )

    #############################
    # prepare_template_torussim #
    #############################

    def prepare_template_torussim(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        #check_first(self.n1097_template_fullspec,taskname)

        # cleanup directories
        input_dir  = self.dir_ready + "inputs/"
        output_dir = self.dir_ready + "outputs/"
        ms_dir     = self.dir_ready + "ms/"
        #os.system("rm -rf " + input_dir)
        #os.system("rm -rf " + output_dir)
        #os.system("rm -rf " + ms_dir)
        if not glob.glob(input_dir):
            os.mkdir(input_dir)
        if not glob.glob(output_dir):
            os.mkdir(output_dir)
        if not glob.glob(ms_dir):
            os.mkdir(ms_dir)

        # assume ngc1068 torus
        rmaj_out     = str(10.  / 72.)+"arcsec" # arcsec, 10pc at ngc1068, Gamez-Rosas et al. 2022 Nature
        rmin_out     = str(1.74 / 72.)+"arcsec" # arcsec, 10pc at ngc1068, Gamez-Rosas et al. 2022 Nature
        pa           = '-50.0deg' # Gamez-Rosas et al. 2022 Nature
        totalflux    = 13.8 / 1000. * 345**3.8 / 693.9640232**3.8 # continuum flux (mJy) at 432um (693.9640232 GHz), Garcia-Burillo et al. 2017

        rmaj_in      = str(10.  / 5. / 72.)+"arcsec"
        rmin_in      = str(1.74 / 5. / 72.)+"arcsec"
        totalflux_in = -1 * totalflux / 5**2

        direction = "J2000 02h42m40.70912s -00d00m47.9449s" # ngc1068 decl = -00d00m47.859690204s
        mycl.done()
        mycl.addcomponent(dir=direction, flux=totalflux, fluxunit='Jy', freq='693.9640232GHz', shape="disk", 
                        majoraxis=rmaj_out, minoraxis=rmin_out, positionangle=pa)
        mycl.addcomponent(dir=direction, flux=totalflux_in, fluxunit='Jy', freq='693.9640232GHz', shape="disk", 
                        majoraxis=rmaj_in, minoraxis=rmin_in, positionangle=pa)
        #
        myia.fromshape("torus.im",[256,256,1,1],overwrite=True)
        cs=myia.coordsys()
        cs.setunits(['rad','rad','','Hz'])
        cell_rad=myqa.convert(myqa.quantity("0.0005arcsec"),"rad")['value']
        cs.setincrement([-cell_rad,cell_rad],'direction')
        cs.setreferencevalue([myqa.convert("2.7113080889h",'rad')['value'],myqa.convert("-0.01331803deg",'rad')['value']],type="direction")
        cs.setreferencevalue("345GHz",'spectral')
        cs.setincrement('7.5GHz','spectral')
        myia.setcoordsys(cs.torecord())
        myia.setbrightnessunit("Jy/pixel")
        myia.modify(mycl.torecord(),subtract=False)
        exportfits(imagename='torus.im',fitsimage=input_dir+self.torus_template_file,overwrite=True)

        myia.close()
        mycl.close()
        os.system("rm -rf torus.im")

    ##########################
    # do_simaca_lst_n1097sim #
    ##########################

    def do_simaca_lst_n1097sim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1097_template_fullspec,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.n1097_template_fullspec,
            antennalist=self.config_7m_lst,
            project=self.project_n1097+"_LSTconnected_7m_"+totaltimetint,
            totaltime=totaltime,
            incenter=self.incenter,
            )

    #############
    # do_tp2vis #
    #############

    def do_tp2vis(self,sdimage,vis,outfile):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(lowres,taskname)

        # calculate hex ptg table (e.g., J2000 05h39m45.660s -70d07m57.524s)
        # use matplotlib.hexbin?
        print("# TBE.")

        # run tp2vis
        execfile("tp2vis.py")
        #tp2vis('tp.im','tp.ms','12m.ptg',rms=0.67)

    ##############
    # do_feather #
    ##############

    def do_feather(self,highres,lowres,outfile,effdishdiam):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(lowres,taskname)
        check_first(highres)

        os.system("rm -rf " + outfile)
        feather(
            imagename = outfile,
            highres = highres,
            lowres = lowres,
            effdishdiam = effdishdiam,
            )

    #######################
    # calc_collectingarea #
    #######################

    def calc_collectingarea(self):
        """
        """

        # alma
        area_12m_array = 50 * (12/2.)**2 * np.pi
        area_7m_array  = 12 * (7/2.)**2 * np.pi
        area_tp_array  = 4 * (12/2.)**2 * np.pi
        area_alma      = 50 * (12/2.)**2 * np.pi + 12 * (7/2.)**2 * np.pi + 4 * (12/2.)**2 * np.pi
        area_aca       = 12 * (7/2.)**2 * np.pi + 4 * (12/2.)**2 * np.pi

        # LST
        area_lst       = (50/2.)**2 * np.pi

        # ratio to LST and str
        print_12m  = str(int(np.round(area_12m_array))) + " m^2 (this/lst = " + str(np.round(area_12m_array/area_lst,2)) + ")"
        print_7m   = str(int(np.round(area_7m_array))) + " m^2 (this/lst = " + str(np.round(area_7m_array/area_lst,2)) + ")"
        print_tp   = str(int(np.round(area_tp_array))) + " m^2 (this/lst = " + str(np.round(area_tp_array/area_lst,2)) + ")"
        print_alma = str(int(np.round(area_alma))) + " m^2 (this/lst = " + str(np.round(area_alma/area_lst,2)) + ")"
        print_aca  = str(int(np.round(area_aca))) + " m^2 (this/lst = " + str(np.round(area_aca/area_lst,2)) + ")"

        print_lst  = str(int(np.round(area_lst))) + " m^2"
        print_lst_12m  = str(int(np.round(area_lst+area_12m_array))) + " m^2 (this/12m = " + str(np.round((area_lst+area_12m_array)/area_12m_array,2)) + ")"
        print_lst_alma = str(int(np.round(area_lst+area_alma))) + " m^2 (this/alma = " + str(np.round((area_lst+area_alma)/area_alma,2)) + ")"
        print_lst_aca  = str(int(np.round(area_lst+area_aca))) + \
            " m^2 (this/aca = " + str(np.round((area_lst+area_aca)/area_aca,2)) + \
            ", this/lst = " + str(np.round((area_lst+area_aca)/area_lst,2)) + ")"

        print("#################################################")
        print("### Array collecting area and ratio")
        print("# ALMA 12m-array     = " + print_12m)
        print("# ALMA  7m-array     =  " + print_7m)
        print("# ALMA  TP-array     =  " + print_tp)
        print("# ")
        print("# ALMA 12m+7m+TP     = " + print_alma)
        print("# ACA 7m+TP          =  " + print_aca)
        print("#")
        print("# LST                = " + print_lst)
        print("#")
        print("# LST+ALMA 12m       = " + print_lst_12m)
        print("# LST+ALMA 12m+7m+TP = " + print_lst_alma)
        print("# LST+ACA 7m+TP      = " + print_lst_aca)
        print("##################################################")

        #################################################
        ### Array collecting area and ratio
        # ALMA 12m-array     = 5655 m^2 (this/lst = 2.88)
        # ALMA  7m-array     =  462 m^2 (this/lst = 0.24)
        # ALMA  TP-array     =  452 m^2 (this/lst = 0.23)
        # 
        # ALMA 12m+7m+TP     = 6569 m^2 (this/lst = 3.35)
        # ACA 7m+TP          =  914 m^2 (this/lst = 0.47)
        #
        # LST                = 1963 m^2
        #
        # LST+ALMA 12m       = 7618 m^2 (this/12m = 1.35)
        # LST+ALMA 12m+7m+TP = 8533 m^2 (this/alma = 1.3)
        # LST+ACA 7m+TP      = 2878 m^2 (this/aca = 3.15, this/lst = 1.47)
        ##################################################

    ###############
    # plot_config #
    ###############

    def plot_config(self):
        """
        Reference:
        http://math_research.uct.ac.za/~siphelo/admin/interferometry/4_Visibility_Space/4_4_1_UV_Coverage_UV_Tracks.html
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.config_c10,taskname)

        # define center as ACA
        x_cnt = -50.06162725-0.9
        y_cnt = -568.9553881-0.4
        decl  = -37.755 # 0=celestial equator, 90=north pole, -90=south pole
        tinteg = 1
        #lst_position = np.array([0,0,0]) # km/s
        lst_position = np.array([6.452141+0.1, 7.886675+0.1, -0.245131]) # km/s

        # get data
        data  = np.loadtxt(self.config_c10,"str")
        x_12m = data[:,0].astype(np.float32) / 1000. - x_cnt / 1000.
        y_12m = data[:,1].astype(np.float32) / 1000. - y_cnt / 1000.
        z_12m = data[:,2].astype(np.float32) / 1000.

        data  = np.loadtxt(self.config_7m,"str")
        x_7m  = data[:,0].astype(np.float32) / 1000. - x_cnt / 1000.
        y_7m  = data[:,1].astype(np.float32) / 1000. - y_cnt / 1000.
        z_7m  = data[:,2].astype(np.float32) / 1000.

        # get dist and angle: alma-alma baselines
        this_data = np.c_[x_12m.flatten(),y_12m.flatten(),z_12m.flatten()]
        #this_data = np.c_[x_12m.flatten()+[lst_position[0]],y_12m.flatten()+[lst_position[1]],z_12m.flatten()+[lst_position[2]]]
        u_alma, v_alma = self._get_baselines(this_data,this_data,decl=decl,tinteg=tinteg)
        u1_lst_center, v1_lst_center = self._get_baselines([lst_position],this_data,decl=decl,tinteg=tinteg)
        u2_lst_center, v2_lst_center = self._get_baselines(this_data,[lst_position],decl=decl,tinteg=tinteg)

        ###############################
        # plot: C-10 antenna position #
        ###############################
        ad    = [0.215,0.83,0.10,0.90]
        xlim  = [-10,10]
        ylim  = [-10,10]
        title = "Antenna positions"
        xlabel = "East-West (km)"
        ylabel = "North-South (km)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(6.452141+0.1, 7.886675+0.1, color="tomato", marker="*", lw=0, s=900)
        ax1.scatter(8, -8, color="tomato", marker="*", lw=0, s=900)
        ax1.scatter(-8, 0, color="tomato", marker="*", lw=0, s=900)
        ax1.scatter(0, 0, color="tomato", marker="*", lw=0, s=900)
        ax1.scatter(x_12m, y_12m, color="grey", lw=0, s=100)
        ax1.scatter(x_7m, y_7m, color="deepskyblue", lw=0, s=100)

        # text
        ax1.text(0.05,0.92, "ALMA 12-m array", color="grey", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "ACA 7-m array", color="deepskyblue", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.82, "LSTsim 50-m", color="tomato", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_config_12m)
        plt.savefig(self.outpng_config_12m, dpi=self.fig_dpi)

        #############################
        # plot: 7m antenna position #
        #############################
        ad    = [0.215,0.83,0.10,0.90]
        dev   = 100
        xlim  = [-dev,dev]
        ylim  = [-dev+50,dev+50]
        title = "Morita Array (7m+TP) and LST$_{\mathrm{sim,50m}}$ positions" # $\\Sigma_{\mathrm{H_2}}$
        xlabel = "East-West (m)"
        ylabel = "North-South (m)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.rcParams["font.family"] = "Calibri"
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, None, xlim, ylim, title, xlabel, ylabel, adjust=ad)

        for i in range(len(x_7m)):
            this_x = x_7m[i] * 1000.
            this_y = y_7m[i] * 1000.
            antenna = patches.Ellipse(xy=(this_x,this_y), width=7.0,
                height=7.0, angle=0, fill=True, color="deepskyblue", edgecolor="deepskyblue",
                alpha=1.0, lw=0)
            ax1.add_patch(antenna)

        # LST
        antenna = patches.Ellipse(xy=(-50.06162725-x_cnt,-457.2313425-y_cnt), width=50.0,
            height=50.0, angle=0, fill=True, color="tomato", edgecolor="tomato",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)

        # TPx4
        antenna = patches.Ellipse(xy=(-36.00,27.50), width=12.0,
            height=12.0, angle=0, fill=True, color="grey", edgecolor="grey",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)
        antenna = patches.Ellipse(xy=(-31.00,-32.00), width=12.0,
            height=12.0, angle=0, fill=True, color="grey", edgecolor="grey",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)
        antenna = patches.Ellipse(xy=(35.50,-26.00), width=12.0,
            height=12.0, angle=0, fill=True, color="grey", edgecolor="grey",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)
        antenna = patches.Ellipse(xy=(36.00,29.50), width=12.0,
            height=12.0, angle=0, fill=True, color="grey", edgecolor="grey",
            alpha=1.0, lw=0)
        ax1.add_patch(antenna)

        # LST at the center
        antenna = patches.Ellipse(xy=(0,0), width=50.0,
            height=50.0, angle=0, fill=False, color="tomato", edgecolor="tomato",
            alpha=0.7, lw=2, ls="dashed")
        ax1.add_patch(antenna)

        # ann
        xpos = x_7m[np.argmax(y_7m*1000.)] * 1000.
        print(xpos)
        ax1.plot([xpos,xpos],[np.max(y_7m*1000.),-457.2313425-y_cnt],"-",color="black",lw=2)
        ax1.text(xpos+1,77,"90m $\simeq$ 50m x cot 30$^{\circ}$",ha="left",fontsize=17)
        ax1.text(xpos+1,70,"(No shadowing until El. $\simeq$ 30$^{\circ}$)",ha="left",fontsize=17)

        # text
        ax1.text(0.05,0.92, "ACA 7-m array", color="deepskyblue", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "ACA TP array", color="grey", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.82, "LST$_{\mathrm{sim,50m}}$", color="tomato", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_config_7m)
        plt.savefig(self.outpng_config_7m, dpi=self.fig_dpi)

        ############
        # plot: uv #
        ############
        ad    = [0.215,0.83,0.10,0.90]
        xlim  = [-20,20]
        ylim  = [-20,20]
        title = "$u-v$ coverage"
        xlabel = "East-West (km)"
        ylabel = "North-South (km)"

        fig = plt.figure(figsize=(13,10))
        gs  = gridspec.GridSpec(nrows=10, ncols=10)
        ax1 = plt.subplot(gs[0:10,0:10])
        plt.subplots_adjust(left=ad[0], right=ad[1], bottom=ad[2], top=ad[3])
        myax_set(ax1, "both", xlim, ylim, title, xlabel, ylabel, adjust=ad)

        ax1.scatter(u_alma, v_alma, color="grey", lw=0, s=5, alpha=0.5)
        ax1.scatter(u1_lst_center, v1_lst_center, color="tomato", lw=0, s=5, alpha=0.5)
        ax1.scatter(u2_lst_center, v2_lst_center, color="tomato", lw=0, s=5, alpha=0.5)

        # text
        ax1.text(0.05,0.92, "Baselines: ALMA - ALMA", color="grey", weight="bold", transform=ax1.transAxes)
        ax1.text(0.05,0.87, "Baselines: ALMA - LSTsim", color="tomato", weight="bold", transform=ax1.transAxes)

        # save
        plt.subplots_adjust(hspace=.0)
        os.system("rm -rf " + self.outpng_uv_alma_lst1)
        plt.savefig(self.outpng_uv_alma_lst1, dpi=self.fig_dpi)

    ###########################
    # phangs_pipeline_imaging #
    ###########################

    def phangs_pipeline_imaging(self,this_proj,this_array,this_target,do_cont=False):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        #check_first(self.template_fullspec,taskname)

        # prepare dir_cleanmask = dir_singledish
        dir_cleanmask = self.dir_ready + "outputs/"
        if not glob.glob(dir_cleanmask):
            os.mkdir(dir_cleanmask)

        # set piepline
        master_key = self.dir_pipeline + "master_key.txt"

        pipepath = os.environ.get('PHANGSPIPE')
        if pipepath is not None:
            sys.path.append(os.environ.get('PHANGSPIPE'))
        else:
            sys.path.append(os.getcwd())

        # Check casa environment by importing CASA-only packages
        try:
            import taskinit
        except ImportError:
            print('Please run this script inside CASA!')
            sys.exit()

        # Set the logging
        from phangsPipeline import phangsLogger as pl
        reload(pl)
        pl.setup_logger(level='DEBUG', logfile=None)
        # Imports

        from phangsPipeline import handlerKeys as kh
        from phangsPipeline import handlerVis as uvh
        from phangsPipeline import handlerImaging as imh
        from phangsPipeline import handlerPostprocess as pph

        # Reloads for debugging
        reload(kh)
        reload(uvh)
        reload(imh)
        reload(pph)

        # Initialize key handler
        this_kh  = kh.KeyHandler(master_key = master_key)
        this_uvh = uvh.VisHandler(key_handler = this_kh)
        this_imh = imh.ImagingHandler(key_handler = this_kh)
        this_pph = pph.PostProcessHandler(key_handler= this_kh)
        dry_run_key = False
        this_uvh.set_dry_run(dry_run_key)
        this_imh.set_dry_run(dry_run_key)
        this_pph.set_dry_run(dry_run_key)

        if do_cont==False:
            set_no_cont_products = True
        else:
            set_no_cont_products = False

        # set handlers
        for this_hander in [this_uvh,this_imh,this_pph]:
            this_hander.set_targets(only=[this_target])
            this_hander.set_line_products(only=["ci10"])
            this_hander.set_no_cont_products(set_no_cont_products)
            this_hander.set_no_line_products(False)
            this_hander.set_interf_configs(only=[this_array])

        # run piepline
        if do_cont==False:
            this_uvh.loop_stage_uvdata(\
                    do_copy           = True,
                    do_remove_staging = True,
                    do_contsub        = False,
                    do_extract_line   = True,
                    do_extract_cont   = False,
                    overwrite         = False,
                    )

        this_imh.loop_imaging(\
                do_dirty_image          = True,
                do_revert_to_dirty      = False,
                do_read_clean_mask      = False,
                do_multiscale_clean     = False,
                do_revert_to_multiscale = False,
                do_singlescale_mask     = False,
                do_singlescale_clean    = True,
                do_export_to_fits       = False,
                extra_ext_in            = '',
                extra_ext_out           = '',
                )
        this_pph.loop_postprocess(\
                do_prep               = True,
                do_feather            = False,
                do_mosaic             = True,
                do_cleanup            = True,
                do_summarize          = True,
                # feather_apod          = True,
                feather_noapod        = True,
                # feather_before_mosaic = False,
                # feather_after_mosaic  = False,
                )

    ###################
    # simlst_n1097sim #
    ###################

    def simlst_n1097sim(self,lst_res="3.04arcsec",tp_res="11.8arcsec",totaltimetint="2p0h",dryrun=True):
        """
        1. measure rms_7m (= rms level of the 7m-only cleaned image)
        2. measure tp-tinteg (= 7m-tinteg * 1.7)
        3. provide tp-tinteg and 7m-tinteg to ASC to calculate achievable sensitivity and TP/7m sensitivity ratio (= 2.326550129182734/1.4789569480812979)
        4. scale rms_7m to get rms_tp
        5. rms_tp must be also scaled by LST/TP beam area ratio (to match the sensitivity in K units, not Jy/beam)
        6. then, scale to match sensitivity "after" convolution a common TP beam.
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.dir_ready+"inputs/"+self.n1097_template_fullspec,taskname)

        image_7m = self.dir_ready + "outputs/imaging/"+self.project_n1097+"_"+totaltimetint + "/"+self.project_n1097+"_"+totaltimetint+"_7m_ci10.image"
        rms_7m = 0.08273 # measure_rms(image_7m, snr=3.0,rms_or_p84 = "p84")
        rms_lst = rms_7m * 2.326550129182734/1.4789569480812979 * float(lst_res.replace("arcsec",""))**2 / float(tp_res.replace("arcsec",""))**2
        rms_lst = rms_lst * float(tp_res.replace("arcsec","")) / float(lst_res.replace("arcsec",""))
        rms_lst_K = 1.222e6 * float(lst_res.replace("arcsec",""))**-2 * self.observed_freq**-2 * rms_lst

        # calc pointing number
        header       = imhead(self.dir_ready+"inputs/"+self.n1097_template_fullspec,mode="list")
        area_in_as   = (header["shape"][0]*header["cdelt2"]*3600*180/np.pi) * (header["shape"][1]*header["cdelt2"]*3600*180/np.pi)
        one_hex_as   = (float(lst_res.replace("arcsec",""))/4.0)**2 * 6/np.sqrt(3) # hex with 1/4-beam length
        num_pointing = int(np.ceil(area_in_as / one_hex_as))

        # ACA TP sim at 492.16065100 GHz
        print("### LST observations")
        print("# achieved 7m sensitivity (Jy/b)  = " + str(np.round(rms_7m,5)))
        print("# required LST sensntivity (Jy/b) = " + str(np.round(rms_lst,5)))
        print("# sensitivity per pointing (K)    = " + str(np.round(rms_lst_K,5)))
        print("# beam size (arcsec)              = " + str(np.round(float(lst_res.replace("arcsec","")),2)))
        print("# number of pointing              = " + str(num_pointing))
        print("# survey area (degree^2)          = " + str(np.round(area_in_as/3600.**2,8)))
        print("#")

        # run
        if dryrun==False:
            simtp(
                working_dir=self.dir_ready,
                template_fullspec=self.n1097_template_fullspec,
                sdimage_fullspec=self.n1097_lstimage_fullspec.replace(".image","_"+totaltimetint+"7m.image"),
                sdnoise_image=self.n1097_lstnoise_image.replace(".image","_"+totaltimetint+"7m.image"),
                singledish_res=lst_res,
                singledish_noise=rms_lst, # Jy/beam at final res
                observed_freq=self.observed_freq*1e9,
                )
        else:
            print("# skipped simtp as dryrun==True")
            print("#")

    ##################
    # simtp_n1097sim #
    ##################

    def simtp_n1097sim(self,singledish_res="11.8arcsec",totaltimetint="2p0h",dryrun=True):
        """
        1. measure rms_7m (= rms level of the 7m-only cleaned image)
        2. measure tp-tinteg (= 7m-tinteg * 1.7)
        3. provide tp-tinteg and 7m-tinteg to ASC to calculate achievable sensitivity and TP/7m sensitivity ratio (= 2.326550129182734/1.4789569480812979)
        4. scale rms_7m to get rms_tp
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.dir_ready+"inputs/"+self.n1097_template_fullspec,taskname)

        image_7m = self.dir_ready + "outputs/imaging/"+self.project_n1097+"_"+totaltimetint + "/"+self.project_n1097+"_"+totaltimetint+"_7m_ci10.image"
        rms_7m = 0.08273 # measure_rms(image_7m, snr=3.0,rms_or_p84 = "p84")
        rms_tp = rms_7m * 2.326550129182734/1.4789569480812979 # ASC TPrms/7mrms ratio
        rms_tp_K = 1.222e6 * float(singledish_res.replace("arcsec",""))**-2 * self.observed_freq**-2 * rms_tp

        # calc pointing number
        header       = imhead(self.dir_ready+"inputs/"+self.n1097_template_fullspec,mode="list")
        area_in_as   = (header["shape"][0]*header["cdelt2"]*3600*180/np.pi) * (header["shape"][1]*header["cdelt2"]*3600*180/np.pi)
        one_hex_as   = (float(singledish_res.replace("arcsec",""))/4.0)**2 * 6/np.sqrt(3) # hex with 1/4-beam length
        num_pointing = int(np.ceil(area_in_as / one_hex_as))

        """ failed to calculate SD sensitivity based on 7m observing time...
        # tinteg per pointing
        # sensitivity at 492.16065100 GHz based on ASC (1hr = 3.033450239598523 Jy/beam)
        tinteg_per_pointing = float(totaltime.replace("h","")) / num_pointing
        singledish_noise_per_pointing = 3.033450239598523 / 1000. / tinteg_per_pointing**2
        singledish_noise_per_pointing_K = 1.222e6 * float(singledish_res.replace("arcsec",""))**-2 * self.observed_freq**-2 * singledish_noise_per_pointing
        """

        # ACA TP sim at 492.16065100 GHz
        print("### ACA TP observations")
        print("# achieved 7m sensitivity (Jy/b) = " + str(np.round(rms_7m,5)))
        print("# required TP sensntivity (Jy/b) = " + str(np.round(rms_tp,5)))
        print("# sensitivity per pointing (K)   = " + str(np.round(rms_tp_K,5)))
        print("# beam size (arcsec)             = " + str(np.round(float(singledish_res.replace("arcsec","")),2)))
        print("# number of pointing             = " + str(num_pointing))
        print("# survey area (degree^2)         = " + str(np.round(area_in_as/3600.**2,8)))
        print("#")

        # run
        if dryrun==False:
            simtp(
                working_dir=self.dir_ready,
                template_fullspec=self.n1097_template_fullspec,
                sdimage_fullspec=self.n1097_sdimage_fullspec.replace(".image","_"+totaltimetint+"7m.image"),
                sdnoise_image=self.n1097_sdnoise_image.replace(".image","_"+totaltimetint+"7m.image"),
                singledish_res=singledish_res,
                singledish_noise=rms_tp, # Jy/beam at final res
                observed_freq=self.observed_freq*1e9,
                )
        else:
            print("# skipped simtp as dryrun==True")
            print("#")

    ###################
    # simaca_n1097sim #
    ###################

    def simaca_n1097sim(self,totaltime="2.0h",totaltimetint="2p0h"):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1097_template_fullspec,taskname)

        run_simobserve(
            working_dir=self.dir_ready,
            template=self.n1097_template_fullspec,
            antennalist=self.config_7m,
            project=self.project_n1097+"_7m_"+totaltimetint,
            totaltime=totaltime,
            incenter=self.incenter,
            )

    #############################
    # prepare_template_n1097sim #
    #############################

    def prepare_template_n1097sim(self):
        """
        """

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.n1097_template_file,taskname)

        gen_cube(
            template_dir=self.dir_raw,
            template_file=self.n1097_template_file,
            template_mask=self.n1097_template_mask,
            working_dir=self.dir_ready,
            template_in_jypix=self.n1097_template_in_jypix,
            template_clipped=self.n1097_template_clipped,
            template_mask_imported=self.n1097_template_mask_imported,
            template_rotated=self.n1097_template_rotated,
            template_shrunk=self.n1097_template_shrunk,
            template_fullspec=self.n1097_template_fullspec,
            template_fullspec_div3=self.n1097_template_fullspec_div3,
            template_fullspec_div5=self.n1097_template_fullspec_div5,
            template_fullspec_div10=self.n1097_template_fullspec_div10,
            template_fullspec_div30=self.n1097_template_fullspec_div30,
            template_fullspec_div100=self.n1097_template_fullspec_div100,
            template_withcont=self.n1097_template_withcont,
            template_withcont_div3=self.n1097_template_withcont_div3,
            template_withcont_div5=self.n1097_template_withcont_div5,
            template_withcont_div10=self.n1097_template_withcont_div10,
            template_withcont_div30=self.n1097_template_withcont_div30,
            template_withcont_div100=self.n1097_template_withcont_div100,
            pa=self.image_rot_n1097sim, # rotation angle
            )

    ##################
    # _get_baselines #
    ##################

    def _get_baselines(self,x,y,decl=60,tinteg=0):
        """
        """
        latitude = np.radians(-67.755) # degree, alma site

        list_dist  = []
        list_theta = []
        list_phi   = []
        combinations = itertools.product(x,y)
        for comb in combinations:
            this_vec = comb[0] - comb[1]

            this_d = np.linalg.norm(this_vec)
            list_dist.append(this_d)

            this_a = np.degrees(np.arcsin(this_vec[2] / this_d))
            list_theta.append(this_a)

            this_h = np.degrees(np.arctan2(this_vec[0], this_vec[1]))
            list_phi.append(this_h)

        l = np.array(list_dist)
        t = np.radians( np.array(list_theta) )
        p = np.radians( np.array(list_phi) )
        dec = np.radians(decl)

        X = l*(np.cos(latitude)*np.sin(t) - np.sin(latitude)*np.cos(t)*np.cos(p)) # l*np.sin(t)*np.cos(p)
        Y = l*np.cos(t)*np.sin(p)
        Z = l*(np.sin(latitude)*np.sin(t) + np.cos(latitude)*np.cos(t)*np.cos(p)) # l*np.cos(t)

        # output
        list_u = []
        list_v = []
        trange = np.r_[np.arange(-tinteg/24.*360/2.0, tinteg/24.*360/2.0, 0.1), tinteg/24.*360/2.0]
        for this_t in trange:
            H = np.radians(this_t)
 
            this_u = X*np.sin(H) + Y*np.cos(H)
            this_v = -X*np.sin(dec)*np.cos(H) + Y*np.sin(dec)*np.sin(H) + Z*np.cos(dec)

            # output
            list_u = np.r_[list_u, this_u]
            list_v = np.r_[list_v, this_v]

        return list_u, list_v

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

######################
# end of ToolsLSTSim #
######################