import os, sys, glob
import numpy as np

from mycasa_sampling import *
from mycasa_lowess import *
from mycasa_tasks import *
from mycasa_plots import *
from mycasa_pca import *

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

        self.outfits_b3 = self.dir_ready + self._read_key("outfits_b3")
        self.outfits_b4 = self.dir_ready + self._read_key("outfits_b4")
        self.outfits_b6 = self.dir_ready + self._read_key("outfits_b6")
        self.outfits_b7 = self.dir_ready + self._read_key("outfits_b7")
        self.outfits_b8 = self.dir_ready + self._read_key("outfits_b8")
        self.outfits_b9 = self.dir_ready + self._read_key("outfits_b9")

    def _set_input_param(self):
        """
        """

        self.scale_pc  = float(self._read_key("scale", "gal"))
        self.scale_kpc = self.scale_pc / 1000.
        self.redshift  = float(self._read_key("redshift", "gal"))

        self.ra        = self._read_key("ra", "gal")
        self.dec       = self._read_key("dec", "gal")
        c = Skycoord(self.ra+" "+self.dec,unit=(u.hourangle, u.deg))
        self.ra        = c.ra.degree
        self.dec       = c.dec.degree
        self.ra_str    = str(self.ra) + "deg"
        self.dec_str   = str(self.dec) + "deg"

        self.imsize    = 85.0

    def _set_output_txt_png(self):
        """
        """

        self.outpng_b3 = self.dir_products + self._read_key("outpng_b3")
        self.outpng_b4 = self.dir_products + self._read_key("outpng_b4")
        self.outpng_b6 = self.dir_products + self._read_key("outpng_b6")
        self.outpng_b7 = self.dir_products + self._read_key("outpng_b7")
        self.outpng_b8 = self.dir_products + self._read_key("outpng_b8")
        self.outpng_b9 = self.dir_products + self._read_key("outpng_b9")

    ######################
    # run_ngc6240_contin #
    ######################

    def run_ngc6240_contin(
        self,
        # analysis
        do_prepare       = False,
        # plot figures in paper
        plot_showcase    = False,
        # calc
        calc_image_stats = False,
        ):
        """
        This method runs all the methods which will create figures in the paper.
        """

        # analysis
        if do_prepare==True:
            self.align_maps()

        # plot
        if plot_showcase==True:
            self.showcase()

        # calc
        if calc_image_stats==True:
            self.calc_image_stats()

    ############
    # showcase #
    ############

    def showcase(self):
        """
        """

        beamstr="0p7as"

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.outfits_b3.replace("???",beamstr),taskname)

        #self.outfits_b3.replace("???",beamstr)

        scalebar = 500. / self.scale_pc
        label_scalebar = "500 pc"

        levels_cont1 = [0.05, 0.1, 0.2, 0.4, 0.8, 0.96]
        width_cont1  = [1.0]
        set_bg_color = "white" # cm.rainbow(0)

        # plot b3
        this_map = self.outfits_b3.replace("???",beamstr)
        this_out = self.outpng_b3
        myfig_fits2png(
            imcolor=this_map,
            outfile=this_out,
            imcontour1=this_map,
            imsize_as=self.imsize,
            ra_cnt=self.ra_str,
            dec_cnt=self.dec_str,
            levels_cont1=levels_cont1,
            width_cont1=width_cont1,
            set_title="110 GHz continuum (Band 3)",
            colorlog=False,
            scalebar=scalebar,
            label_scalebar=label_scalebar,
            set_cbar=True,
            label_cbar="Jy beam$^{-1}$",
            clim=None,
            set_bg_color=None,
            #numann="13co",
            )

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
        run_immath_one(self.pb_b3,inmask,"iif(IM0>=0.99,1,0)")
        outmask = "mask.image2"
        run_immath_one(self.pb_b3,outmask,"iif(IM0<0.99,1,0)")

        # measure diameter of the mask
        data,_ = imval_all(inmask)
        data   = data["data"] * data["mask"]
        data   = np.array(data.flatten())
        numpix = len(data[data>0])
        pix    = abs(imhead(inmask)["incr"][0]) * 3600 * 180/np.pi
        #
        fov_as = numpix * pix**2

        # measure b3 stats
        print("######################")
        print("# calculate B3 stats #")
        print("######################")
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
        print("######################")
        print("# calculate B4 stats #")
        print("######################")
        this_map = self.map_b4
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
        print("######################")
        print("# calculate B6 stats #")
        print("######################")
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

        # measure b7 stats
        print("######################")
        print("# calculate B7 stats #")
        print("######################")
        this_map = self.map_b7
        run_imregrid(inmask,this_map,inmask+"_b7")
        run_imregrid(outmask,this_map,outmask+"_b7")
        #
        run_immath_two(this_map,inmask+"_b7",this_map+"_in","IM0*IM1")
        run_immath_two(this_map,outmask+"_b7",this_map+"_out","IM0*IM1")
        data_in,_  = imval_all(this_map+"_in")
        data_in    = data_in["data"] * data_in["mask"]
        data_in    = data_in.flatten()
        data_in    = data_in[data_in!=0]
        data_out,_ = imval_all(this_map+"_out")
        data_out   = data_out["data"] * data_out["mask"]
        data_out   = data_out.flatten()
        data_out   = data_out[data_out!=0]
        #
        b7_beam    = beam_area(this_map)
        b7_max     = np.max(data_in)
        b7_sum     = np.sum(data_in) / b7_beam
        b7_rms     = np.sqrt(np.mean(np.square(data_out)))

        # measure b8 stats
        print("######################")
        print("# calculate B8 stats #")
        print("######################")
        this_map = self.map_b8
        run_imregrid(inmask,this_map,inmask+"_b8")
        run_imregrid(outmask,this_map,outmask+"_b8")
        #
        run_immath_two(this_map,inmask+"_b8",this_map+"_in","IM0*IM1")
        run_immath_two(this_map,outmask+"_b8",this_map+"_out","IM0*IM1")
        data_in,_  = imval_all(this_map+"_in")
        data_in    = data_in["data"] * data_in["mask"]
        data_in    = data_in.flatten()
        data_in    = data_in[data_in!=0]
        data_out,_ = imval_all(this_map+"_out")
        data_out   = data_out["data"] * data_out["mask"]
        data_out   = data_out.flatten()
        data_out   = data_out[data_out!=0]
        #
        b8_beam    = beam_area(this_map)
        b8_max     = np.max(data_in)
        b8_sum     = np.sum(data_in) / b8_beam
        b8_rms     = np.sqrt(np.mean(np.square(data_out)))

        # measure b9 stats
        print("######################")
        print("# calculate B9 stats #")
        print("######################")
        this_map = self.map_b9
        run_imregrid(inmask,this_map,inmask+"_b9")
        run_imregrid(outmask,this_map,outmask+"_b9")
        #
        run_immath_two(this_map,inmask+"_b9",this_map+"_in","IM0*IM1")
        run_immath_two(this_map,outmask+"_b9",this_map+"_out","IM0*IM1")
        data_in,_  = imval_all(this_map+"_in")
        data_in    = data_in["data"] * data_in["mask"]
        data_in    = data_in.flatten()
        data_in    = data_in[data_in!=0]
        data_out,_ = imval_all(this_map+"_out")
        data_out   = data_out["data"] * data_out["mask"]
        data_out   = data_out.flatten()
        data_out   = data_out[data_out!=0]
        #
        b9_beam    = beam_area(this_map)
        b9_max     = np.max(data_in)
        b9_sum     = np.sum(data_in) / b9_beam
        b9_rms     = np.sqrt(np.mean(np.square(data_out)))

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
        print("#")
        print("B6 max = ",b6_max,"Jy/b")
        print("B6 tot = ",b6_sum,"Jy")
        print("B6 rms = ",b6_rms,"Jy/b")
        print("B6 peak S/N = ",b6_max/b6_rms,"Jy/b")
        print("#")
        print("B7 max = ",b7_max,"Jy/b")
        print("B7 tot = ",b7_sum,"Jy")
        print("B7 rms = ",b7_rms,"Jy/b")
        print("B7 peak S/N = ",b7_max/b7_rms,"Jy/b")
        print("#")
        print("B8 max = ",b8_max,"Jy/b")
        print("B8 tot = ",b8_sum,"Jy")
        print("B8 rms = ",b8_rms,"Jy/b")
        print("B8 peak S/N = ",b8_max/b8_rms,"Jy/b")
        print("#")
        print("B9 max = ",b9_max,"Jy/b")
        print("B9 tot = ",b9_sum,"Jy")
        print("B9 rms = ",b9_rms,"Jy/b")
        print("B9 peak S/N = ",b9_max/b9_rms,"Jy/b")

    ##############
    # align_maps #
    ##############

    def align_maps(self,targetbeam=0.7):
        """
        """

        template = self.map_b3
        beamstr  = str(targetbeam).replace(".","p") + "as"

        taskname = self.modname + sys._getframe().f_code.co_name
        check_first(self.map_b3,taskname)

        # process b3
        print("################")
        print("# align B3 map #")
        print("################")
        this_in  = self.map_b3
        this_pb  = self.pb_b3
        this_out = self.outfits_b3.replace("???",beamstr)
        os.system("rm -rf " + this_in + "_pbcor")
        impbcor(this_in,this_pb,this_in+"_pbcor")
        run_roundsmooth(this_in+"_pbcor",this_in+"_pbcor2",targetbeam,targetres=True,delin=True)
        run_exportfits(this_in+"_pbcor2",this_out,True,True,True)

        # process b4
        print("################")
        print("# align B4 map #")
        print("################")
        this_in  = self.map_b4
        this_pb  = self.pb_b4
        this_out = self.outfits_b4.replace("???",beamstr)
        os.system("rm -rf " + this_in + "_pbcor")
        impbcor(this_in,this_pb,this_in+"_pbcor")
        run_roundsmooth(this_in+"_pbcor",this_in+"_pbcor2",targetbeam,targetres=True,delin=True)
        run_imregrid(this_in+"_pbcor2",self.map_b3,this_in+"_pbcor3",delin=True)
        run_exportfits(this_in+"_pbcor3",this_out,True,True,True)

        # process b6
        print("################")
        print("# align B6 map #")
        print("################")
        this_in  = self.map_b6
        this_pb  = self.pb_b6
        this_out = self.outfits_b6.replace("???",beamstr)
        os.system("rm -rf " + this_in + "_pbcor")
        impbcor(this_in,this_pb,this_in+"_pbcor")
        run_roundsmooth(this_in+"_pbcor",this_in+"_pbcor2",targetbeam,targetres=True,delin=True)
        run_imregrid(this_in+"_pbcor2",self.map_b3,this_in+"_pbcor3",delin=True)
        run_exportfits(this_in+"_pbcor3",this_out,True,True,True)

        # process b7
        print("################")
        print("# align B7 map #")
        print("################")
        this_in  = self.map_b7
        this_pb  = self.pb_b7
        this_out = self.outfits_b7.replace("???",beamstr)
        os.system("rm -rf " + this_in + "_pbcor")
        impbcor(this_in,this_pb,this_in+"_pbcor")
        run_roundsmooth(this_in+"_pbcor",this_in+"_pbcor2",targetbeam,targetres=True,delin=True)
        run_imregrid(this_in+"_pbcor2",self.map_b3,this_in+"_pbcor3",delin=True)
        run_exportfits(this_in+"_pbcor3",this_out,True,True,True)

        # process b8
        print("################")
        print("# align B8 map #")
        print("################")
        this_in  = self.map_b8
        this_pb  = self.pb_b8
        this_out = self.outfits_b8.replace("???",beamstr)
        os.system("rm -rf " + this_in + "_pbcor")
        impbcor(this_in,this_pb,this_in+"_pbcor")
        run_roundsmooth(this_in+"_pbcor",this_in+"_pbcor2",targetbeam,targetres=True,delin=True)
        run_imregrid(this_in+"_pbcor2",self.map_b3,this_in+"_pbcor3",delin=True)
        run_exportfits(this_in+"_pbcor3",this_out,True,True,True)

        # process b9
        print("################")
        print("# align B9 map #")
        print("################")
        this_in  = self.map_b9
        this_pb  = self.pb_b9
        this_out = self.outfits_b9.replace("???",beamstr)
        os.system("rm -rf " + this_in + "_pbcor")
        impbcor(this_in,this_pb,this_in+"_pbcor")
        run_roundsmooth(this_in+"_pbcor",this_in+"_pbcor2",targetbeam,targetres=True,delin=True)
        run_imregrid(this_in+"_pbcor2",self.map_b3,this_in+"_pbcor3",delin=True)
        run_exportfits(this_in+"_pbcor3",this_out,True,True,True)

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