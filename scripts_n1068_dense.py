"""
Python class for the NGC 1068 dense gas vs. N2H+ project
using CASA.

history:
2021-07-06   created
2021-07-26   major updates
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
class ToolsDense():
    """
    Class for the NGC 1068 dense gas vs. N2H+ project.
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

        # intialize shuffling and stacking
        self.velimage  = None
        self.binimage  = None
        self.maskimage = None

        # import parameters
        if keyfile_fig is not None:
            self.modname = "ToolsDense."
            
            # get directories
            self.dir_proj = self._read_key("dir_proj")
            self.dir_rawc = self.dir_proj + self._read_key("dir_raw")
            self.dir_rawm = self.dir_proj + self._read_key("dir_raw_mom")
            self.dir_ready = self.dir_proj + self._read_key("dir_ready")
            self.dir_other = self.dir_proj + self._read_key("dir_other")
            self.dir_products = self.dir_proj + self._read_key("dir_products")
            self.dir_process = self.dir_proj + self._read_key("dir_process")

            self._create_dir(self.dir_ready)
            self._create_dir(self.dir_products)

            # create output png directories
            self.dir_hexmap_mom0  = self.dir_products + "hexmap_mom0/"
            self.dir_hexmap_ratio = self.dir_products + "hexmap_ratio/"
            self.dir_hexr_mom0    = self.dir_products + "hexradial_mom0/"
            self.dir_hexr_ratio   = self.dir_products + "hexradial_ratio/"
            self.dir_hexs_mom0    = self.dir_products + "hexscatter_mom0/"
            self.dir_hexs_ratio   = self.dir_products + "hexscatter_ratio/"
            self.dir_hexh_mom0    = self.dir_products + "hexhist_mom0/"
            self.dir_hexh_ratio   = self.dir_products + "hexhist_ratio/"
            self.dir_hexp         = self.dir_products + "hexpca/"

            self._create_dir(self.dir_hexmap_mom0)
            self._create_dir(self.dir_hexmap_ratio)
            self._create_dir(self.dir_hexr_mom0)
            self._create_dir(self.dir_hexr_ratio)
            self._create_dir(self.dir_hexs_mom0)
            self._create_dir(self.dir_hexs_ratio)
            self._create_dir(self.dir_hexh_mom0)
            self._create_dir(self.dir_hexh_ratio)
            self._create_dir(self.dir_hexp)

            # get ngc1068 properties
            self.scale_pc = float(self._read_key("scale", "gal"))
            l = self._read_key("ra_agn", "gal")
            self.ra_agn = float(l.replace("deg",""))
            l = self._read_key("dec_agn", "gal")
            self.dec_agn = float(l.replace("deg",""))
            
            # some numbers
            self.snr_mom = 3.0
            self.r_cnd = 3.0 * self.scale_pc / 1000. # kpc
            self.r_sbr = 10.0 * self.scale_pc / 1000. # kpc

            # get 0p8as maps
            l = glob.glob(self.dir_rawc + self._read_key("cubes_0p8as"))
            self.cubes_0p8as = [s for s in l if not "_co10_" in s]
            self.cubes_0p8as.sort()
            self.vel_0p8as = self.dir_rawm + self._read_key("velimage_0p8as")
            self.bin_av_0p8as = self.dir_other + self._read_key("bin_av_0p8as")
            self.bin_co_0p8as = self.dir_other + self._read_key("bin_co_0p8as")
            self.mask_0p8as = self.dir_rawc + self._read_key("mask_0p8as")

            l = self.dir_rawm + self._read_key("mom0_0p8as")
            self.mom0_0p8as = glob.glob(l)
            self.mom0_0p8as.sort()
            l = self.dir_rawm + self._read_key("emom0_0p8as")
            self.emom0_0p8as = glob.glob(l)
            self.emom0_0p8as.sort()

            self.txt_hex_0p8as = self.dir_ready + self._read_key("txt_hex_0p8as")
            self.beam_0p8as = 0.8
            self.grid_0p8as = 70
            
            # get 150pc maps
            l = glob.glob(self.dir_rawc + self._read_key("cubes_150pc"))
            self.cubes_150pc = [s for s in l if not "_co10_" in s]
            self.cubes_150pc.sort()
            self.vel_150pc = self.dir_rawm + self._read_key("velimage_150pc")
            self.bin_av_150pc = self.dir_other + self._read_key("bin_av_150pc")
            self.bin_co_150pc = self.dir_other + self._read_key("bin_co_150pc")
            self.mask_150pc = self.dir_rawc + self._read_key("mask_150pc")

            l = self.dir_rawm + self._read_key("mom0_150pc")
            self.mom0_150pc = glob.glob(l)
            self.mom0_150pc.sort()
            l = self.dir_rawm + self._read_key("emom0_150pc")
            self.emom0_150pc = glob.glob(l)
            self.emom0_150pc.sort()

            self.txt_hex_150pc = self.dir_ready + self._read_key("txt_hex_150pc")
            self.beam_150pc = 150/72.
            self.grid_150pc = int(70 * 0.8 / (150/72.))

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
    # run_cube_stacking #
    #####################

    def run_cube_stacking(self):

        taskname = self.modname + sys._getframe().f_code.co_name

        this_cubes = self.cubes_0p8as
        for this_cube in this_cubes:
            mytask.check_first(this_cube,taskname)

            # run cube_stacking
            this_stack_bins, this_stack_spec = cube_stacking(
                this_cube,
                self.vel_0p8as,
                self.bin_av_0p8as,
                self.mask_0p8as,
                self.ra_agn,
                self.dec_agn,
                )
            _, this_stack_spec_nomask = cube_stacking(
                this_cube,
                self.vel_0p8as,
                self.bin_av_0p8as,
                self.mask_0p8as,
                self.ra_agn,
                self.dec_agn,
                masking = False,
                )
            this_stack_spec = np.c_[
                this_stack_spec,
                np.delete(this_stack_spec_nomask,0,1),
                ]

            # write spectra to txt
            fmt = "%.6f"
            header = "Col 1 = v (km/s)\n" + \
                "Col 2-9 = masked spec (K.km/s)\n" + \
                "Col 10-17 = unmasked spec (K.km/s)"
            output = this_cube.split("/")[-1].replace(".fits","")
            output = output.replace("b3_12m+7m_","").replace("b3_12m_","")
            output = output.replace("b3_7m_","") + ".txt"
            output = self.dir_ready + "stack_" + output
            os.system("rm -rf " + output)
            np.savetxt(output,this_stack_spec,fmt=fmt,header=header)

        # write bins to txt
        output = self.dir_ready + "stack_bins_av.txt"
        os.system("rm -rf " + output)
        np.savetxt(output,this_stack_bins,fmt=fmt)

    ###################
    # run_hex_binning #
    ###################

    def run_hex_binning(
            self,
            resolution    = [],
            sampling      = False,
            do_all_plot   = False,
            plot_map      = False,
            plot_rmap     = False,
            plot_radial   = False,
            plot_rradial  = False,
            plot_scatter  = False,
            plot_rscatter = False,
            plot_corner   = False,
            plot_hist     = False,
            plot_rhist    = False,
            plot_pca      = False,
            plot_rpca     = False,
            ):
        """
        """

        self.resolution    = resolution
        self.sampling      = sampling
        self.do_all_plot   = do_all_plot
        self.plot_map      = plot_map
        self.plot_rmap     = plot_rmap
        self.plot_radial   = plot_radial
        self.plot_rradial  = plot_rradial
        self.plot_scatter  = plot_scatter
        self.plot_rscatter = plot_rscatter
        self.plot_corner   = plot_corner
        self.plot_hist     = plot_hist
        self.plot_rhist    = plot_rhist
        self.plot_pca      = plot_pca
        self.plot_rpca     = plot_rpca

        if do_all_plot==True:
            self.resolution    = ["150pc","130pc","90pc","0p8as"]
            self.plot_map      = True
            self.plot_rmap     = True
            self.plot_radial   = True
            self.plot_rradial  = True
            self.plot_scatter  = True
            self.plot_rscatter = True
            self.plot_corner   = True
            self.plot_rhist    = True
            self.plot_pca      = True
            self.plot_rpca     = True

        if "0p8as" in self.resolution:
            self._run_hex_binning_single_res(self.mom0_0p8as,self.emom0_0p8as,
                self.txt_hex_0p8as,"_0p8as",self.beam_0p8as,self.grid_0p8as)

        if "150pc" in self.resolution:
            self._run_hex_binning_single_res(self.mom0_150pc,self.emom0_150pc,
                self.txt_hex_150pc,"_150pc",self.beam_150pc,self.grid_150pc)

    def _run_hex_binning_single_res(
            self,
            list_mom,
            list_emom,
            txtdata,
            bstr,
            beam,
            gridsize,
            ):

        if self.sampling==True:
            self.run_hex_sampling(list_mom,list_emom,bstr,txtdata,beam,gridsize)

        if self.plot_map==True:
            self.run_hex_map(txtdata,bstr,beam,gridsize)

        if self.plot_rmap==True:
            self.run_hex_rmap(txtdata,bstr,beam,gridsize,denom="13co10")
            self.run_hex_rmap(txtdata,bstr,beam,gridsize,denom="hcn10")
            self.run_hex_rmap(txtdata,bstr,beam,gridsize,denom="cs21")

        if self.plot_radial==True:
            self.run_hex_radial(txtdata,bstr)

        if self.plot_rradial==True:
            self.run_hex_rradial(txtdata,bstr,denom="13co10")
            self.run_hex_rradial(txtdata,bstr,denom="hcn10")
            self.run_hex_rradial(txtdata,bstr,denom="cs21")

        if self.plot_scatter==True:
            self.run_hex_scatter(txtdata,bstr)

        if self.plot_rscatter==True:
            self.run_hex_rscatter(txtdata,bstr)

        if self.plot_corner==True:
            self.run_hex_coeff_corner(txtdata,bstr)

        if self.plot_hist==True:
            self.run_hex_hist(txtdata,bstr)

        if self.plot_rhist==True:
            self.run_hex_rhist(txtdata,bstr,"13co10")
            self.run_hex_rhist(txtdata,bstr,"hcn10")

        if self.plot_pca==True:
            self.run_hex_pca(txtdata,beam,gridsize,bstr,"")
            self.run_hex_pca(txtdata,beam,gridsize,bstr,"center")
            self.run_hex_pca(txtdata,beam,gridsize,bstr,"outer")

        if self.plot_rpca==True:
            self.run_hex_rpca(txtdata,beam,gridsize,bstr,"13co10","")
            self.run_hex_rpca(txtdata,beam,gridsize,bstr,"13co10","center")
            self.run_hex_rpca(txtdata,beam,gridsize,bstr,"13co10","outer")
            self.run_hex_rpca(txtdata,beam,gridsize,bstr,"hcn10","")
            self.run_hex_rpca(txtdata,beam,gridsize,bstr,"hcn10","center")
            self.run_hex_rpca(txtdata,beam,gridsize,bstr,"hcn10","outer")

    ################
    # run_hex_pca #
    ################

    def run_hex_pca(self,txtdata,beam,gridsize,bstr,mask=""):

        # extract line name
        with open(txtdata) as f:
            header = f.readline()

        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt(x**2 + y**2) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]

        # main
        data, data_name = [], []
        for i in range(len(list_name)):
            this_flux = array_data[:,i]
            this_err  = array_err[:,i]
            this_name = list_name[i]

            # sn cut and zero padding
            this_thres = abs(this_err * self.snr_mom)
            this_flux  = np.where(this_flux>=this_thres, this_flux, 0)
            this_flux  = ( this_flux - np.mean(this_flux) ) / np.std(this_flux)
            this_flux[np.isnan(this_flux)] = 0
            this_flux[np.isinf(this_flux)] = 0

            # masking
            if mask=="center":
                this_flux = np.where(r<=self.r_sbr, this_flux, 0)
                this_err  = np.where(r<=self.r_sbr, this_err, 0)
                if i==0: bstr = bstr + "_" + mask
            elif mask=="outer":
                this_flux = np.where(r>self.r_sbr, this_flux, 0)
                this_err  = np.where(r>self.r_sbr, this_err, 0)
                if i==0: bstr = bstr + "_" + mask

            # limiting by #data
            len_data = len(this_flux[this_flux>0])
            if len_data>=10:
                print("# meet " + this_name + " " + str(len_data))
                data.append(this_flux.flatten())
                data_name.append(this_name)
            else:
                print("# skip " + this_name + " " + str(len_data))

        data = np.array(data)

        # run
        print(np.shape(x))
        print(np.shape(y))
        print(np.shape(data))
        print(np.shape(data_name))
        os.system("rm -rf " + self.dir_hexp + "hexp_mom0*" + mask + ".png")
        output = self.dir_hexp + "hexp_mom0" + bstr + ".png"
        pca_2d_hex(x,y,data,data_name,output,bstr,self.snr_mom,beam,gridsize)

    ################
    # run_hex_rpca #
    ################

    def run_hex_rpca(self,txtdata,beam,gridsize,bstr,denom="13co10",mask=""):

        # extract line name
        with open(txtdata) as f:
            header = f.readline()

        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt(x**2 + y**2) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]

        # extract denom line
        index       = np.where(list_name==denom)[0][0]
        data_denom  = array_data[:,index]
        err_denom   = array_err[:,index]
        thres_denom = abs(err_denom * self.snr_mom)
        data_denom  = np.where(data_denom>=thres_denom, data_denom, 0)

        # main
        data, data_name = [], []
        for i in range(len(list_name)):
            this_flux = array_data[:,i]
            this_err  = array_err[:,i]
            this_name = list_name[i] + "_" + denom

            if list_name[i]==denom:
                continue

            # sn cut and zero padding
            this_thres = abs(this_err * self.snr_mom)
            this_flux  = np.where(this_flux>=this_thres, this_flux, 0)
            this_flux  = this_flux / data_denom
            this_flux[np.isnan(this_flux)] = 0
            this_flux[np.isinf(this_flux)] = 0
            this_flux  = ( this_flux - np.mean(this_flux) ) / np.std(this_flux)
            this_flux[np.isnan(this_flux)] = 0
            this_flux[np.isinf(this_flux)] = 0

            # masking
            if mask=="center":
                this_flux = np.where(r<=self.r_sbr, this_flux, 0)
                this_err  = np.where(r<=self.r_sbr, this_err, 0)
                if i==0: bstr = bstr + "_" + mask
            elif mask=="outer":
                this_flux = np.where(r>self.r_sbr, this_flux, 0)
                this_err  = np.where(r>self.r_sbr, this_err, 0)
                if i==0: bstr = bstr + "_" + mask

            # limiting by #data
            len_data = len(this_flux[this_flux>0])
            if len_data>=10:
                print("# meet " + this_name + " " + str(len_data))
                data.append(this_flux.flatten())
                data_name.append(this_name)
            else:
                print("# skip " + this_name + " " + str(len_data))

        data = np.array(data)

        # run
        print(np.shape(x))
        print(np.shape(y))
        print(np.shape(data))
        print(np.shape(data_name))
        os.system("rm -rf " + self.dir_hexp + "hexp_r_to_" + denom + "*"+mask+".png")
        output = self.dir_hexp + "hexp_r_to_" + denom + bstr + ".png"
        pca_2d_hex(x,y,data,data_name,output,bstr,self.snr_mom,beam,gridsize)

    #################
    # run_hex_rhist #
    #################

    def run_hex_rhist(self,txtdata,bstr,denom="13co10"):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt( x**2 + y**2 ) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]

        # extract denom line
        index       = np.where(list_name==denom)[0][0]
        data_denom  = array_data[:,index]
        err_denom   = array_err[:,index]
        thres_denom = abs(err_denom * self.snr_mom)

        # extract 13co10 line
        index        = np.where(list_name=="13co10")[0][0]
        data_13co10  = array_data[:,index]
        err_13co10   = array_err[:,index]
        thres_13co10 = abs(err_13co10 * self.snr_mom)

        # extract hcn10 line
        index       = np.where(list_name=="hcn10")[0][0]
        data_hcn10  = array_data[:,index]
        err_hcn10   = array_err[:,index]
        thres_hcn10 = abs(err_hcn10 * self.snr_mom)

        for i in range(len(list_name)):
            this_data  = array_data[:,i]
            this_err   = array_err[:,i]
            this_name  = list_name[i]
            this_thres = abs(this_err * self.snr_mom)
            print("# hist " + this_name)

            if this_name==denom:
                continue

            this_name = "r_" + this_name + "_" + denom
            out_data = self.dir_hexh_ratio+"hexh_"+this_name+bstr+".png"

            # sn cut
            cut = np.where((this_data>=this_thres) & \
                (data_13co10>=thres_13co10) & (data_hcn10>=thres_hcn10))
            this_r      = r[cut]
            this_data   = this_data[cut]
            this_13co10 = data_13co10[cut]
            this_hcn10  = data_hcn10[cut]
            this_denom  = data_denom[cut]

            # extract all
            x_all    = np.log10(this_data/this_denom)
            w13_all  = this_13co10
            whcn_all = this_hcn10
            b_all    = np.ceil( (np.log(len(x_all))+1)*3.0 )
            if len(x_all)!=0:
                r_all = [np.min(x_all), np.max(x_all)]
            else:
                r_all = None

            # extract cnd
            cut      = np.where(this_r<=self.r_cnd)
            x_cnd    = np.log10(this_data[cut]/this_denom[cut])
            w13_cnd  = this_13co10[cut]
            whcn_cnd = this_hcn10[cut]
            b_cnd    = np.ceil( (np.log(len(x_cnd))+1)*3.0 )
            if len(x_cnd)!=0:
                r_cnd = [np.min(x_cnd), np.max(x_cnd)]
            else:
                r_cnd = None

            # extract outflow
            cut      = np.where((this_r>self.r_cnd) & (this_r<=self.r_sbr))
            x_out    = np.log10(this_data[cut]/this_denom[cut])
            w13_out  = this_13co10[cut]
            whcn_out = this_hcn10[cut]
            b_out    = np.ceil( (np.log(len(x_out))+1)*3.0 )
            if len(x_out)!=0:
                r_out = [np.min(x_out), np.max(x_out)]
            else:
                r_out = None

            # extract sbr
            cut      = np.where(this_r>self.r_sbr)
            x_sbr    = np.log10(this_data[cut]/this_denom[cut])
            w13_sbr  = this_13co10[cut]
            whcn_sbr = this_hcn10[cut]
            b_sbr    = np.ceil( (np.log(len(x_sbr))+1)*3.0 )
            if len(x_sbr)!=0:
                r_sbr = [np.min(x_sbr), np.max(x_sbr)]
            else:
                r_sbr = None

            # get histogram and percentiles
            if len(x_all)!=0:
                h0_all,p0_all,hw13_all,pw13_all = get_hists(x_all, b_all, r_all, w13_all)
                _,_,hwhcn_all,pwhcn_all = get_hists(x_all, b_all, r_all, whcn_all)

                if len(x_cnd)!=0:
                    h0_cnd,p0_cnd,hw13_cnd,pw13_cnd = get_hists(x_cnd, b_all, r_all, w13_cnd)
                    _,_,hwhcn_cnd,pwhcn_cnd = get_hists(x_cnd, b_all, r_all, whcn_cnd)
                else:
                    h0_cnd,p0_cnd,hw13_cnd,pw13_cnd,hwhcn_cnd,pwhcn_cnd = \
                        None,None,None,None,None,None

                if len(x_out)!=0:
                    h0_out,p0_out,hw13_out,pw13_out = get_hists(x_out, b_all, r_all, w13_out)
                    _,_,hwhcn_out,pwhcn_out = get_hists(x_out, b_all, r_all, whcn_out)
                else:
                    h0_out,p0_out,hw13_out,pw13_out,hwhcn_out,pwhcn_out = \
                        None,None,None,None,None,None

                if len(x_sbr)!=0:
                    h0_sbr,p0_sbr,hw13_sbr,pw13_sbr = get_hists(x_sbr, b_all, r_all, w13_sbr)
                    _,_,hwhcn_sbr,pwhcn_sbr = get_hists(x_sbr, b_all, r_all, whcn_sbr)
                else:
                    h0_sbr,p0_sbr,hw13_sbr,pw13_sbr,hwhcn_sbr,pwhcn_sbr = \
                        None,None,None,None,None,None

                # plot
                self._myfig_hex_hist(
                    out_data,
                    h0_all,p0_all,hw13_all,pw13_all,hwhcn_all,pwhcn_all,
                    h0_cnd,p0_cnd,hw13_cnd,pw13_cnd,hwhcn_cnd,pwhcn_cnd,
                    h0_out,p0_out,hw13_out,pw13_out,hwhcn_out,pwhcn_out,
                    h0_sbr,p0_sbr,hw13_sbr,pw13_sbr,hwhcn_sbr,pwhcn_sbr,
                    this_name,xlabel="Ratio in log")

    ################
    # run_hex_hist #
    ################

    def run_hex_hist(self,txtdata,bstr):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt( x**2 + y**2 ) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]

        # extract 13co10 line
        index        = np.where(list_name=="13co10")[0][0]
        data_13co10  = array_data[:,index]
        err_13co10   = array_err[:,index]
        thres_13co10 = abs(err_13co10 * self.snr_mom)

        # extract hcn10 line
        index       = np.where(list_name=="hcn10")[0][0]
        data_hcn10  = array_data[:,index]
        err_hcn10   = array_err[:,index]
        thres_hcn10 = abs(err_hcn10 * self.snr_mom)

        for i in range(len(list_name)):
            this_data  = array_data[:,i]
            this_err   = array_err[:,i]
            this_name  = list_name[i]
            this_thres = abs(this_err * self.snr_mom)
            print("# hist " + this_name)

            out_data = self.dir_hexh_mom0+"hexh_mom0_"+this_name+bstr+".png"

            # sn cut
            cut = np.where((this_data>=this_thres) & \
                (data_13co10>=thres_13co10) & (data_hcn10>=thres_hcn10))
            this_r      = r[cut]
            this_data   = this_data[cut]
            this_13co10 = data_13co10[cut]
            this_hcn10  = data_hcn10[cut]

            # extract all
            x_all    = np.log10(this_data)
            w13_all  = this_13co10
            whcn_all = this_hcn10
            b_all    = np.ceil( (np.log(len(x_all))+1)*3.0 )
            if len(x_all)!=0:
                r_all = [np.min(x_all), np.max(x_all)]
            else:
                r_all = None

            # extract cnd
            cut      = np.where(this_r<=self.r_cnd)
            x_cnd    = np.log10(this_data[cut])
            w13_cnd  = this_13co10[cut]
            whcn_cnd = this_hcn10[cut]
            b_cnd    = np.ceil( (np.log(len(x_cnd))+1)*3.0 )
            if len(x_cnd)!=0:
                r_cnd = [np.min(x_cnd), np.max(x_cnd)]
            else:
                r_cnd = None

            # extract outflow
            cut      = np.where((this_r>self.r_cnd) & (this_r<=self.r_sbr))
            x_out    = np.log10(this_data[cut])
            w13_out  = this_13co10[cut]
            whcn_out = this_hcn10[cut]
            b_out    = np.ceil( (np.log(len(x_out))+1)*3.0 )
            if len(x_out)!=0:
                r_out = [np.min(x_out), np.max(x_out)]
            else:
                r_out = None

            # extract sbr
            cut      = np.where(this_r>self.r_sbr)
            x_sbr    = np.log10(this_data[cut])
            w13_sbr  = this_13co10[cut]
            whcn_sbr = this_hcn10[cut]
            b_sbr    = np.ceil( (np.log(len(x_sbr))+1)*3.0 )
            if len(x_sbr)!=0:
                r_sbr = [np.min(x_sbr), np.max(x_sbr)]
            else:
                r_sbr = None

            # get histogram and percentiles
            if len(x_all)!=0:
                h0_all,p0_all,hw13_all,pw13_all = get_hists(x_all, b_all, r_all, w13_all)
                _,_,hwhcn_all,pwhcn_all = get_hists(x_all, b_all, r_all, whcn_all)

                if len(x_cnd)!=0:
                    h0_cnd,p0_cnd,hw13_cnd,pw13_cnd = get_hists(x_cnd, b_all, r_all, w13_cnd)
                    _,_,hwhcn_cnd,pwhcn_cnd = get_hists(x_cnd, b_all, r_all, whcn_cnd)
                else:
                    h0_cnd,p0_cnd,hw13_cnd,pw13_cnd,hwhcn_cnd,pwhcn_cnd = \
                        None,None,None,None,None,None

                if len(x_out)!=0:
                    h0_out,p0_out,hw13_out,pw13_out = get_hists(x_out, b_all, r_all, w13_out)
                    _,_,hwhcn_out,pwhcn_out = get_hists(x_out, b_all, r_all, whcn_out)
                else:
                    h0_out,p0_out,hw13_out,pw13_out,hwhcn_out,pwhcn_out = \
                        None,None,None,None,None,None

                if len(x_sbr)!=0:
                    h0_sbr,p0_sbr,hw13_sbr,pw13_sbr = get_hists(x_sbr, b_all, r_all, w13_sbr)
                    _,_,hwhcn_sbr,pwhcn_sbr = get_hists(x_sbr, b_all, r_all, whcn_sbr)
                else:
                    h0_sbr,p0_sbr,hw13_sbr,pw13_sbr,hwhcn_sbr,pwhcn_sbr = \
                        None,None,None,None,None,None

                # plot
                self._myfig_hex_hist(
                    out_data,
                    h0_all,p0_all,hw13_all,pw13_all,hwhcn_all,pwhcn_all,
                    h0_cnd,p0_cnd,hw13_cnd,pw13_cnd,hwhcn_cnd,pwhcn_cnd,
                    h0_out,p0_out,hw13_out,pw13_out,hwhcn_out,pwhcn_out,
                    h0_sbr,p0_sbr,hw13_sbr,pw13_sbr,hwhcn_sbr,pwhcn_sbr,
                    this_name)

    ###################
    # _myfig_hex_hist #
    ###################

    def _myfig_hex_hist(
            self,
            output,
            h0_all,p0_all,hw13_all,pw13_all,hwhcn_all,pwhcn_all,
            h0_cnd,p0_cnd,hw13_cnd,pw13_cnd,hwhcn_cnd,pwhcn_cnd,
            h0_out,p0_out,hw13_out,pw13_out,hwhcn_out,pwhcn_out,
            h0_sbr,p0_sbr,hw13_sbr,pw13_sbr,hwhcn_sbr,pwhcn_sbr,
            title,
            xlabel="K km s$^{-1}$ in log",
            ):

        # prepare
        if h0_all!=None:
            ylim1 = [0, 1.5 * np.max(h0_all[:,1]/np.sum(h0_all[:,1]))]
        else:
            ylim1 = None

        if hw13_all!=None:
            ylim2 = [0, 1.5 * np.max(hw13_all[:,1]/np.sum(hw13_all[:,1]))]
        else:
            ylim2 = None

        if hwhcn_all!=None:
            ylim3 = [0, 1.5 * np.max(hwhcn_all[:,1]/np.sum(hwhcn_all[:,1]))]
        else:
            ylim3 = None

        # plot
        fig = plt.figure(figsize=(10,10))
        gs  = gridspec.GridSpec(nrows=30, ncols=30)
        ax1 = plt.subplot(gs[0:10,0:28])
        ax2 = plt.subplot(gs[10:20,0:28])
        ax3 = plt.subplot(gs[20:30,0:28])
        
        myax_set(ax1,ylim=ylim1,grid="x",title=title+" histograms",labelbottom=False)
        myax_set(ax2,ylim=ylim2,grid="x",labelbottom=False)
        myax_set(ax3,ylim=ylim3,grid="x",xlabel=xlabel)

        # plot histograms
        myax_hists(ax1, h0_all, p0_all, h0_all, 1.4, "grey")
        myax_hists(ax1, h0_cnd, p0_cnd, h0_all, 1.3, "red")
        myax_hists(ax1, h0_out, p0_out, h0_all, 1.2, "green")
        myax_hists(ax1, h0_sbr, p0_sbr, h0_all, 1.1, "blue")

        myax_hists(ax2, hw13_all, pw13_all, hw13_all, 1.4, "grey")
        myax_hists(ax2, hw13_cnd, pw13_cnd, hw13_all, 1.3, "red")
        myax_hists(ax2, hw13_out, pw13_out, hw13_all, 1.2, "green")
        myax_hists(ax2, hw13_sbr, pw13_sbr, hw13_all, 1.1, "blue")

        myax_hists(ax3, hwhcn_all, pwhcn_all, hwhcn_all, 1.4, "grey")
        myax_hists(ax3, hwhcn_cnd, pwhcn_cnd, hwhcn_all, 1.3, "red")
        myax_hists(ax3, hwhcn_out, pwhcn_out, hwhcn_all, 1.2, "green")
        myax_hists(ax3, hwhcn_sbr, pwhcn_sbr, hwhcn_all, 1.1, "blue")

        # text
        ax1.text(0.02,0.85,"area-weighted",color="black",transform=ax1.transAxes,
            horizontalalignment="left")
        ax2.text(0.02,0.85,"13co-weighted",color="black",transform=ax2.transAxes,
            horizontalalignment="left")
        ax3.text(0.02,0.85,"hcn-weighted",color="black",transform=ax3.transAxes,
            horizontalalignment="left")

        ax1.text(0.02,0.75,"Whole",color="grey",transform=ax1.transAxes,
            horizontalalignment="left")
        ax1.text(0.02,0.65,"CND",color="red",transform=ax1.transAxes,
            horizontalalignment="left")
        ax1.text(0.02,0.55,"CND<r<SBR",color="green",transform=ax1.transAxes,
            horizontalalignment="left")
        ax1.text(0.02,0.45,"SBR",color="blue",transform=ax1.transAxes,
            horizontalalignment="left")

        print("# output = " + output)
        fig.savefig(output, dpi=fig_dpi)

    ########################
    # run_hex_coeff_corner #
    ########################

    def run_hex_coeff_corner(self,txtdata,bstr):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt( x**2 + y**2 ) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]

        # output name
        out_data = self.dir_hexs_mom0 + "hexs_corner" + bstr + ".png"

        # plot
        myfig_hex_rcorner(
            r,
            array_data,
            array_err,
            list_name,
            out_data,
            snr=3.0,
            numlimit=10,
            )

    ####################
    # run_hex_rscatter #
    ####################

    def run_hex_rscatter(self,txtdata,bstr):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt( x**2 + y**2 ) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]

        # plot
        for i in itertools.combinations(range(len(list_name)), 2):
            this_xdata  = array_data[:,i[0]]
            this_xerr   = array_err[:,i[0]]
            this_xname  = list_name[i[0]]

            this_ydata  = array_data[:,i[1]]
            this_yerr   = array_err[:,i[1]]
            this_yname  = list_name[i[1]]

            this_rdata  = this_yerr / this_xerr
            this_rerr   = this_rdata * np.sqrt( (this_xerr/this_xdata)**2 + (this_yerr/this_ydata)**2 )

            out_data = self.dir_hexs_ratio + "hexs_r_" + this_xname + "_vs_"
            out_data = out_data + this_yname + bstr + ".png"

            # plot hex map
            this_xlen = this_xdata.flatten()
            this_ylen = this_ydata.flatten()

            this_xlen = this_xlen[~np.isnan(this_xlen)]
            this_ylen = this_ylen[~np.isnan(this_xlen)]
            this_xlen = this_xlen[~np.isnan(this_ylen)]
            this_ylen = this_ylen[~np.isnan(this_ylen)]
            this_xlen = this_xlen[~np.isinf(this_xlen)]
            this_ylen = this_ylen[~np.isinf(this_xlen)]
            this_xlen = this_xlen[~np.isinf(this_ylen)]
            this_ylen = this_ylen[~np.isinf(this_ylen)]

            this_len = len(this_xlen) * len(this_ylen)

            if this_len>0:
                myfig_hex_scatter(
                    r,
                    this_xdata,
                    this_xerr,
                    this_rdata,
                    this_rerr,
                    out_data,
                    this_xname,
                    this_yname + "/" + this_xname,
                    self.snr_mom,
                    plot_lines=False)

            else:
                print("# skip " + out_data.split("/")[-1])

    ###################
    # run_hex_scatter #
    ###################

    def run_hex_scatter(self,txtdata,bstr):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt( x**2 + y**2 ) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]
        
        # plot
        for i in itertools.combinations(range(len(list_name)), 2):
            this_xdata  = array_data[:,i[0]]
            this_xerr   = array_err[:,i[0]]
            this_xname  = list_name[i[0]]

            this_ydata  = array_data[:,i[1]]
            this_yerr   = array_err[:,i[1]]
            this_yname  = list_name[i[1]]

            out_data = self.dir_hexs_mom0 + "hexs_mom0_" + this_xname + "_vs_"
            out_data = out_data + this_yname + bstr + ".png"

            # plot hex map
            this_xlen = this_xdata.flatten()
            this_xlen = this_xlen[~np.isnan(this_xlen)]
            this_ylen = this_ydata.flatten()
            this_ylen = this_ylen[~np.isnan(this_ylen)]
            this_len = len(this_xlen) * len(this_ylen)

            if this_len>0:
                myfig_hex_scatter(
                    r,
                    this_xdata,
                    this_xerr,
                    this_ydata,
                    this_yerr,
                    out_data,
                    this_xname,
                    this_yname,
                    self.snr_mom)

            else:
                print("# skip " + out_data.split("/")[-1])

    ###################
    # run_hex_rradial #
    ###################

    def run_hex_rradial(self,txtdata,bstr,denom="13co10"):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt( x**2 + y**2 ) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]
        
        # extract denom line
        index       = np.where(list_name==denom)[0][0]
        data_13co10 = array_data[:,index]
        err_13co10  = array_err[:,index]
        
        # take ratio to denom line
        for i in range(len(list_name)):
            this_data  = array_data[:,i]
            this_err   = array_err[:,i]
            this_name  = list_name[i]
            
            if this_name==denom:
                continue

            this_ratio = this_data / data_13co10
            err1       = this_err / this_data
            err2       = err_13co10 / data_13co10
            this_err   = this_ratio * np.sqrt( err1**2 + err2**2 )
            this_name  = list_name[i] + "_" + denom
            this_thres = abs(this_err * self.snr_mom)

            out_data = self.dir_hexr_ratio+"hexr_r_"+this_name+bstr+".png"

            # plot hex radial
            this_len = this_ratio.flatten()
            this_len = len( this_len[~np.isnan(this_len)] )

            if this_len>1:
                myfig_hex_radial(r,this_ratio,this_err,out_data,snr=self.snr_mom,
                    ylabel="Ratio")

    ##################
    # run_hex_radial #
    ##################

    def run_hex_radial(self,txtdata,bstr):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        r           = np.sqrt( x**2 + y**2 ) * self.scale_pc / 1000.
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]

        for i in range(len(array_data[0,:])):
            this_data  = array_data[:,i]
            this_err   = array_err[:,i]
            this_name  = list_name[i]
            this_thres = abs(this_err * self.snr_mom)
            
            out_data = self.dir_hexr_mom0+"hexr_mom0_"+this_name+bstr+".png"

            # plot hex radial dist
            this_len = this_data.flatten()
            this_len = len( this_len[~np.isnan(this_len)] )

            if this_len>1:
                myfig_hex_radial(r,this_data,this_err,out_data,snr=self.snr_mom)

            else:
                print("# skip " + out_data.split("/")[-1])

    ###############
    # run_hex_map #
    ###############

    def run_hex_map(self,txtdata,bstr,beam,grid):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]

        for i in range(len(array_data[0,:])):
            this_data  = array_data[:,i]
            this_err   = array_err[:,i]
            this_name  = list_name[i]
            this_thres = abs(this_err * self.snr_mom)
            
            this_x    = np.where(this_data>=this_thres,-x,np.nan)
            this_y    = np.where(this_data>=this_thres,y,np.nan)
            this_err  = np.where(this_data>=this_thres,this_err,np.nan)
            this_data = np.where(this_data>=this_thres,this_data,np.nan)
            this_snr  = this_data / this_err

            out_data = self.dir_hexmap_mom0+"hexmap_mom0_"+this_name+bstr+".png"
            out_err = self.dir_hexmap_mom0+"hexmap_emom0_"+this_name+bstr+".png"
            out_snr = self.dir_hexmap_mom0+"hexmap_snrmom0_"+this_name+bstr+".png"

            # plot hex map
            this_len = this_data.flatten()
            this_len = len( this_len[~np.isnan(this_len)] )

            if this_len>0:
                myfig_hex_map(this_x,this_y,this_data,out_data,beam,grid)
                myfig_hex_map(this_x,this_y,this_err,out_err,beam,grid)
                myfig_hex_map(this_x,this_y,this_snr,out_snr,beam,grid,cblabel="")

            else:
                print("# skip " + out_data.split("/")[-1])

    ################
    # run_hex_rmap #
    ################

    def run_hex_rmap(self,txtdata,bstr,beam,grid,denom="13co10"):

        if not glob.glob(txtdata):
            print("# no input txtfile!")
            return None

        taskname = self.modname + sys._getframe().f_code.co_name

        # extract line name
        with open(txtdata) as f:
            header = f.readline()
        list_name = [s for s in header.split(" ")[3:-1] if "err" not in s]
        list_name = np.array(list_name)

        # extract data
        data = np.loadtxt(txtdata)
        x           = data[:,0]
        y           = data[:,1]
        array_data  = data[:,2::2]
        array_err   = data[:,3::2]
        
        # extract denom line
        index       = np.where(list_name==denom)[0][0]
        data_13co10 = array_data[:,index]
        err_13co10  = array_err[:,index]
        
        # take ratio to denom line
        for i in range(len(list_name)):
            this_data  = array_data[:,i]
            this_err   = array_err[:,i]
            this_name  = list_name[i]
            
            if this_name==denom:
                continue

            this_ratio = this_data / data_13co10
            err1       = this_err / this_data
            err2       = err_13co10 / data_13co10
            this_err   = this_ratio * np.sqrt( err1**2 + err2**2 )
            this_name  = list_name[i] + "_" + denom
            this_thres = abs(this_err * self.snr_mom)

            this_x     = np.where(this_ratio>=this_thres,-x,np.nan)
            this_y     = np.where(this_ratio>=this_thres,y,np.nan)
            this_err   = np.where(this_ratio>=this_thres,this_err,np.nan)
            this_ratio = np.where(this_ratio>=this_thres,this_ratio,np.nan)
            this_snr   = this_ratio / this_err

            this_ratio[np.isinf(this_ratio)] = np.nan

            out_data = self.dir_hexmap_ratio+"hexmap_r_"+this_name+bstr+".png"
            out_err = self.dir_hexmap_ratio+"hexmap_er_"+this_name+bstr+".png"
            out_snr = self.dir_hexmap_ratio+"hexmap_snrr_"+this_name+bstr+".png"

            # plot hex map
            this_len = this_ratio.flatten()
            this_len = len( this_len[~np.isnan(this_len)] )

            if this_len>0:
                myfig_hex_map(this_x,this_y,this_ratio,out_data,beam,grid,cblabel="")
                myfig_hex_map(this_x,this_y,this_err,out_err,beam,grid,cblabel="")
                myfig_hex_map(this_x,this_y,this_snr,out_snr,beam,grid,cblabel="")

            else:
                print("# skip " + out_data.split("/")[-1])

    ####################
    # run_hex_sampling #
    ####################

    def run_hex_sampling(self,list_mom0,list_emom0,bstr,output,beam,grid):

        taskname = self.modname + sys._getframe().f_code.co_name

        fmt = "%.5f"
        hexdata = []
        header = "x(as) y(as) "
        
        for i in range(len(list_mom0)):
            this_mom0  = list_mom0[i]
            this_emom0 = list_emom0[i]
            this_name  = this_mom0.split("ngc1068_b3_12m_")[-1]
            this_name  = this_name.split(bstr+"_broad_mom0.fits")[0]

            mytask.check_first(this_mom0,taskname)

            # run hexbin_sampling mom0
            hexx,hexy,hex_mom0 = hexbin_sampling(
                this_mom0,
                self.ra_agn,
                self.dec_agn,
                beam,
                grid,
                )

            # run hexbin_sampling emom0
            _,_,hex_emom0 = hexbin_sampling(
                this_emom0,
                self.ra_agn,
                self.dec_agn,
                beam,
                grid,
                )

            if i==0:
                hexdata = hexx
                hexdata = np.c_[hexdata,hexy]

            hexdata = np.c_[hexdata,hex_mom0]
            hexdata = np.c_[hexdata,hex_emom0]
            header  = header + this_name + " err "
           
        np.savetxt(output,hexdata,fmt=fmt,header=header)

