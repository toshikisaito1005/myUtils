import os
import sys
import glob

"""
hard-code...
- directory names
- input FITS file names
- output FITS file names
"""

#################################
# define and create directories #
#################################
# directories
dir_data_co10 = "./data_raw/data_co10/"    # where all co10 maps are already available
dir_data_ci10 = "./data_raw/data_ci10/"    # where all ci10 maps are already available
dir_data_cont = "./data_raw/data_cont_b8/" # where all band 8 continuum maps are already available
dir_analysis  = "./analysis/"              # where all output FITS files will go
dir_figures   = "./figures/"               # where all output dir_figuresd will go

# create some output directories if not present
done = glob.glob(dir_analysis)
if not done:
    os.mkdir(dir_analysis)

done = glob.glob(dir_figures)
if not done:
    os.mkdir(dir_figures)

###########################
# define input FITS files #
###########################
# co10 maps
input_cube_co10  = dir_data_co10 + "ngc1068_b3_12m+7m_co10_0p8as.fits"
input_mom0_co10  = dir_data_co10 + "ngc1068_b3_12m+7m_co10_0p8as_strict_mom0.fits"
input_mom1_co10  = dir_data_co10 + "ngc1068_b3_12m+7m_co10_0p8as_strict_mom1.fits"
input_mom2_co10  = dir_data_co10 + "ngc1068_b3_12m+7m_co10_0p8as_strict_mom2.fits"

# co10 noise maps
input_ecube_co10 = dir_data_co10 + "ngc1068_b3_12m+7m_co10_0p8as_noise.fits"
input_emom0_co10 = dir_data_co10 + "ngc1068_b3_12m+7m_co10_0p8as_strict_emom0.fits"
input_emom1_co10 = dir_data_co10 + "ngc1068_b3_12m+7m_co10_0p8as_strict_emom1.fits"
input_emom2_co10 = dir_data_co10 + "ngc1068_b3_12m+7m_co10_0p8as_strict_emom2.fits"

# ci10 maps
input_cube_ci10  = dir_data_ci10 + "ngc1068_b8_12m+7m_ci10.fits"
input_mom0_ci10  = dir_data_ci10 + "ngc1068_b8_12m+7m_ci10_strict_mom0.fits"
input_mom1_ci10  = dir_data_ci10 + "ngc1068_b8_12m+7m_ci10_strict_mom1.fits"
input_mom2_ci10  = dir_data_ci10 + "ngc1068_b8_12m+7m_ci10_strict_mom2.fits"

# ci10 noise maps
input_ecube_ci10 = dir_data_ci10 + "ngc1068_b8_12m+7m_ci10_noise.fits"
input_emom0_ci10 = dir_data_ci10 + "ngc1068_b8_12m+7m_ci10_strict_emom0.fits"
input_emom1_ci10 = dir_data_ci10 + "ngc1068_b8_12m+7m_ci10_strict_emom1.fits"
input_emom2_ci10 = dir_data_ci10 + "ngc1068_b8_12m+7m_ci10_strict_emom2.fits"

# cont maps
input_map_cont_fov1 = dir_data_cont + "ngc1068_b8_1_12m+7m_cont.fits"
input_map_cont_fov2 = dir_data_cont + "ngc1068_b8_2_12m+7m_cont.fits"
input_map_cont_fov3 = dir_data_cont + "ngc1068_b8_3_12m+7m_cont.fits"

############################
# define output FITS files #
############################
# co10 maps
cube_co10  = dir_analysis + "n1068_co10_cube.fits"
mom0_co10  = dir_analysis + "n1068_co10_mom0.fits"
mom1_co10  = dir_analysis + "n1068_co10_mom1.fits"
mom2_co10  = dir_analysis + "n1068_co10_mom2.fits"

# co10 noise maps
ecube_co10 = dir_analysis + "n1068_co10_ecube.fits"
emom0_co10 = dir_analysis + "n1068_co10_emom0.fits"
emom1_co10 = dir_analysis + "n1068_co10_emom1.fits"
emom2_co10 = dir_analysis + "n1068_co10_emom2.fits"

# ci10 maps
cube_ci10  = dir_analysis + "n1068_ci10_cube.fits"
mom0_ci10  = dir_analysis + "n1068_ci10_mom0.fits"
mom1_ci10  = dir_analysis + "n1068_ci10_mom1.fits"
mom2_ci10  = dir_analysis + "n1068_ci10_mom2.fits"

# ci10 noise maps
ecube_ci10 = dir_analysis + "n1068_ci10_ecube.fits"
emom0_ci10 = dir_analysis + "n1068_ci10_emom0.fits"
emom1_ci10 = dir_analysis + "n1068_ci10_emom1.fits"
emom2_ci10 = dir_analysis + "n1068_ci10_emom2.fits"

# cont maps
map_cont_fov1 = dir_analysis + "n1068_1_cont.fits"
map_cont_fov2 = dir_analysis + "n1068_2_cont.fits"
map_cont_fov3 = dir_analysis + "n1068_3_cont.fits"

#######
# end #
#######