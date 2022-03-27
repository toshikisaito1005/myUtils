import os
from scripts_lst_wp import ToolsLSTSim as tools

# key
tl = tools(
    refresh     = False,
    keyfile_gal = "/home02/saitots/myUtils/keys_projects/galkey_ngc1068.txt",
    keyfile_fig = "/home02/saitots/myUtils/keys_projects/key_lst_wp.txt",
    )

freq = 492.16065100

# main
tl.run_sim_lst_alma(
    ##############
    # ngc1097sim #
    ##############
    # prepare
    tinteg_n1097sim        = 11.4,  # hours
    observed_freq          = freq,  # GHz, for LST, ACA beam sizes
    do_template_n1097sim   = False,
    # ACA-alone
    do_simACA_n1097sim     = False, # take time!
    do_imaging_n1097sim    = False,
    # SD-alone
    dryrun_simSD           = False, # just print parameters
    do_simTP_n1097sim      = False,
    do_simLST_n1097sim     = False,
    do_feather             = False,
    # postprocess
    do_mom0_n1097sim       = False,
    #
    ############
    # torussim #
    ############
    tinteg_torussim        = 12,    # hours
    do_template_torussim   = False,
    do_simint_torussim     = False,
    do_imaging_torussim    = False, # take time!
    do_process_torussim    = True,
    #
    ########
    # plot #
    ########
    plot_config            = False,
    plot_mosaic            = False,
    plot_mom0_n1097sim     = False,
    plot_scatter_n1097sim  = False,
    plot_mom0_torussim     = True,
    # calc
    calc_collectingarea    = False,
    #
    ############
    # checksim #
    ############
    tinteg_checksim        = 1,
    do_template_checksim   = False,
    do_simint_checksim     = False,
    do_imaging_checksim    = False,
    )

# cleanup
os.system("rm -rf *.last")
