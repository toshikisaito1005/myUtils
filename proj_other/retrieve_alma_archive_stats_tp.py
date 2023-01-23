import os
import sys
import csv
import glob
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

file_csv = 'observations_1674016987845.csv'
#  0 'Project code'
#  1 'ALMA source name'
#  4 'Band'
#  5 'Cont. sens.'
#  6 'Frequency support'
#  8 'Publications'
# 10 'Min. vel. res.'
# 15 'Scientific category'
# 17 'Int. Time'
# 20 'Min. freq. res.'
# 24 'Obs. date'
# 26 'SB name'
# 28 'Line sens. (10 km/s)'
# 29 'PWV'
# 30 'Group ous id'
# 31 'Member ous id'
# 32 'Asdm'
# 36 'QA2 Status'

#
def plot_hist(
    data,
    title,
    xlabel,
    output,
    ):
    pmin  = 'min  = ' + str(np.min(data))
    p16   = '16th = ' + str(int(np.percentile(data,16)))
    p50   = '50th = ' + str(int(np.percentile(data,50)))
    p84   = '84th = ' + str(int(np.percentile(data,84)))
    pmax  = 'max  = ' + str(np.max(data))
    pmod  = 'mode = ' + str(st.mode(data))
    pmean = 'mean = ' + str(np.round(np.mean(data)))
    pstd  = 'std  = ' + str(np.round(np.std(data)))
    pwid  = '16th-84th width = ' + str(np.percentile(data,84)-np.percentile(data,16))

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(data, bins=50)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.text(0.65,0.95,pmin,transform=ax.transAxes)
    ax.text(0.65,0.90,p16,transform=ax.transAxes)
    ax.text(0.65,0.85,p50,transform=ax.transAxes)
    ax.text(0.65,0.80,p84,transform=ax.transAxes)
    ax.text(0.65,0.75,pmax,transform=ax.transAxes)
    ax.text(0.65,0.70,pmod,transform=ax.transAxes)
    ax.text(0.65,0.65,pmean,transform=ax.transAxes)
    ax.text(0.65,0.60,pstd,transform=ax.transAxes)
    ax.text(0.65,0.55,pwid,transform=ax.transAxes)
    fig.savefig(output)

# read csv
with open(file_csv, 'r') as x:
    sample_data = list(csv.reader(x, delimiter=','))
 
sample_data = np.array(sample_data)

# cut OFF position rows
selected_data = []
for i in range(len(sample_data)):
    this_data = sample_data[i]
    if "_OFF_0" not in this_data[1]:
        selected_data.append(this_data.tolist())

selected_data = np.array(selected_data[1:])

#####################################
# select unique projects and MOUSes #
#####################################
# projects
projs = np.unique(selected_data[:,0])
moues = np.unique(selected_data[:,31])

# MOUSes
indice = []
selected_data2 = selected_data[np.argsort(selected_data[:, 31])]

for i in range(len(selected_data2)):
    if i==0:
        this_mous = selected_data2[i,31]
        indice.append(i)
    else:
        if selected_data2[i,31]!=this_mous:
            this_mous = selected_data2[i,31]
            indice.append(i)

selected_mous = selected_data2[indice]

#########################
# number of TP projects #
#########################
print("# total number of project =", len(projs))
print("# total number of MOUS =", len(selected_mous))

"""
##########################
# number of EBs per MOUS #
##########################
xdata  = selected_mous[:,32].astype(np.int)
title  = '#EBs per TP MOUS (#Cy8 TP MOUS = ' + str(len(selected_mous)) + ')'
xlabel = '#EBs'
output = 'fig_hist_n_eb_per_mous.png'
plot_hist(xdata,title,xlabel,output)


###########################
# number of spws per MOUS #
###########################
this_freq_list = selected_mous[:,6]
xdata = []
for i in range(len(this_freq_list)):
    nspw = len(this_freq_list[i].split(' U '))
    xdata.append(nspw)

title  = '#spws per TP MOUS (#Cy8 TP MOUS = ' + str(len(selected_mous)) + ')'
xlabel = '#spws'
output = 'fig_hist_n_spw_per_mous.png'
plot_hist(xdata,title,xlabel,output)

############################
# number of chans per MOUS #
############################
this_freq_list  = selected_mous[:,6]
xdata = []
for i in range(len(this_freq_list)):
    list_spw       = this_freq_list[i].split(' U ')
    this_spw_nchan = 0
    for j in range(len(list_spw)):
        this_spw       = list_spw[j].replace('[','').replace(']','').split(',')
        this_freq      = this_spw[0].replace('GHz','').split('..')
        this_bandwidth = float(this_freq[1]) - float(this_freq[0])
        this_chanwidth = float(this_spw[1].replace('kHz',''))
        this_nchan     = int( this_bandwidth*1e6 / float(this_chanwidth))
        this_spw_nchan = this_spw_nchan + this_nchan

    if this_spw_nchan==0:
        this_spw_nchan=1 # solar obs?

    xdata.append(this_spw_nchan)

title  = '#channels per TP MOUS (#Cy8 TP MOUS = ' + str(len(selected_mous)) + ')'
xlabel = '#channels'
output = 'fig_hist_n_chan_per_mous.png'
plot_hist(xdata,title,xlabel,output)

###########################
# number of chans per spw #
###########################
this_freq_list  = selected_mous[:,6]
xdata = []
for i in range(len(this_freq_list)):
    list_spw       = this_freq_list[i].split(' U ')
    for j in range(len(list_spw)):
        this_spw       = list_spw[j].replace('[','').replace(']','').split(',')
        this_freq      = this_spw[0].replace('GHz','').split('..')
        this_bandwidth = float(this_freq[1]) - float(this_freq[0])
        this_chanwidth = float(this_spw[1].replace('kHz',''))
        this_nchan     = int( this_bandwidth*1e6 / this_chanwidth)

        if this_nchan==0:
            this_nchan=1 # solar obs?

        xdata.append(this_nchan)

title  = '#channels per TP spw (#Cy8 TP spw = ' + str(len(xdata)) + ')'
xlabel = '#channels'
output = 'fig_hist_n_chan_per_spw.png'
plot_hist(xdata,title,xlabel,output)
"""

##################################
# number of source per MOUS hist #
##################################
xdata = []
for this_mous in moues:
    this_data = selected_data[selected_data[:,31]==this_mous]
    xdata.append(len(np.unique(this_data[:,1])))

title  = '#sources per TP MOUS (#Cy8 TP MOUS = ' + str(len(selected_mous)) + ')'
xlabel = '#sources'
output = 'fig_hist_n_source_per_mous.png'
plot_hist(xdata,title,xlabel,output)

#######
# end #
#######
