import os
import sys
import csv
import glob
import datetime
import numpy as np
import statistics as st
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

d_today = datetime.date.today()
file_csv = '/Users/saitonu/Desktop/observations_1677244219917.csv'
dir_plot = '/Users/saitonu/Desktop/n1068_archive_for_cycle10/'

#  0 'Project code'
#  1 'ALMA source name'
#  2 'RA'
#  3 'Dec'
#  4 'Band'
#  5 'Cont. sens.'
#  6 'Frequency support'
#  8 'Publications'
#  9 'Ang. res.'
# 10 'Min. vel. res.'
# 11 'Array'
# 12 'mosaic'
# 13 'MRS'
# 14 'FoV'
# 15 'Scientific category'
# 17 'Int. Time'
# 20 'Min. freq. res.'
# 24 'Obs. date'
# 25 'PI name'
# 26 'SB name'
# 28 'Line sens. (10 km/s)'
# 29 'PWV'
# 30 'Group ous id'
# 31 'Member ous id'
# 32 'Asdm'
# 36 'QA2 Status'

# read csv
with open(file_csv, 'r') as x:
    sample_data = list(csv.reader(x, delimiter=','))
 
sample_data = np.array(sample_data)

# cut OFF position rows
selected_data = []
for i in range(len(sample_data)):
    this_data = sample_data[i]
    if '_OFF_0' not in this_data[1]:
        if 'TP' not in this_data[11]:
            selected_data.append(this_data.tolist())

data = np.array(selected_data[1:])

############
# function #
############
def plot_band(
    selected_data,
    band='3',
    color='tomato',
    xlim=[84.0,116.0],
    outpng='/Users/saitonu/Desktop/n1068_archive_b3.png',
    ):
    """
    """
    list_spatial_res = np.array([1,2,5,10,20,40,80,160,320,640])

    # data
    this_band        = selected_data[selected_data[:,4]==band]
    this_freq_list   = this_band[:,6]
    x_freq           = []
    for i in range(len(this_freq_list)):
        list_spw = this_freq_list[i].split(' U ')
        this_x_freq = []
        for j in range(len(list_spw)):
            this_spw  = list_spw[j].replace('[','').replace(']','').split(',')
            this_freq = this_spw[0].replace('GHz','').split('..')
            this_x_freq.append([float(this_freq[0]), float(this_freq[1])])
        x_freq.append(this_x_freq)

    y_res = this_band[:,9].astype(np.float32)
    y_mrs = this_band[:,13].astype(np.float32)

    # plot
    xlim   = [xlim[0]-2,xlim[1]+2]
    ylim   = [np.log10(np.min(y_res))-0.1, np.log10(np.max(y_res))+0.1]
    height = (ylim[1] - ylim[0]) / 100
    right  = xlim[1] + (xlim[1]-xlim[0])*0.01

    fig = plt.figure(figsize=(13,10))
    plt.rcParams['font.size'] = 18
    gs  = gridspec.GridSpec(nrows=10, ncols=10)
    ax  = plt.subplot(gs[1:10,0:10])

    for i in range(len(x_freq)):
        list_x_freq = x_freq[i]
        this_y_res  = np.log10(y_res[i])
        this_y_mrs  = np.log10(y_mrs[i])
        for this_x_freq in list_x_freq:
            r = patches.Rectangle(
                xy     = (this_x_freq[0],this_y_res),
                width  = this_x_freq[1]-this_x_freq[0],
                height = height,
                ec     = None,
                fc     = color,
                fill   = True,
                )
            ax.add_patch(r)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for this_y in list_spatial_res:
        this_y2 = np.log10(this_y / 72.)
        ax.plot(xlim,[this_y2,this_y2],ls='dotted',color='grey',lw=1,alpha=0.5)
        ax.text(right,this_y2,str(this_y)+' pc',va='center',ha='left')

    ax.set_title('Band ' + band + ' Archive as of ' + str(d_today))
    ax.set_xlabel('Observed Freq (GHz)')
    ax.set_ylabel('log Angular Resolution (arcsec) [72 pc/arcsec]')

    plot_lines(ax,ylim)

    plt.savefig(outpng, dpi=200)

def plot_lines(ax,ylim,z=0.00379):
    """
    """

    lines = [
    88.63160230,  # hcn10
    86.84696000,  # sio21
    89.18852470,  # hcop10
    90.66356800,  # hnc10
    97.98095330,  # cs21
    109.78217340, # c18o10
    110.20135430, # 13co10
    115.27120,    # co10
    130.26861000, # sio32
    146.96902870, # cs32
    173.68831000, # sio43
    177.26111150, # hcn21
    178.37505630, # hcop21
    181.32475800, # hnc21
    183.31008700, # h2o32
    195.95421090, # cs43
    # band 6
    217.10498000, # sio54
    219.56035410, # c18o21
    220.39868420, # 13co21
    224.71418700, # c17o21
    230.53800000, # co21
    244.93555650, # cs54
    260.51802000, # sio65
    265.88643430, # hcn32
    267.55762590, # hcop32
    271.98114200, # hnc32
    # band 7
    293.91208650, # cs65
    303.92696000, # sio76
    325.15289900, # h2o54
    329.33055250, # c18o32
    330.58796530, # 13co32
    342.88285030, # cs76
    345.79598990, # co32
    347.33063100, # sio87
    354.50547790, # hcn43
    356.73422300, # hcop43
    362.63030300, # hcn43
    # band 8
    390.72844830, # sio98
    391.84688980, # cs87
    434.11955210, # sio109
    439.08876580, # c18o43
    440.76517350, # 13co43
    440.80323200, # cs98
    443.11614930, # hcn54
    445.90287210, # hcop54
    449.39519100, # c17o43
    453.26992200, # hnc54
    461.04076820, # co43
    477.50309650, # sio1110
    489.75092100, # cs109
    492.16065100, # ci10
    # band 9
    674.00920240, # c17o65
    691.47307630, # co65
    708.87700510, # hcn87
    713.34122780, # hcop87
    ]
    names = [
    # band 3
    'hcn10',
    'sio21',
    'hcop10',
    'hnc10',
    'cs21',
    'c18o10',
    '13co10',
    'co10',
    # band 4
    'sio32',
    'cs32',
    # band 5
    'sio43',
    'hcn21',
    'hcop21',
    'hnc21',
    'h2o32',
    'cs43',
    # band 6
    'sio54',
    'c18o21',
    '13co21',
    'c17o21',
    'co21',
    'cs54',
    'sio65',
    'hcn32',
    'hcop32',
    'hnc32',
    # band 7
    'cs65',
    'sio76',
    'h2o54?',
    'c18o32',
    '13co32',
    'cs76',
    'co32',
    'sio87',
    'hcn43',
    'hcop43',
    'hcn43',
    # band 8
    'sio98',
    'cs87',
    'sio109',
    'c18o43',
    '13co43',
    'cs98',
    'hcn54',
    'hcop54',
    'c17o43',
    'hnc54',
    'co43',
    'sio1110',
    'cs109',
    'ci10',
    # band 9
    'c17o65',
    'co65',
    'hcn87',
    'hcop87',
    ]

    y = (ylim[1] - ylim[0]) * 0.07 + ylim[1]
    for i in range(len(lines)):
        this_line = lines[i]/(1+z)
        this_name = names[i]
        ax.plot([this_line,this_line],ylim,ls='dashed',color='black')
        ax.text(this_line,y,this_name,rotation=90,ha='center')

def plot_proj(
    selected_data,
    band='3',
    color='tomato',
    xlim=[84.0,116.0],
    outpng='/Users/saitonu/Desktop/n1068_archive_b3_proj.png',
    ):
    """
    """

    # data
    this_band      = selected_data[selected_data[:,4]==band]
    this_freq_list = this_band[:,6]
    x_freq         = []
    z_proj         = []
    for i in range(len(this_freq_list)):
        list_spw = this_freq_list[i].split(' U ')
        this_x_freq = []
        for j in range(len(list_spw)):
            this_spw  = list_spw[j].replace('[','').replace(']','').split(',')
            this_freq = this_spw[0].replace('GHz','').split('..')
            this_x_freq.append([float(this_freq[0]), float(this_freq[1])])
        x_freq.append(this_x_freq)
        z_proj.append(this_band[i,0]+" ("+this_band[i,25].split(",")[0]+", "+str( np.round(this_band[i,9].astype(np.float32),2) )+"\")")

    # plot
    fig = plt.figure(figsize=(13,10))
    plt.rcParams['font.size'] = 18
    gs  = gridspec.GridSpec(nrows=10, ncols=10)
    ax  = plt.subplot(gs[1:10,3:10])

    for i in range(len(z_proj)):
        list_x_freq = x_freq[i]
        for this_x_freq in list_x_freq:
            ax.plot(this_x_freq,[i+1,i+1],color=color,lw=4)

    ax.set_xlim([xlim[0]-2,xlim[1]+2])
    ax.set_ylim([0,len(z_proj)+1])
    ax.set_yticks(np.array(range(len(z_proj)))+1,z_proj,ha='left', fontsize=15)
    ax.get_yaxis().set_tick_params(pad=290)

    ax.grid(axis='y',ls='dotted')
    ax.set_title('Band ' + band + ' Archive as of ' + str(d_today))
    ax.set_xlabel('Observed Frequency (GHz)')
    ax.set_ylabel('Project Code')

    plot_lines(ax,[0,len(z_proj)+1])

    plt.savefig(outpng, dpi=200)

done = glob.glob(dir_plot)
if not done:
    os.system('mkdir ' + dir_plot)

########
# plot #
########
"""
plot_band(data,'3',cm.rainbow(6/6.),[84,116],dir_plot+'n1068_archive_b3.png')
plot_band(data,'4',cm.rainbow(5/6.),[125,163],dir_plot+'n1068_archive_b4.png')
plot_band(data,'5',cm.rainbow(4/6.),[163,211],dir_plot+'n1068_archive_b5.png')
plot_band(data,'6',cm.rainbow(3/6.),[211,275],dir_plot+'n1068_archive_b6.png')
plot_band(data,'7',cm.rainbow(2/6.),[275,373],dir_plot+'n1068_archive_b7.png')
plot_band(data,'8',cm.rainbow(1/6.),[385,500],dir_plot+'n1068_archive_b8.png')
plot_band(data,'9',cm.rainbow(0/6.),[602,720],dir_plot+'n1068_archive_b9.png')
"""

plot_proj(data,'3',cm.rainbow(6/6.),[84,116],dir_plot+'n1068_archive_b3_proj.png')
plot_proj(data,'4',cm.rainbow(5/6.),[125,163],dir_plot+'n1068_archive_b4_proj.png')
plot_proj(data,'5',cm.rainbow(4/6.),[163,211],dir_plot+'n1068_archive_b5_proj.png')
plot_proj(data,'6',cm.rainbow(3/6.),[211,275],dir_plot+'n1068_archive_b6_proj.png')
plot_proj(data,'7',cm.rainbow(2/6.),[275,373],dir_plot+'n1068_archive_b7_proj.png')
plot_proj(data,'8',cm.rainbow(1/6.),[385,500],dir_plot+'n1068_archive_b8_proj.png')
plot_proj(data,'9',cm.rainbow(0/6.),[602,720],dir_plot+'n1068_archive_b9_proj.png')

#######
# end #
#######