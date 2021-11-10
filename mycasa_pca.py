"""
Standalone plotting routines that use CASA.

contents:
    pca_2d

history:
2021-08-02   created by TS
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys, glob, copy, math, inspect
import numpy as np
import pyfits
import matplotlib.pyplot as plt
plt.ioff()

import numpy.linalg as LA
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

from mycasa_tasks import *
from mycasa_plots import *

#

modname = "mypca."

##############
# hex_pca_2d #
##############

def pca_2d_hex(
    X,
    Y,
    data,
    list_name,
    output,
    bstr,
    snr=3.0,
    beam=0.8,
    gridsize=70,
    ):
    """
    References
    ----------
    https://tips-memo.com/python-2d-pca
    """

    # prepare hex grid
    size   = 0.5 * gridsize * beam
    extent = [size, -size, -size, size]

    # eigenvector
    data    = np.array(data)
    num     = len(data[0,:])
    m       = np.mean(data, axis=0)
    data_m  = data - m[None, :]
    S_inner = np.array((1./num) * ((data_m).dot((data_m).T)))

    [eig, u_inner] = _eigsort(S_inner)
    u = np.array( (data_m.T.dot(u_inner)) / np.sqrt(num * eig) )

    # derive CCR
    y = np.cumsum(eig)
    ylist = y / y[-1]
    index = np.where(ylist>=0.9)[0]

    print(ylist)
    print("The number of axes which first achieve 90%: " + str(index[0] + 1))

    # visualize
    principals = index[0] + 1
    u_compressed = u[:, 0:principals]

    # re hex sampling
    u_drawing = []
    for i in range(principals):
        this_u = u_compressed[:,i]

        fig = plt.figure()
        gs  = gridspec.GridSpec(nrows=1,ncols=1)
        ax  = plt.subplot(gs[0:1,0:1])
        hexdata = ax.hexbin(X, Y, C=this_u, gridsize=gridsize, extent=extent)
        this_hex = np.array(hexdata.get_array())

        plt.clf()

        u_drawing.append(this_hex)

    # plot eigenvector
    for i in range(principals):
        this_out = output.replace(bstr+".png","_pc"+str(i+1)+bstr+".png")
        myfig_hex_map(-X, Y, u_drawing[i], this_out, beam=beam,
        gridsize=gridsize, cmap="bwr", cblabel=None, zerocbar=True)

    # plot clusters
    pca_score        = np.dot(data_m, u_compressed)

    # plot scatter
    for i in range(principals):

        if i==0:
            continue

        pc1         = pca_score[:,0]
        this_pc     = pca_score[:,i]
        this_pcname = "PC" + str(i+1)

        fig = plt.figure(figsize=(10,9))
        gs  = gridspec.GridSpec(nrows=30, ncols=30)
        ax  = plt.subplot(gs[0:30,0:30])
        myax_set(ax,xlim=None,ylim=None,title=None)

        for j in range(len(list_name)):
            x, y = pc1[j], this_pc[j]
            ax.plot([0,x],[0,y])
            this_name = list_name[j]
            ax.text(x,y,this_name,fontsize=14)

        ax.set_xlabel("PC1")
        ax.set_ylabel(this_pcname)

        new_name = "_pc1_vs_pc" + str(i+1) + bstr + ".png"
        this_out = output.replace(bstr + ".png", new_name)
        plt.savefig(this_out)

############
# _eigsort #
############

def _eigsort(S):
    eigv_raw, u_raw = LA.eig(S)
    eigv_index = np.argsort(eigv_raw)[::-1]
    eigv = eigv_raw[eigv_index]
    u = u_raw[:, eigv_index]

    return [eigv, u]

