"""
Standalone routines that are used for hyperfine fitting using CASA.

contents:
    hf_cn10

history:
2021-12-14   created
Toshiki Saito@Nichidai/NAOJ
"""

import os, sys
import mycasa_tasks as mytask
reload(mytask)

execfile(os.environ["HOME"] + "/myUtils/stuff_casa.py")

#

##########
# fitmom #
##########

def hf_cn10(
    cubeimage,
    ):
    """
    """

    taskname = sys._getframe().f_code.co_name
    mytask.check_first(cubeimage, taskname)

    data,_ = mytask.imval_all(cubeimage)

    data = data["data"]
    

#############
# _get_grid #
#############

def _get_grid(imagename):

    print("# _get_grid " + imagename.split("/")[-1])

    head  = imhead(imagename,mode="list")
    shape = head["shape"][0:2]
    pix   = abs(head["cdelt1"]) * 3600 * 180/np.pi
    
    return shape, pix
