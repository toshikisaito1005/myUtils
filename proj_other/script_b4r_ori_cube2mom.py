import os, sys, glob

imagename = '1OMC1.large.079134.spw1.LSRK.image_n_full8chrebin-14asec.fits'

# line to select
#chans     = '1690~1751' # H35alpha
#linename  = 'h35alpha'

chans     = '3836~3849' # c33s
linename  = 'c33s'

#########
# start #
#########
outcube = '1OMC1_' + linename + '_14as.cube'
outmask = '1OMC1_' + linename + '_14as.cubemask'
outmom  = '1OMC1_' + linename + '_14as.mom'

### extract the line
os.system('rm -rf ' + outcube)
imsubimage(
    imagename = imagename,
    chans     = chans,
    outfile   = outcube,
    )
os.system('rm -rf imsubimage.last')

### measure original rms
rms = imstat(outcube)['rms'][0]
os.system('rm -rf imstat.last')

### smooth the cube
os.system('rm -rf ' + outcube + '.smooth')
imsmooth(
    imagename = outcube,
    major     = '70arcsec',
    minor     = '70arcsec',
    pa        = '0deg',
    targetres = True,
    outfile   = outcube + '.smooth',
    )
os.system('rm -rf imsmooth.last')

### measure smoothed rms
rms_smooth = imstat(outcube + '.smooth')['rms'][0]
os.system('rm -rf imstat.last')

### create cube mask
os.system('rm -rf ' + outmask + '_tmp1')
immath(
    imagename = outcube + '.smooth',
    expr      = 'iif( IM0>' + str(rms_smooth*3) + ', 1.0 , 0.0 )',
    outfile   = outmask + '_tmp1',
    )
os.system('rm -rf ' + outmask + '_tmp2')
specsmooth(
    imagename = outmask + '_tmp1',
    outfile   = outmask + '_tmp2',
    dmethod   = '',
    function  = 'boxcar',
    width     = 3,
    )
os.system('rm -rf specsmooth.last')
os.system('rm -rf ' + outmask + '_tmp3')
immath(
    imagename = outmask + '_tmp2',
    expr      = 'iif( IM0>0, 1.0 , 0.0 )',
    outfile   = outmask + '_tmp3',
    )
os.system('rm -rf ' + outmask + '_tmp2')

### clip the last channel of outcube
os.system('rm -rf ' + outcube + '_tmp1')
imsubimage(
    imagename = imgaename,
    chans     = str(int(chans.split('~')[0])+1) + '~' + str(int(chans.split('~')[1])-1),
    outfile   = outcube + '_tmp1',
    )
os.system('rm -rf imsubimage.last')

### replace 0 with mask
os.system('rm -rf ' + outcube + '.smooth')
os.system('rm -rf ' + outmask)
makemask(
    mode = 'copy',
    inpimage = outmask + '_tmp3',
    inpmask  = outmask + '_tmp3',
    output   = outmask + ':mask0',
    )
os.system('rm -rf immath.last makemask.last ' + outmask + '_tmp1')
os.system('rm -rf ' + outmask + '_tmp3')

### apply mask
os.system('rm -rf ' + outcube + '.masked')
immath(
    imagename = [outcube + '_tmp1', outmask],
    expr      = 'IM0*IM1',
    outfile   = outcube + '.masked',
    )
os.system('rm -rf immath.last')

### moments
os.system('rm -rf ' + outmom + "*")
immoments(
    imagename = outcube + '.masked',
    moments   = [0],
    outfile   = outmom + '0',
    )
immoments(
    imagename  = outcube + '.masked',
    moments    = [1],
    includepix = [rms*1,1000000],
    outfile    = outmom + '1',
    )
immoments(
    imagename  = outcube + '.masked',
    moments    = [2],
    includepix = [rms*1,1000000],
    outfile    = outmom + '2',
    )
os.system('rm -rf immoments.last')
os.system('rm -rf ' + outcube + '_tmp1')

#######
# end #
#######
