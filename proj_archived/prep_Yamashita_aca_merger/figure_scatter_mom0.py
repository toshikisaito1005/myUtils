import os, glob, math
import numpy as np
from PIL import Image, ImageOps
from astropy.io import fits
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ms9201 xaxis = 6.34 kpc
# vv0316 xaxis = 24.2 kpc

#from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
#cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
from astropy.cosmology import WMAP9 as cosmo

# csvファイルを読み込む
csvfile = 'doc/GSWLC_v20230404.csv'
data = np.loadtxt(csvfile, delimiter=',', dtype = 'unicode')

# 天体名を変更する
name = []
for this_name in data[:,0]:
    this_name = this_name.lower()
    if 'disk' in this_name:
        this_name = 'ms' + this_name[-4:]
    name.append(this_name)

# カタログ再構成
catalog = np.c_[
    np.array(data[:,13]).astype(np.float32), # z
    np.array(data[:,15]).astype(np.float32), # log10 Mstar
    np.array(data[:,16]).astype(np.float32), # error log10 Mstar
    np.array(data[:,17]).astype(np.float32), # log10 SFR
    np.array(data[:,18]).astype(np.float32), # error log10 SFR
    ]

os.system('mkdir temp_png/')

"""
0: Yamashita aca ID
1: R.A.
2: Decl.
3: ALMA array
6: GSWLC ID
11: R.A. (deg)
12: Decl. (deg)
13: redshift
15: stelalr mass in log10
16: error
17: SFR in log10
18: error
19: A_FUV
20: error
21: A_B
22: error
23: A_V
24: error
"""

# main
images = []
zoom = []
for i in range(len(name)):
    this_gal = name[i]
    dist = cosmo.luminosity_distance(catalog[i,0])
    fitsimage = glob.glob('data_raw/' + this_gal.replace('vv','vv*') + '/' + this_gal.replace('vv','vv*') + '_7m_co21_strict_mom0.fits')[0]

    # FITSファイルを読み込む
    hdul = fits.open(fitsimage)
    data = hdul[0].data

    # ヘッダーからピクセルのサイズ情報を取得
    pixel_size = hdul[0].header['CDELT2']
    image_size = data.shape[0]

    # 描画範囲を10 kpc x 10 kpc にする
    arcsec_per_pixel = abs(pixel_size) * 3600
    kpc_per_pixel = (dist * math.tan(math.radians(arcsec_per_pixel/3600.))).value * 1000
    #new_size = int(18 / kpc_per_pixel)
    #left = int((image_size - new_size) / 2)
    #top = int((image_size - new_size) / 2)
    #right = int((image_size + new_size) / 2)
    #bottom = int((image_size + new_size) / 2)

    # 画像データを0-255に正規化してカラースケール反転
    data[np.isnan(data)] = 0
    # data[left:right, top:bottom]
    img_data = ((data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))) * 255
    #img_data = (255 - img_data)
    if 'vv' in this_gal:
        img_data = np.dstack([img_data, img_data*0.3, img_data*0.3])
    else:
        img_data = np.dstack([img_data*0.3, img_data*0.3, img_data])

    # PIL Imageを作成してデータを挿入
    img = Image.fromarray(np.uint8(img_data))#, mode='P')
    img = img.convert('RGBA')
    img = img.rotate(180)
    img = ImageOps.mirror(img)

    # 背景を透明にする
    pixdata = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y][0] == 0 and pixdata[x, y][1] == 0 and pixdata[x, y][2] == 0:
                pixdata[x, y] = (255, 255, 255, 0)

    # PNGファイルとして保存
    img.save('temp_png/' + this_gal + '_mom0.png')
    images.append('temp_png/' + this_gal + '_mom0.png')

    # 
    #img_data
    zoom.append(kpc_per_pixel)

# plot
fig, ax = plt.subplots(figsize=(7,5))

for i in range(len(images)):
    this_x = catalog[i,1]
    this_y = catalog[i,3] - catalog[i,1]
    this_zoom = zoom[i] * 1.3
    this_im = images[i]
    this_im = plt.imread(this_im)
    this_im = OffsetImage(this_im, zoom=this_zoom)
    this_im.image.axes = ax

    if 'ms9201' in images[i]:
        this_y = this_y - 0.03
    elif 'ms1126' in images[i]:
        this_y = this_y - 0.04
        this_x = this_x + 0.02
    elif 'ms6552' in images[i]:
        this_y = this_y - 0.03
        this_x = this_x + 0.03
    elif 'ms8714' in images[i]:
        this_y = this_y + 0.03
        this_x = this_x + 0.03
    elif 'ms2018' in images[i]:
        this_y = this_y - 0.03
        this_x = this_x - 0.04
    elif 'ms7882' in images[i]:
        this_y = this_y + 0.03
        this_x = this_x - 0.02
    elif 'ms7994' in images[i]:
        this_x = this_x - 0.04
    elif 'ms1181' in images[i]:
        this_y = this_y - 0.09
        this_x = this_x - 0.07
    elif 'ms5961' in images[i]:
        this_y = this_y - 0.09
        this_x = this_x + 0.03
    elif 'ms7069' in images[i]:
        this_y = this_y + 0.19
        this_x = this_x - 0.04
    elif 'ms9438' in images[i]:
        this_x = this_x + 0.09

    elif 'vv754s' in images[i]:
        this_y = this_y + 0.18
        this_x = this_x - 0.02
    elif 'vv316' in images[i]:
        this_y = this_y + 0.05
    elif 'vv847' in images[i]:
        this_y = this_y - 0.03

    ab = AnnotationBbox(this_im,
                        [this_x, this_y],
                        xycoords='data',
                        pad=0.0,
                        frameon=False,
                        )
    ax.add_artist(ab)

# z0MGS MS
mmin = 9.5
mmax = 11.2
ssfrmin = -0.32*np.log10(10**mmin/10**10) - 10.17
ssfrmax = -0.32*np.log10(10**mmax/10**10) - 10.17
ax.plot([mmin, mmax],[ssfrmin+1, ssfrmax+1], 'k--', lw=1, color='grey')
ax.plot([mmin, mmax],[ssfrmin, ssfrmax], 'k--', lw=2, color='grey')
ax.plot([mmin, mmax],[ssfrmin-1, ssfrmax-1], 'k--', lw=1, color='grey')

# Fix the display limits to see everything
ax.set_xlim(mmin, mmax)
ax.set_ylim(-12.4, -9.2)

ax.set_xlabel('log$_{10}$ $M_{\star}$ (M$_{\odot}$)')
ax.set_ylabel('log$_{10}$ SFR / $M_{\star}$ (yr$^{-1}$)')

ax.text(0.04,0.10,'merging galaxies',color='red',transform=ax.transAxes,horizontalalignment='left',weight='bold')
ax.text(0.04,0.05,'disk galaxies',color='blue',transform=ax.transAxes,horizontalalignment='left',weight='bold')

ax.plot([10.98-1.365,11.0337-1.365],[-11.9,-11.9],"k-")
ax.text(0.04,0.17,'10 kpc',color='black',transform=ax.transAxes,horizontalalignment='left')

plt.savefig('png/scatter.png',dpi=200, transparent=False)

os.system('rm -rf temp_png/')

#######
# end #
#######