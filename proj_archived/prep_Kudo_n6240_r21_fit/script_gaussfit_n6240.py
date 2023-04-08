import os
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from scipy import ndimage

fitsimage = 'n6240_co21_cube.fits'
prefix = fitsimage.replace('n6240_','').replace('_cube.fits','')
rms = 0.3
snr = 1
width_thres = 100. # km/s

########
# main #
########

# データを読み込む
with fits.open(fitsimage) as hdul:
    cube = hdul[0].data
    header = hdul[0].header
    wcs = WCS(header)

# データキューブのサイズを取得
nz, ny, nx = cube.shape
cube[np.isnan(cube)] = 0

# ベストフィットパラメータを保持するために、新しいデータキューブを作成
best_fit_cube = np.zeros((nz, ny, nx))
residual_cube = np.zeros((nz, ny, nx))
broad_cube    = np.zeros((nz, ny, nx))
narrow_cube   = np.zeros((nz, ny, nx))
best_fit_cube[best_fit_cube==0] = np.nan
residual_cube[residual_cube==0] = np.nan
broad_cube[broad_cube==0]       = np.nan
narrow_cube[narrow_cube==0]     = np.nan

# 速度分解能を計算
restfreq = header['RESTFRQ'] * u.Unit(header['CUNIT3'])
freq_res = abs(header['CDELT3']) * u.Unit(header['CUNIT3'])
freq_to_vel = u.doppler_radio(restfreq)
vel_res = (restfreq-freq_res).to(u.km / u.s, equivalencies=freq_to_vel)

def gaussian1(x, a1, b1, c1):
    return a1 * np.exp(-(x - b1) ** 2 / (2 * c1 ** 2))

def gaussian2(x, a1, b1, c1, a2, b2, c2):
    return a1 * np.exp(-(x - b1) ** 2 / (2 * c1 ** 2)) + a2 * np.exp(-(x - b2) ** 2 / (2 * c2 ** 2))

def numpy2fits(
    infile,
    vel_res,
    out_prefix,
    ):

    mom0 = np.sum(infile, axis=0) * vel_res
    mom1 = np.argmax(infile, axis=0)
    mom1 = np.where(mom1!=0, (mom1 - np.nanmean(mom1)) * vel_res, np.nan)
    mom2 = np.sum(infile, axis=0) * vel_res / np.max(infile, axis=0) / np.sqrt(2*np.pi)
    mom8 = np.max(infile, axis=0)

    os.system('rm -rf ' + out_prefix + '_mom0.fits')
    hdu = fits.PrimaryHDU(mom0, header=header)
    hdu.writeto(out_prefix + '_mom0.fits')

    os.system('rm -rf ' + out_prefix + '_mom1.fits')
    hdu = fits.PrimaryHDU(mom1, header=header)
    hdu.writeto(out_prefix + '_mom1.fits')

    os.system('rm -rf ' + out_prefix + '_mom2.fits')
    hdu = fits.PrimaryHDU(mom2, header=header)
    hdu.writeto(out_prefix + '_mom2.fits')

    os.system('rm -rf ' + out_prefix + '_mom8.fits')
    hdu = fits.PrimaryHDU(mom8, header=header)
    hdu.writeto(out_prefix + '_mom8.fits')

# 各ピクセルを反復処理し、フィットを実行します
for i in range(ny):
    for j in range(nx):

        # (i,j)座標からスペクトルを取得
        spectrum = cube[:, i, j]
        spectrum[np.isnan(spectrum)] = 0
        peak = np.nanmax(spectrum)

        # 移動平均畳み込み
        if i>1 and j>1:
            spectrum = np.average(np.average(cube[:, i-2:i+3, j-2:j+3],axis=1),axis=1)

        # スペクトルのピークが指定したSN比以下の時にスキップ
        if peak < rms * snr:
            continue

        # ガウシアン2つでフィッティング
        try:
            # intial guess
            position = np.nanargmax(spectrum)
            disp = np.nanstd(spectrum)
            p0 = [peak,position,disp*0.2,peak*0.2,position,disp*7]

            popt, _ = curve_fit(gaussian2, np.arange(nz), spectrum, p0=p0)

            # 小さい方のガウシアンがSN比以上の場合は保存: best, residual
            if np.max([popt[0],popt[3]])>rms*snr:
                best_fit_cube[:, i, j] = gaussian2(np.arange(nz), *popt)
                residual_cube[:, i, j] = spectrum - best_fit_cube[:, i, j]

            # 小さい方のガウシアンがSN比以上の場合は保存: narrow, broad
            if np.max([popt[0],popt[3]])>rms*snr:
                if abs(popt[5])>abs(popt[2]):
                    if popt[5]<width_thres/vel_res.value:
                        popt[5]=np.nan
                    narrow_cube[:, i, j] = gaussian1(np.arange(nz), popt[0], popt[1], popt[2])
                    broad_cube[:, i, j]  = gaussian1(np.arange(nz), popt[3], popt[4], popt[5])
                else:
                    if popt[2]<width_thres/vel_res.value:
                        popt[2]=np.nan
                    narrow_cube[:, i, j] = gaussian1(np.arange(nz), popt[3], popt[4], popt[5])
                    broad_cube[:, i, j]  = gaussian1(np.arange(nz), popt[0], popt[1], popt[2])

        # ガウシアン1つでフィッティング
        except:
            try:
                # intial guess
                position = np.nanargmax(spectrum)
                disp = np.nanstd(spectrum)
                p0 = [peak, position, disp]

                popt, _ = curve_fit(gaussian1, np.arange(nz), spectrum, p0=p0)

                # 保存: best, residual
                best_fit_cube[:, i, j] = gaussian1(np.arange(nz), *popt)
                residual_cube[:, i, j] = spectrum - best_fit_cube[:, i, j]

                # 保存: narrow, broad
                if abs(popt[2])>width_thres/vel_res.value:
                    narrow_cube[:, i, j] = np.nan
                    broad_cube[:, i, j]  = gaussian1(np.arange(nz), *popt)
                else:
                    narrow_cube[:, i, j] = gaussian1(np.arange(nz), *popt)
                    broad_cube[:, i, j]  = np.nan

            except:
                continue

# export: broad component
os.system('rm -rf ' + prefix + '_broad_cube.fits')
hdu = fits.PrimaryHDU(broad_cube, header=header)
hdu.writeto(prefix + '_broad_cube.fits')
numpy2fits(broad_cube,vel_res.value,prefix + '_broad')

# export: narrow component
os.system('rm -rf ' + prefix + '_narrow_cube.fits')
hdu = fits.PrimaryHDU(narrow_cube, header=header)
hdu.writeto(prefix + '_narrow_cube.fits')
numpy2fits(narrow_cube,vel_res.value,prefix + '_narrow')

# export: best component
os.system('rm -rf ' + prefix + '_best_cube.fits')
hdu = fits.PrimaryHDU(best_fit_cube, header=header)
hdu.writeto(prefix + '_best_cube.fits')
numpy2fits(best_fit_cube,vel_res.value,prefix + '_best')

# export: obs data
numpy2fits(cube,vel_res.value,prefix + '_data')

#######
# end #
#######