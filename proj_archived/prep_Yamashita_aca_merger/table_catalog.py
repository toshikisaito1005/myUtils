import glob, math
import numpy as np

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
dat = np.c_[
    name,
    np.array(data[:,4]), # ra
    np.array(data[:,5]), # dec
    np.array(data[:,13]), # z
    np.array(data[:,15]), # log10 Mstar
    np.array(data[:,16]), # error log10 Mstar
    np.array(data[:,17]), # log10 SFR
    np.array(data[:,18]), # error log10 SFR
    ]

for i in range(len(name)):
    ra    = str(np.round(float(dat[i,1]),6))
    ra    = ra.split('.')[0] + '.' + ra.split('.')[1].rjust(6, '0')
    dec   = str(np.round(float(dat[i,2]),6)).replace('-','$-$')
    dec   = dec.split('.')[0].rjust(5, ' ') + '.' + dec.split('.')[1].rjust(6, '0')
    mass  = str(np.round(float(dat[i,4]),2)).ljust(5, '0')
    emass = str(np.round(float(dat[i,5]),2)).ljust(4, '0').replace('0.00','0.01')
    sfr   = str(np.round(float(dat[i,6]),2)).ljust(4, '0').replace('-','$-$').rjust(7, ' ')
    esfr = str(np.round(float(dat[i,7]),2)).ljust(4, '0')

    this = dat[i,0].rjust(6, ' ') + ' & ' + \
           ra.rjust(10, ' ') + ' & ' + \
           dec.rjust(10, ' ') + ' & ' + \
           dat[i,3].ljust(6, '0') + ' & ' + \
           mass + ' \\pm ' + emass + ' & ' + \
           sfr + ' \\pm ' + esfr + ' \\\\'
    print(this)

#######
# end #
#######