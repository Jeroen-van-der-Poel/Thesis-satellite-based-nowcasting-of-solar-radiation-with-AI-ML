#!/usr/bin/env python3

import sys
import os
import glob
import subprocess
import datetime
import numpy as np


def main():
    '''
    This script reduces an MSG-CPP subset file to a further subset
    '''

    # start/end date
    dt_s = datetime.datetime(2024, 1, 1, 0, 0, 0)
    dt_e = datetime.datetime(2024, 12, 31, 0, 0, 0)

    # Directory with input/output
    datadir_in_base = '/nobackup_1/users/meirink/msgcpp_aws/for_hkv/'
    datadir_out_base = '/net/pc200258/nobackup_1/users/meirink/Jeroen/'
    
    # Satellite and coverage
    #    SEVIR + AODC, SEVIR + IODC, ABI__ + EAST, AHI__ + PRIM
    sat = 'SEVIR'
    cov = 'AODC'

    # Which variables to select
    #varlist = 'sds,sds_cs,x,y,time,projection,lon,lat'
    varlist = 'sds,sds_cs,x,y,time,projection'

    # Areas for input and output (defined with respect to full disk)
    areasel_in = [1467, 2681, 141, 763]     # region HKV+DXT (approx. Europe)
    #areasel_out = [1711, 2100, 260, 509]    # region covering Benelux and France (adjusted to 390×250)
    areasel_out = [1711, 2100, 257, 512]  # region covering Benelux and France (adjusted to 390×256)

    # Extraction 
    areasel = [areasel_out[0] - areasel_in[0], areasel_out[1] - areasel_in[0],
               areasel_out[2] - areasel_in[2], areasel_out[3] - areasel_in[2]]
    xsel = 'x,'+str(areasel[0])+','+str(areasel[1])
    ysel = 'y,'+str(areasel[2])+','+str(areasel[3])

    width = areasel[1] - areasel[0] + 1
    height = areasel[3] - areasel[2] + 1
    print(f"{height}x{width}")

    dt = dt_s
    tstep = datetime.timedelta(minutes=15)

    while dt < dt_e:
        datadir_in = datadir_in_base + dt.strftime('%Y/')
        datadir_out = datadir_out_base + dt.strftime('%Y/')
        if not os.path.isdir(datadir_out): subprocess.call(['mkdir', '-p', datadir_out])
        
        pattern = datadir_in + sat + '*' + cov + '_L2__' + dt.strftime('%Y%m%dT%H%M' + '*.nc')
        files = glob.glob(pattern)
        
        if len(files) == 0:
            print('No input file for: ' + dt.strftime('%Y%m%dT%H%M'))
        else:
            fname_in = files[0]
            fname_out = datadir_out + files[0].split('/')[-1][:-3] + '_reduced.nc'
            
            print('Creating file; ' + fname_out)
            subprocess.call(['ncks', '-C', '-v', varlist, '-d', xsel, '-d', ysel,
                             fname_in, '-O', fname_out])

        dt += tstep


main()
