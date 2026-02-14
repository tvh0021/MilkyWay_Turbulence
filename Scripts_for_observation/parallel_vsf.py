# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:16:50 2021

@author: Trung Ha
"""
import functions as fcs
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
import os
def vsf_from_sample(m,n_bins,d_max,sample_size,file_loc):
    i = int(m)
    tableq = pd.DataFrame()
    for k in range(len(file_loc)):
        if os.name == 'nt':
            table_read = pd.read_csv(file_loc[k]+'\\generated_'+str(i)+'.csv',index_col=0)
        else:
            table_read = pd.read_csv(file_loc[k]+'/generated_'+str(i)+'.csv',index_col=0)
        tableq = tableq.append(table_read,ignore_index=True)
     
    tableq = tableq.drop([np.random.randint(0,len(tableq))]) #drop 1 random element in each generated realization
    
    if list(tableq.columns).__contains__('vlsr' and 'vlsrra' and 'vlsrdec') == False: # earlier iterations of the random samples generator named the columns differently
        tableq['vlsr'] = tableq['rv']
        tableq['vlsrra'] = tableq['vra']
        tableq['vlsrdec'] = tableq['vdec']

    celes_pos = SkyCoord(ra=tableq.ra.values*u.deg, dec=tableq.dec.values*u.deg, distance=tableq.dist.values*u.pc, radial_velocity=tableq.vlsr.values*u.km/u.s, pm_ra_cosdec=tableq.vlsrra.values*u.mas/u.yr, pm_dec=tableq.vlsrdec.values*u.mas/u.yr, frame='icrs')
    galac_pos = celes_pos.galactic

    tableq['lat'] = np.asarray(galac_pos.b.data)
    tableq['lon'] = np.asarray(galac_pos.l.data)
    galac_pos.representation_type = 'cartesian'
    
    tableq['x_axis'] = np.asarray(galac_pos.u.data)
    tableq['y_axis'] = np.asarray(galac_pos.v.data)
    tableq['z_axis'] = np.asarray(galac_pos.w.data)
    tableq['v_x'] = np.asarray(galac_pos.U.data)
    tableq['v_y'] = np.asarray(galac_pos.V.data)
    tableq['v_z'] = np.asarray(galac_pos.W.data)
        
    dist_array, v_diff_mean, np_dist, v_diff_sigma = fcs.VSF_3Dcart(tableq.x_axis.values,tableq.y_axis.values,tableq.z_axis.values,tableq.v_x.values,tableq.v_y.values,tableq.v_z.values,d_max,n_bins,1,savemem=1)
    return dist_array,v_diff_mean, v_diff_sigma

