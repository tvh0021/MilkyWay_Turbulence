#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:59:33 2021

This script is for functions only. This works better since all functions can be optimized here alone without affecting the individual codes

@author: trungha
"""

import warnings
warnings.filterwarnings("ignore")

#%% Function to convert ra-dec coordinate and velocity to cartesian coordinate. Distance in pc; ra, dec in degrees; radial velocity in km/s; vra, vdec in mas/yr
import numpy as np
def radec2cartesian(ra,dec,dist,rv,vra,vdec):
    scale_dist = (3.0857 * 10**13) # scale factor from pc to km, applied to dist
    scale_vel = 1./(3.6 * 10**6) * 1./(3.1536 * 10**7) * np.pi / 180. # scale factor from mas to degrees, years to seconds, degrees to radians, applied to vdec and vra to convert from mas/yr to rad/s.
    x = dist * np.cos(dec) * np.cos(ra)
    y = dist * np.cos(dec) * np.sin(ra)
    z = dist * np.sin(dec)
    v_x = rv * np.cos(dec) * np.cos(ra) - dist * scale_dist * np.sin(dec) * np.cos(ra) * vdec * scale_vel - dist * scale_dist * np.cos(dec) * np.sin(ra) * vra * scale_vel
    v_y = rv * np.cos(dec) * np.sin(ra) - dist * scale_dist * np.sin(dec) * np.sin(ra) * vdec * scale_vel + dist * scale_dist * np.cos(dec) * np.cos(ra) * vra * scale_vel
    v_z = rv * np.sin(dec) + dist * scale_dist * np.cos(dec) * vdec * scale_vel
    return x,y,z,v_x,v_y,v_z

#%% Function to determine the distance and velocity error in x, y, and z
def err_cartesian(ra,dec,dist,dist_err,vra,vdec,rv_err,vra_err,vdec_err):
    scale_dist = (3.08567782 * 10**13) # scale factor from pc to km, applied to dist
    scale_vel = 1./(3.6 * 10**6) * 1./(31536000) * np.pi / 180. # scale factor from mas to degrees, years to seconds, degrees to radians, applied to vdec and vra to convert from mas/yr to rad/s.
    err_x = abs(np.cos(dec) * np.cos(ra) * dist_err)
    err_y = abs(np.cos(dec) * np.sin(ra) * dist_err)
    err_z = abs(np.sin(dec) * dist_err)
    err_vx = np.sqrt((np.cos(dec)*np.cos(ra)*rv_err)**2 + (scale_dist*scale_vel*(np.sin(dec)*np.cos(ra)*vdec+np.cos(dec)*np.sin(ra)*vra)*dist_err)**2 + (dist*scale_vel*scale_dist*np.sin(dec)*np.cos(ra)*vdec_err)**2 + (dist*scale_vel*scale_dist*np.cos(dec)*np.sin(ra)*vra_err)**2)
    err_vy = np.sqrt((np.cos(dec)*np.sin(ra)*rv_err)**2 + (scale_dist*scale_vel*(-np.sin(dec)*np.sin(ra)*vdec+np.cos(dec)*np.cos(ra)*vra)*dist_err)**2 + (dist*scale_vel*scale_dist*np.sin(dec)*np.sin(ra)*vdec_err)**2 + (dist*scale_vel*scale_dist*np.cos(dec)*np.cos(ra)*vra_err)**2)
    err_vz = np.sqrt((np.sin(dec)*rv_err)**2 + (scale_dist*scale_vel*np.cos(dec)*vdec*dist_err)**2 + (dist*scale_dist*scale_vel*np.cos(dec)*vdec_err)**2)
    return err_x, err_y, err_z, err_vx, err_vy, err_vz

#%% Function to generate a random sample based on the measured data and its uncertainties
import pandas as pd
def rand_sample (sample, size, directory):
    np.random.seed(2) #seed to ensure replicable generated dataset
    # n_1_sample = sample.drop([np.random.randint(1,len(sample))])
    n_1_sample = sample.reset_index() #just need to do this to avoid indexing conflict
    
    Pool = np.zeros((len(n_1_sample),6,size)) #3d np array that saves everything
    
    for i in range (0,len(n_1_sample)): #each iteration take the measurement and error of 1 star to generate "size" number of gaussian random number of the listed properties
        generate_vlsr = np.random.normal(n_1_sample.vlsr[i], n_1_sample.vlsr_err[i], size)
        generate_vlsrra = np.random.normal(n_1_sample.vlsrra[i], n_1_sample.vlsrra_err[i], size)
        generate_vlsrdec = np.random.normal(n_1_sample.vlsrdec[i], n_1_sample.vlsrdec_err[i], size)
        generate_dist = np.random.normal(n_1_sample.dist[i], n_1_sample.dist_err[i], size)
        for j in range (0,size): #each iteration assign the previously generated gaussian distribution into separate pool 
            Pool[i,0,j] = generate_dist[j]
            Pool[i,1,j] = generate_vlsr[j]
            Pool[i,2,j] = generate_vlsrra[j]
            Pool[i,3,j] = generate_vlsrdec[j]
            Pool[i,4,j] = n_1_sample.ra[i]
            Pool[i,5,j] = n_1_sample.dec[i]
        print('Star '+str(i+1)+' generated.')
    
    for k in range (0,size):
        frame = pd.DataFrame(Pool[:,:,k], columns=('dist','vlsr','vlsrra','vlsrdec','ra','dec')) #each of these "frame" is a set of randomly generated set of stars based on the normal distribution of observed stars in the set
        frame.to_csv(directory+'generated_'+str(k)+'.csv')
        
    print('Done, files saved.')
    return 

#%% Function to generate a random sample based on a PPV dataset
def PPVrand_sample (sample, size, directory):
    np.random.seed(1) #seed to ensure replicable generated dataset
    # n_1_sample = sample.drop([np.random.randint(1,len(sample))])
    n_1_sample = sample.reset_index() #just need to do this to avoid indexing conflict
    
    Pool = np.zeros((len(n_1_sample),5,size)) #3d np array that saves everything
    
    for i in range (0,len(n_1_sample)): #each iteration take the measurement and error of 1 pixel to generate "size" number of gaussian random number of the listed properties
        generate_vlsr = np.random.normal(n_1_sample.vlsr[i], n_1_sample.vlsr_err[i], size)
        for j in range (0,size): #each iteration assign the previously generated gaussian distribution into separate pool 
            Pool[i,0,j] = generate_vlsr[j]
            Pool[i,1,j] = n_1_sample.x_axis[i]
            Pool[i,2,j] = n_1_sample.y_axis[i]
            Pool[i,3,j] = n_1_sample.z_axis[i]
            Pool[i,4,j] = n_1_sample.flux[i]
        print('Pixel '+str(i+1)+' generated.')
    
    for k in range (0,size):
        frame = pd.DataFrame(Pool[:,:,k], columns=('vlsr','X','Y','Z','flux')) #each of these "frame" is a set of randomly generated set of pixels based on the normal distribution of observed flux-velocity in the set
        frame.to_csv(directory+'generated_'+str(k)+'.csv')
    print('Done, files saved.')
    return 

#%% 3d Cartesian vsf
from scipy.stats import norm
from tqdm import tqdm
# Function to calculate any order of 3D VSFs up to 3rd
def VSF_3Dcart(X,Y,Z,vx,vy,vz,d_max,n_bins=50,order=1,savemem=0):
    
    if savemem == 2: #default is float64, but it takes too much memory. So if needed, reduce to float32
        X = X.astype('float16')
        Y = Y.astype('float16')
        Z = Z.astype('float16')
        vx = vx.astype('float16')
        vy = vy.astype('float16')
        vz = vz.astype('float16')
    elif savemem == 1:
        X = X.astype('float32')
        Y = Y.astype('float32')
        Z = Z.astype('float32')
        vx = vx.astype('float32')
        vy = vy.astype('float32')
        vz = vz.astype('float32')
    
    #Velocity difference matrix
    vel_xa = np.reshape(vx, (vx.size, 1))
    vel_xb = np.reshape(vx, (1, vx.size))
    vel_ya = np.reshape(vy, (vy.size, 1))
    vel_yb = np.reshape(vy, (1, vy.size))
    vel_za = np.reshape(vz, (vz.size, 1))
    vel_zb = np.reshape(vz, (1, vz.size))
    v_diff_matrix = np.sqrt((vel_xa - vel_xb)**2 + (vel_ya - vel_yb)**2 + (vel_za - vel_zb)**2)
    
    #distance matrix
    px_a = np.reshape(X, (X.size, 1))
    px_b = np.reshape(X, (1, X.size))
    py_a = np.reshape(Y, (Y.size, 1))
    py_b = np.reshape(Y, (1, Y.size))
    pz_a = np.reshape(Z, (Z.size, 1))
    pz_b = np.reshape(Z, (1, Z.size))
    dist_matrix = np.sqrt((px_a - px_b)**2 + (py_a - py_b)**2 + (pz_a - pz_b)**2)
    
    v_diff_half = np.ndarray.flatten(np.triu(v_diff_matrix, k=0))
    dist_half = np.ndarray.flatten(np.triu(dist_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0 
    del dist_matrix, v_diff_matrix
    
    good_dist = dist_half>0
    
    np_dist=dist_half[good_dist]
    np_v_diff=v_diff_half[good_dist]
    del good_dist
    
    # dist_array=np.logspace(np.log10(np.min(np_dist)),np.log10(d_max),n_bins)
    dist_array=np.logspace(np.log10(1),np.log10(d_max),n_bins)
    v_diff_mean=np.zeros(n_bins)
    v_diff_sigma=np.zeros(n_bins)
    num_in_bin = np.zeros(n_bins)
    
    if order == 2:
        np_v_diff_2=np_v_diff**2
        v_diff_mean2 = np.zeros(n_bins)
        v_diff_sigma2 = np.zeros(n_bins)
        for i in range(0,n_bins-1): 
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            (v_diff_mean2[i],v_diff_sigma2[i])=norm.fit(np_v_diff_2[this_bin])
        v_diff_mean = v_diff_mean2
        v_diff_sigma = v_diff_sigma2
    elif order == 3:
        np_v_diff_3=np_v_diff**3
        v_diff_mean3=np.zeros(n_bins)
        v_diff_sigma3=np.zeros(n_bins)
        for i in range(0,n_bins-1):
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            (v_diff_mean3[i],v_diff_sigma3[i])=norm.fit(np.abs(np_v_diff_3[this_bin]))
        v_diff_mean = v_diff_mean3
        v_diff_sigma = v_diff_sigma3
    else:
         for i in range(0,n_bins-1):
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            num_in_bin[i] = len(np_v_diff[this_bin])
            (v_diff_mean_zero,v_diff_sigma[i])=norm.fit(np_v_diff[this_bin])
            v_diff_mean[i]=np.nanmean(np.abs(np_v_diff[this_bin]))
    return dist_array, v_diff_mean, np_dist, num_in_bin

#%% 3d spherical vsf (work in progress)
def VSF_3Dsphere(grtab,d_max,n_bins=50,order=1,savemem=0): #Not working yet. Possible coordinate transformation error
    
    if savemem == 2: #default is float64, but it takes too much memory. So if needed, reduce to float32
        grtab.ra = grtab.ra.astype('float16')
        grtab.dec = grtab.dec.astype('float16')
        grtab.dist = grtab.dist.astype('float16')
        grtab.vlsr = grtab.vlsr.astype('float16')
        grtab.vlsrra = grtab.vlsrra.astype('float16')
        grtab.vlsrdec = grtab.vlsrdec.astype('float16')
    elif savemem == 1:
        grtab.ra = grtab.ra.astype('float32')
        grtab.dec = grtab.dec.astype('float32')
        grtab.dist = grtab.dist.astype('float32')
        grtab.vlsr = grtab.vlsr.astype('float32')
        grtab.vlsrra = grtab.vlsrra.astype('float32')
        grtab.vlsrdec = grtab.vlsrdec.astype('float32')
    
    scale_dist = (3.0857 * 10**16) # scale factor from pc to m, applied to dist
    scale_rv = 1.e3 # scale factor from km/s to m/s, applied to vlsr
    scale_curve = 1./(3.6 * 10**6) * 1./(31536000) * np.pi / 180. # scale factor from mas to degrees, years to seconds, degrees to radians, applied to vdec and vra to convert from mas/yr to rad/s.
    
    #Distance matrix
    # look up distance between 2 points in spherical coordinates, here, theta = ra, phi = pi/2 - dec
    pra_a = np.reshape(grtab['ra'].values, (grtab['ra'].size, 1))
    pra_b = np.reshape(grtab['ra'].values, (1, grtab['ra'].size))
    pdec_a = np.reshape(grtab['dec'].values, (grtab['dec'].size, 1))
    pdec_b = np.reshape(grtab['dec'].values, (1, grtab['dec'].size))
    pdist_a = np.reshape(grtab['dist'].values*scale_dist, (grtab['dist'].size, 1))
    pdist_b = np.reshape(grtab['dist'].values*scale_dist, (1, grtab['dist'].size))
    dist_matrix = np.sqrt(pdist_a**2+pdist_b**2-2*pdist_a*pdist_b*(np.cos(pra_a)*np.cos(pra_b)*np.cos(pdec_b-pdec_a)+np.sin(pra_a)*np.sin(pra_b))) * 1/scale_dist #convert from m to pc
    
    #Velocity difference matrix
    thetadot_a = np.reshape(np.multiply(np.multiply(grtab['vlsrra'].values*scale_curve,grtab['dist'].values*scale_dist),np.sin(np.pi/2-grtab['vlsrdec'].values)), (grtab['ra'].size, 1))
    thetadot_b = np.reshape(np.multiply(np.multiply(grtab['vlsrra'].values*scale_curve,grtab['dist'].values*scale_dist),np.sin(np.pi/2-grtab['vlsrdec'].values)), (1, grtab['ra'].size))
    phidot_a = np.reshape(-1.*grtab['dist'].values*scale_dist*grtab['vlsrdec'].values*scale_curve, (grtab['dec'].size, 1))
    phidot_b = np.reshape(-1.*grtab['dist'].values*scale_dist*grtab['vlsrdec'].values*scale_curve, (1, grtab['dec'].size))
    rdot_a = np.reshape(grtab['vlsr'].values*scale_rv, (grtab['dist'].size, 1))
    rdot_b = np.reshape(grtab['vlsr'].values*scale_rv, (1, grtab['dist'].size))
    v_diff_matrix = np.sqrt(rdot_a**2+rdot_b**2-2*rdot_a*rdot_b*(np.cos(thetadot_a)*np.cos(thetadot_b)*np.cos(phidot_a-phidot_b)+np.sin(thetadot_a)*np.sin(thetadot_b))) * 1/scale_rv #convert from m/s to km/s
    
    v_diff_half = np.ndarray.flatten(np.triu(v_diff_matrix, k=0))
    dist_half = np.ndarray.flatten(np.triu(dist_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0 
    del dist_matrix, v_diff_matrix
    
    good_dist = dist_half>0
    
    np_dist=dist_half[good_dist]
    np_v_diff=v_diff_half[good_dist]
    del good_dist
    
    # dist_array=np.logspace(np.log10(np.min(np_dist)),np.log10(d_max),n_bins)
    dist_array=np.logspace(np.log10(1),np.log10(d_max),n_bins)
    v_diff_mean=np.zeros(n_bins)
    v_diff_sigma=np.zeros(n_bins)
    
    if order == 2:
        np_v_diff_2=np_v_diff**2
        v_diff_mean2 = np.zeros(n_bins)
        v_diff_sigma2 = np.zeros(n_bins)
        for i in range(0,n_bins-1): 
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            (v_diff_mean2[i],v_diff_sigma2[i])=norm.fit(np_v_diff_2[this_bin])
        v_diff_mean = v_diff_mean2
        v_diff_sigma = v_diff_sigma2
    elif order == 3:
        np_v_diff_3=np_v_diff**3
        v_diff_mean3=np.zeros(n_bins)
        v_diff_sigma3=np.zeros(n_bins)
        for i in range(0,n_bins-1):
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            (v_diff_mean3[i],v_diff_sigma3[i])=norm.fit(np.abs(np_v_diff_3[this_bin]))
        v_diff_mean = v_diff_mean3
        v_diff_sigma = v_diff_sigma3
    else:
         for i in range(0,n_bins-1): # start 4 different threads to speed up the fitting process
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            (v_diff_mean_zero,v_diff_sigma[i])=norm.fit(np_v_diff[this_bin])
            v_diff_mean[i]=np.mean(np.abs(np_v_diff[this_bin]))
    return dist_array, v_diff_mean, np_dist, v_diff_sigma

#%% 2d Cartesian vsf, mostly for gas
def VSF_2Dcart(X,Y,Z,los_v,los_v_err=None,n_bins=50,d_max=200,savemem=0):
    """
    Calculate the VSF with only los_v. X, Y, Z can be generated by setting ra, dec, and constant distance or glon, glat, and constant distance. Output is corrected for projection effect assuming isotropic turbulence.

    Parameters
    ----------
    X, Y, Z : TYPE: float or 1d array
        Location of each point in pc.
    los_v : TYPE: float or 1d array
        Line-of-sight velocity in km/s.
    los_v_err : TYPE: float or 1d array
        Line-of-sight velocity error in km/s.
    n_bins : TYPE: integer, optional
        Number of bins to sort the dist_array into. The default is 50.
    d_max : TYPE: float, optional
        Maximum separation of dist_array. The default is 200.

    Returns
    -------
    dist_array : TYPE
        DESCRIPTION.
    v_diff_mean : TYPE
        DESCRIPTION.
    error_mean : TYPE
        DESCRIPTION.
    v_diff_sigma : TYPE
        DESCRIPTION.

    """
    
    if savemem == 2: #default is float64, but it takes too much memory. So if needed, reduce to float32
        X = X.astype('float16')
        Y = Y.astype('float16')
        Z = Z.astype('float16')
        los_v = los_v.astype('float16')
    elif savemem == 1:
        X = X.astype('float32')
        Y = Y.astype('float32')
        Z = Z.astype('float32')
    
    px_a = np.reshape(X, (X.size, 1))
    px_b = np.reshape(X, (1, X.size))
    py_a = np.reshape(Y, (Y.size, 1))
    py_b = np.reshape(Y, (1, Y.size))
    pz_a = np.reshape(Z, (Z.size, 1))
    pz_b = np.reshape(Z, (1, Z.size))
    dist_matrix = np.sqrt((px_a - px_b)**2 + (py_a - py_b)**2 + (pz_a - pz_b)**2)
    
    v_a = np.reshape(los_v, (los_v.size, 1))
    v_b = np.reshape(los_v, (1, los_v.size))
    v_diff_matrix = v_a - v_b
    
    v_diff_half = np.ndarray.flatten(np.triu(v_diff_matrix, k=0))
    dist_half = np.ndarray.flatten(np.triu(dist_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0
    
    good_dist = dist_half>0
    
    np_dist=dist_half[good_dist]
    np_v_diff=v_diff_half[good_dist]
    
    dist_floor=np.floor(np_dist*100)
    unique=np.unique(dist_floor)/100.
    dist_array=np.append(unique[0:400:40]/np.sqrt(3/2),np.logspace(np.log10(unique[400]/np.sqrt(3/2)),np.log10(d_max/np.sqrt(3/2)),n_bins-10))
    # [15:815:50] for Ophi, [0:400:100] for Ori, [0:400:40] for Per, [8:312:38] for Tau
    
    # dist_array=np.logspace(np.log10(1/np.sqrt(3/2)),np.log10(d_max/np.sqrt(3/2)),n_bins)
    
    v_diff_mean=np.zeros(n_bins)
    v_diff_sigma=np.zeros(n_bins)
    
    
    for i in range(0,n_bins-1):
        this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
        (v_diff_mean_zero,v_diff_sigma[i])=norm.fit(np_v_diff[this_bin])
        v_diff_mean[i]=np.nanmean(np.abs(np_v_diff[this_bin])) 

        
    if los_v_err is not None: # if random sampling is done, no error output is needed.
        error_a = np.reshape(los_v_err**2, (los_v_err.size, 1))
        error_b = np.reshape(los_v_err**2, (1, los_v_err.size))
        error_matrix = error_a + error_b
        
        error_half=np.ndarray.flatten(np.triu(error_matrix, k=0))
        np_error_2=error_half[good_dist]
        error_mean=np.zeros(n_bins)
        
        for i in range(0,n_bins-1):
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            error_mean[i]=np.sqrt(np.abs(np.nanmean(np_error_2[this_bin])))
        return dist_array*np.sqrt(3/2), v_diff_mean*np.sqrt(3), error_mean*np.sqrt(3), np_dist
    else:
        return dist_array*np.sqrt(3/2), v_diff_mean*np.sqrt(3), v_diff_sigma, np_dist

#%% VSF calculation for a PPV dataset, mostly similar to the function above but doesn't require 3 positional inputs

def VSF_PPV(resolution,ra,dec,los_v,los_v_err=None,dist=None,n_bins=50,d_max=30,savemem=0):
    """
    Calculate the VSF of a PPV dataset via angular separation. There is an option to convert separation to physical (projected) distance. Output is corrected for projection effect assuming isotropic turbulence.

    Parameters
    ----------
    resolution : TYPE: float
        Density of sampling pixels per degree. E.g. 0.25 if there are 4 points per degree
    ra, dec : TYPE: float or 1d array
        Location of each point in degree. Alternatively galactic longitude-latitude also work here.
    los_v : TYPE: float or 1d array
        Line-of-sight velocity in km/s.
    los_v_err : TYPE: float or 1d array
        Line-of-sight velocity error in km/s.
    dist : TYPE: float, optional
        If provided, separations are converted to projected distance in pc.
    n_bins : TYPE: integer, optional
        Number of bins to sort the dist_array into. The default is 50.
    d_max : TYPE: float, optional
        Maximum angular separation. The default is 30 degrees (about the size of Orion).

    Returns
    -------
    dist_array : TYPE
        DESCRIPTION.
    v_diff_mean : TYPE
        DESCRIPTION.
    error_mean : TYPE
        DESCRIPTION.
    v_diff_sigma : TYPE
        DESCRIPTION.
    """
    
    if savemem == 2: #default is float64, but it takes too much memory. So if needed, reduce to float32
        ra = ra.astype('float16')
        dec = dec.astype('float16')
        los_v = los_v.astype('float16')
    elif savemem == 1:
        ra = ra.astype('float32')
        dec = dec.astype('float32')
    
    pra_a = np.reshape(ra, (ra.size, 1))
    pra_b = np.reshape(ra, (1, ra.size))
    pdec_a = np.reshape(dec, (dec.size, 1))
    pdec_b = np.reshape(dec, (1, dec.size))
    dist_matrix = np.sqrt((pra_a - pra_b)**2 + (pdec_a - pdec_b)**2)
    
    v_a = np.reshape(los_v, (los_v.size, 1))
    v_b = np.reshape(los_v, (1, los_v.size))
    v_diff_matrix = v_a - v_b
    
    v_diff_half = np.ndarray.flatten(np.triu(v_diff_matrix, k=0))
    dist_half = np.ndarray.flatten(np.triu(dist_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0
    
    good_dist = dist_half>0
    
    np_dist=dist_half[good_dist]
    np_v_diff=v_diff_half[good_dist]
    
    dist_array=np.append(np.linspace(resolution,resolution*6,num=6),np.logspace(np.log10(resolution*7),np.log10(d_max),n_bins-6))
    
    v_diff_mean=np.zeros(n_bins)
    v_diff_sigma=np.zeros(n_bins)
    
    
    for i in range(0,n_bins-1):
        this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
        (v_diff_mean_zero,v_diff_sigma[i])=norm.fit(np_v_diff[this_bin])
        v_diff_mean[i]=np.nanmean(np.abs(np_v_diff[this_bin])) 

        
    if los_v_err is not None: # if random sampling is done, no error output is needed.
        error_a = np.reshape(los_v_err**2, (los_v_err.size, 1))
        error_b = np.reshape(los_v_err**2, (1, los_v_err.size))
        error_matrix = error_a + error_b
        
        error_half=np.ndarray.flatten(np.triu(error_matrix, k=0))
        np_error_2=error_half[good_dist]
        error_mean=np.zeros(n_bins)
        
        for i in range(0,n_bins-1):
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            error_mean[i]=np.sqrt(np.abs(np.nanmean(np_error_2[this_bin])))
        if dist is not None:
            dist_array = dist_array * np.pi / 180 * dist * np.sqrt(3/2)
            v_diff_mean = v_diff_mean * np.sqrt(3)
            error_mean = error_mean * np.sqrt(3)
        return dist_array, v_diff_mean, error_mean, np_dist
    else:
        if dist is not None:
            dist_array = dist_array * np.pi / 180 * dist * np.sqrt(3/2)
            v_diff_mean = v_diff_mean * np.sqrt(3)
            error_mean = error_mean * np.sqrt(3)
        return dist_array, v_diff_mean, v_diff_sigma, np_dist
    

#%% Function to compute the confidence interval of mean velocity difference
import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

#%% Function to compute the best-fit line and root mean squared error of the fit
from sklearn.metrics import mean_squared_error
def slopefit (dist_array, v_diff_array, lowbound, highbound):
    xx = dist_array[max(max(np.where(dist_array < lowbound))):min(min(np.where(dist_array > highbound)))]
    yy = v_diff_array[max(max(np.where(dist_array < lowbound))):min(min(np.where(dist_array > highbound)))]
    log_xx = np.log(xx)
    log_yy = np.log(yy)
    curve_fit = np.polyfit(log_xx, log_yy, 1)
    log_y_fit = curve_fit[1] + curve_fit[0]*log_xx
    slope = curve_fit[0]
    
    rmse_1 = mean_squared_error(yy, np.exp(log_y_fit))
    print("slope = ", round(curve_fit[0],3), ", rmse = ", round(rmse_1,3))
    return log_xx, log_y_fit, slope, rmse_1

#%% Function for a single (or double) gaussian feature. Used in fitting spectra
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss2(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

#%% Function to calculate the M_BH from Maithil et al. 2021 prescription for RFeII correction
def Mass_BH(L_5100, fwhm_hb, rfe2): #L_5100 is nu L nu [5100 A] [erg s^-1], fwhm_hb [km s^-1], rfe2 is EW(Fe II)/EW(Hb)
    alpha = 1.65
    beta = 0.45
    gamma = -0.35
    # alpha_err = 0.06
    # beta_err = 0.03
    # gamma_err = 0.08 # Error propagation can be added later
    R_blr = (10.**(alpha + beta * np.log10(L_5100/1e44) + gamma * rfe2)) / 1191.286 #convert from light-day to pc
    mbh = 1 * R_blr * fwhm_hb**2 / (4.3e-3) # unit of G [pc M_sun^-1 (km/s)^2]
    return mbh

#%% Function to calculate the L/L_Edd using scaling relation specified in Richards et al. 2006...?
def LL_Edd(mbh, L_5100): #mbh is black hole mass [M_sun], L_5100 is nu L nu [5100 A] [erg s^-1]
    L_Edd = 1.5e45 * mbh/1e7
    L_bol = 9.26 * L_5100
    LL_Edd = L_bol/L_Edd
    return LL_Edd

#%% Function to calculate the L/L_Edd with bolometric correction from Marconi et al. (2004)
def LL_Edd_BC(mbh, L_5100, BC):
    return 0.657 * BC * L_5100 / 1e44 / (mbh/1e6)

#%% Function to calculate the Bolometric correction from Marconi et al. (2004), adopted from MATLAB code by Ohad Shemmer
def Bolo_Correction(L_5100):
    L_sun = 3.845e33
    p = -7.391
    q = 463.913
    r = 4869.565 - 434.783 * np.log10((L_5100*np.sqrt(5100./4400))/L_sun)
    x = np.linspace(-5,5,10001)
    y = np.zeros((len(L_5100),len(x)))
    L_Bol = np.zeros(len(L_5100))
    BC = np.zeros(len(L_5100))
    t = np.zeros(len(L_5100))
    for i in range(len(L_5100)):
        y[i] = (x**3) + p * (x**2) + q * x + r[i];
        k = 0
        while y[i][k]<0:
            k = k+1
        t[i] = (x[k-1] * y[i][k] - y[i][k-1] * x[k]) / (y[i][k] - y[i][k-1])
        L_Bol[i] = L_sun * (10**(t[i] + 12))
        BC[i] = L_Bol[i]/(L_5100[i]) 
    return BC

#%% Rearrange elements in an rgb_grid to shift the center of a frame. Only works for sky coordinate grid
def shift_frame(grid,lon_shift,lat_shift): #assuming grid is a (lon x lat x color) matrix
    hpixel_shift = int(lon_shift * 4)
    vpixel_shift = int(lat_shift * 4)
    if hpixel_shift >= grid.shape[1]:
        hpixel_shift = hpixel_shift - grid.shape[0]
    if vpixel_shift >= grid.shape[0]:
        vpixel_shift = vpixel_shift - grid.shape[1]
    shift_grid = np.concatenate((grid[:,hpixel_shift:,:],grid[:,:hpixel_shift,:]),axis=1)
    shift_grid = np.concatenate((shift_grid[vpixel_shift:,:,:],shift_grid[:vpixel_shift,:,:]),axis=0)
    return shift_grid

#%% Convert from degrees to hour-minute-second display
def deg2HMS(ra='', dec='', round=False):
    RA, DEC, rs, ds = '', '', '', ''
    if dec:
        if str(dec)[0] == '-':
            ds, dec = '-', abs(dec)
        deg = int(dec)
        decM = abs(int((dec-deg)*60))
        if round:
            decS = int((abs((dec-deg)*60)-decM)*60)
        else:
            decS = (abs((dec-deg)*60)-decM)*60
        DEC = '{0}{1}h{2}m{3}s'.format(ds, deg, decM, decS)
          
    if ra:
        if str(ra)[0] == '-':
            rs, ra = '-', abs(ra)
        raH = int(ra/15)
        raM = int(((ra/15)-raH)*60)
        if round:
            raS = int(((((ra/15)-raH)*60)-raM)*60)
        else:
            raS = ((((ra/15)-raH)*60)-raM)*60
        RA = '{0}{1}h{2}m{3}s'.format(rs, raH, raM, raS)
          
    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC

#%% Function to plot the star VSF in accordance with Ha et al. 2021
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})   
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
def plot_vsf(dist_array,v_diff_mean,error_mean=None,x_bound=[1,300],y_bound=[1,40],Kolmogorov=False,Supersonic=False,linear=False,Larson=True,name=None,legend=None):
    """
    Function to plot the vsf
    Parameters
    ----------
    dist_array : TYPE: 1d array
        Bins of equally-spaced log separations.
    v_diff_mean : TYPE: 1d array
        VSF per value within dist_array.
    error_mean : TYPE: 1d array, optical
        1 sigma error of the VSF. The default is None.
    x_bound : 2-value list, optional
        Plot as xlim. The default is [1,300].
    y_bound : 2-value list, optional
        Plot as ylim. The default is [1,40].
    Kolmogorov : Bool, optional
        Whether to plot Kolmogorov (1/3) power spectrum. The default is False.
    Supersonic : Bool, optional
        Whether to plot supersonic (1/2) power spectrum. The default is False.
    linear: Bool, optional
        Whether to plot linear (1) power spectrum. The default is False.
    Larson : Bool, optional
        Whether to plot Larson's relation (0.38) power spectrum. The default is True.
    name : string, optional
        Name of the plot. The default is None.

    Returns
    -------
    plt.gca()
        Return the plot and its current axis so that futher manipulations can be made.
    """
    plt.clf()
    f = plt.figure(figsize = (9,8), dpi=300)
    ax = f.add_subplot(111)
    plt.loglog(dist_array[:-1],v_diff_mean[:-1],marker="o",linestyle="-",markersize=10,color="C3",linewidth=2,label=legend)
    
    if Kolmogorov == True:
        y_expect=dist_array**(1.0/3)*2.8
        plt.loglog(dist_array,y_expect,linestyle="-.",label="1/3 (Kolmogorov)",color="black",linewidth=3)
    if Supersonic == True:
        y_expect2=dist_array**(1.0/2)*1.1
        plt.loglog(dist_array,y_expect2,label="1/2 (Supersonic)",color="lime",linestyle='--',linewidth=3)
    if linear == True:
        y_expect3=dist_array*0.5
        plt.loglog(dist_array,y_expect3,label="1 (linear)",color="indigo",linestyle='-.',linewidth=3)
    if Larson == True:
        y_larson= 1.1 * dist_array ** 0.38
        plt.loglog(dist_array,y_larson,label='0.38 (Larson\'s law)', color="blue",linewidth=3)
    if error_mean is not None:
        plt.errorbar(dist_array,v_diff_mean,yerr=error_mean,fmt='none',label=None,elinewidth=3,color='coral')


    plt.xlabel("$\ell$ (pc)",size=24)
    plt.ylabel(r"$\langle |\delta \vec{v}| \rangle$ (km s$^{-1}$)",size=24)
    y_locs=np.array([0.5,1,2,5,10,20,50])
    y_labels=np.array([0.5,1,2,5,10,20,50])
    plt.yticks(y_locs, y_labels)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(y_bound[0],y_bound[1])
    plt.xlim(x_bound[0],x_bound[1])
    
    if name != None:
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, pad=.3)
        ax.text(x_bound[0] + 0.2, y_bound[1] - y_bound[1]/4, name, fontsize=24, bbox=props, fontweight='bold')
    
    plt.grid()
    return plt.draw()

#%% Plot VSF of gas with angular separation 

def plot_gasvsf(dist_array,v_diff_mean,error_mean=None,x_bound=[0.2,30],y_bound=[0.1,30],Kolmogorov=False,Supersonic=False,linear=False,Larson=True,name=None,legend=None):
    """
    Function to plot the vsf
    Parameters
    ----------
    dist_array : TYPE: 1d array
        Bins of angular separations.
    v_diff_mean : TYPE: 1d array
        VSF per value within dist_array.
    error_mean : TYPE: 1d array, optical
        1 sigma error of the VSF. The default is None.
    x_bound : 2-value list, optional
        Plot as xlim. The default is [1,300].
    y_bound : 2-value list, optional
        Plot as ylim. The default is [1,40].
    Kolmogorov : Bool, optional
        Whether to plot Kolmogorov (1/3) power spectrum. The default is False.
    Supersonic : Bool, optional
        Whether to plot supersonic (1/2) power spectrum. The default is False.
    linear: Bool, optional
        Whether to plot linear (1) power spectrum. The default is False.
    Larson : Bool, optional
        Whether to plot Larson's relation (0.38) power spectrum. The default is True.
    name : string, optional
        Name of the plot. The default is None.

    Returns
    -------
    plt.gca()
        Return the plot and its current axis so that futher manipulations can be made.
    """
    plt.clf()
    f = plt.figure(figsize = (9,8), dpi=300)
    ax = f.add_subplot(111)
    plt.loglog(dist_array[:-1],v_diff_mean[:-1],marker='.',linestyle='None',markersize=10,color='red',linewidth=2,label=legend)
    
    dist_array_ref = np.linspace(x_bound[0],x_bound[1])
    if Kolmogorov == True:
        y_expect=dist_array_ref**(1.0/3)*4
        plt.loglog(dist_array_ref,y_expect,linestyle="-.",color="black",linewidth=3)
        plt.text(21, 13, '$1/3$', fontsize=22,fontweight='bold')
    if Supersonic == True:
        y_expect2=dist_array_ref**(1.0/2)*4
        plt.loglog(dist_array_ref,y_expect2,color="pink",linestyle='--',linewidth=3)
        plt.text(17, 20, '$1/2$', fontsize=22, fontweight='bold')
    if linear == True:
        y_expect3=dist_array_ref*4
        plt.loglog(dist_array_ref,y_expect3,color="indigo",linestyle='-.',linewidth=3)
        plt.text(17, 20, '$1$', fontsize=22, fontweight='bold')
    if Larson == True:
        y_larson= 1.1 * dist_array_ref ** 0.38
        plt.loglog(dist_array_ref,y_larson,label='Larson\'s law', color="blue",linewidth=3)
    if error_mean is not None:
        plt.fill_between(dist_array[:-1],v_diff_mean[:-1]-error_mean[:-1],v_diff_mean[:-1]+error_mean[:-1],alpha=0.4,color='coral')
        # plt.errorbar(dist_array,v_diff_mean,yerr=error_mean,fmt='none',label=None,elinewidth=3,color='coral')


    plt.xlabel("$\ell$ (°)",size=24)
    plt.ylabel(r"$\langle |\delta \vec{v}| \rangle$ (km s$^{-1}$)",size=24)
    y_locs=np.array([0.1,0.2,0.5,1,2,5,10,20,50])
    y_labels=np.array([0.1,0.2,0.5,1,2,5,10,20,50])
    x_locs=np.array([0.25,1,10])
    x_labels=np.array([0.25,1,10])
    plt.xticks(x_locs, x_labels)
    plt.yticks(y_locs, y_labels)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(y_bound[0],y_bound[1])
    plt.xlim(x_bound[0],x_bound[1])
    
    if name != None:
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, pad=.3)
        ax.text(x_bound[0] + 0.05, y_bound[1] - y_bound[1]/3, name, fontsize=24, bbox=props, fontweight='bold')
    
    plt.grid()
    return plt.draw()
#%% Function to convert barycentric velocity to local standard-of-rest velocity (depreciated, use astropy.SkyCoord for this operation)
def BSRtoLSR(v_bsr,l,b):
    """
    Function to convert barycentric velocity to local standard-of-rest velocity

    Parameters
    ----------
    v : TYPE: float or 1d array
        Barycentric velocity in km/s. 
    l : TYPE: float or 1d array
        Galactic longitude in deg.
    b : TYPE: float or 1d array
        Galactic latitude in deg.

    Returns
    -------
    vlsr : TYPE: float of ndarray (same as input)
           LSR velocity in km/s
    """
    l = l * np.pi / 180
    b = b * np.pi / 180
    vlsr = v_bsr + 9*np.cos(l)*np.cos(b) + 12*np.sin(l)*np.cos(b) + 7*np.sin(b)
    return vlsr

#%% Function to find the closest point in the galactic rotation-corrected velocity profile and output the velocity at that point
def closest_neighbor(gl,gb,ref_frame):
    """
    Function to find the closest point in the galactic rotation-corrected velocity profile and output the velocity at that point.

    Parameters
    ----------
    gl : TYPE
        DESCRIPTION.
    gb : TYPE
        DESCRIPTION.
    ref_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    dist_l = gl - ref_frame['l'].values
    dist_b = gb - ref_frame['b'].values
    dist_2d = np.sqrt(dist_l**2 + dist_b**2)
    k = np.argmin(dist_2d)
    return ref_frame.loc[k]['v']

#%% Function to convert spherical coordinates and velocities to Cartesian
def spherical2cartesian(r,phi,theta,vr,vphi,vtheta):
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    vx = vr * np.cos(phi) * np.sin(theta) - r * np.sin(phi) * np.sin(theta) * vphi + r * np.cos(phi) * np.cos(theta) * vtheta
    vy = vr * np.sin(phi) * np.sin(theta) + r * np.cos(phi) * np.sin(theta) * vphi + r * np.sin(phi) * np.cos(theta) * vtheta
    vz = vr * np.cos(theta) - r * np.sin(theta) * vtheta
    return [x, y, z, vx, vy, vz]
    
#%% Function to match two spatial datasets by ra-dec or equivalences
def points_identical(set1,set2,tolerance):
    """
    Function to match two spatial datasets by ra-dec or equivalences
    Parameters
    ----------
    set1 : TYPE: 2d DataFrame, x-y columns (or equivalence)
        Dataset to be compared
    set2 : TYPE: 2d DataFrame, x-y columns (or equivalence)
        Dataset to be compared
    tolerance : TYPE: float
        The furthest distance allowed for two points to be considered identical

    Returns
    -------
    Index of set2 that is closest spatially to the corresponding index of set1

    """
    RA_1 = np.repeat(np.reshape(set1.iloc[:,0].values,(1,len(set1))),len(set2),axis=0)
    RA_2 = np.repeat(np.reshape(set2.iloc[:,0].values,(len(set2),1)),len(set1),axis=1)
    DEC_1 = np.repeat(np.reshape(set1.iloc[:,1].values,(1,len(set1))),len(set2),axis=0)
    DEC_2 = np.repeat(np.reshape(set2.iloc[:,1].values,(len(set2),1)),len(set1),axis=1)
    sub_RA = RA_1 - RA_2
    sub_DEC = DEC_1 - DEC_2
    dist_matrix = np.sqrt(np.square(sub_RA) + np.square(sub_DEC))
    
    min_dist = dist_matrix.min(axis=0)
    loc_min_dist = dist_matrix.argmin(axis=0)
    
    for i in range(len(min_dist)):
        if min_dist[i] > tolerance:
            loc_min_dist[i] = -1
    
    return loc_min_dist
    

#%% Convert line centers to blueshift

def LC2blueshift(z_sys, LC, lambda_rest):
    """
    Function to convert wavelength of a line center to a velocity offset. Based on Eq 2 of Dix et al. (2020)

    Parameters
    ----------
    z_sys : float or 1d array
        Systemic redshift of the quasar, ideally computed based on the [OIII] line.
    LC : float or 1d array
        Wavelength of the line center of interest.
    lambda_rest : float
        Rest-frame wavelength of the emission line. E.g. 1549 for CIV, 2798 for MgII, 4861 for Hb.

    Returns
    -------
    Returns velocity offset (blueshift) in km/s.

    """
    z_meas = LC / lambda_rest - 1
    return 2.9979e5 * (z_meas - z_sys) / (1 + z_sys)


#%% Calculate the slope of the VSF with uncertainty using y = A * x^B

from scipy.optimize import curve_fit
def fitVSFslope(x_data, y_data, y_err, x_bound):
    """
    Function to calculate the slope of a VSF given associated errors

    Parameters
    ----------
    x_data : 1d array
        x-coordinates of the points.
    y_data : 1d array
        y-coordinates of the points.
    y_err : 1d array
        Errors associated with the y-coordinates.
    x_bound : 2-element list
        Bound of the line fitted on the x-axis

    Returns
    -------
    2-parameter result [popt, perr]
        popt: A and B, based on equation y = A * x^B
        perr: associate uncertainties of A and B

    """
    
    def log_function(x, a, b):
        return a * x**b

    x_fit = x_data[(x_data > x_bound[0]) & (x_data < x_bound[1])]
    y_fit = y_data[(x_data > x_bound[0]) & (x_data < x_bound[1])]
    y_fit_err = y_err[(x_data > x_bound[0]) & (x_data < x_bound[1])]
    
    popt, pcov = curve_fit(log_function, x_fit, y_fit, sigma=y_fit_err, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


#%% Make a movie given a sequence of files (images):
    
import moviepy.editor as mpy
def make_movie(filenames, outname, fps=10, loops=1):
    """
    Function to make a movie given a sequence of image paths

    Parameters
    ----------
    filenames : list of strings
        All paths to the images that make up the movie.
    outname : string
        Path and name of the movie file. Should end with '.mp4'
    fps : int, optional
        The frames-per-second of the movie. The default is 10.
    loops : int, optional
        Indicate if the movie should be loops, and how many times. The default is 1 (no looping).

    Returns
    -------
    None.

    """
    
    image_list = []
    
    for i in range(loops):
        image_list = image_list.append(filenames)
    
    clip = mpy.ImageSequenceClip(image_list,fps=fps)

    return clip.write_videofile(outname)  

#%% Compute luminosity distance

def luminosity_distance(z, H0, WM, WV):
    """
    Function to calculate the luminosity distance. Based on script by Ned Wright and James Schombert.

    Parameters
    ----------
    z : float or 1d array
        Systemic redshift of the quasar.
    H0 : float
        Hubble constant, in km/s.
    WM : float
        Omega_matter of the universe.
    WV : float
        Omega_lambda of the universe.

    Returns
    -------
    float or 1d array
        Returns the luminosity distance of objects in cm

    """
    WR = 0.        # Omega(radiation)
    WK = 0.        # Omega curvaturve = 1-Omega(total)
    c = 299792.458 # velocity of light in km/sec
    DTT = 0.5      # time from z to now in units of 1/H0
    age = 0.5      # age of Universe in units of 1/H0
    zage = 0.1     # age of Universe at redshift z in units of 1/H0
    DCMR = 0.0     # comoving radial distance in units of c/H0
    DA = 0.0       # angular size distance
    DL = 0.0       # luminosity distance
    DL_Mpc = 0.0
    a = 1.0        # 1/(1+z), the scale factor of the Universe
    az = 0.5       # 1/(1+z(object))

    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    az = 1.0/(1+1.0*z)
    age = 0.
    n=1000         # number of points in integrals
    for i in range(n):
      a = az*(i+0.5)/n
      adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
      age = age + 1./adot

    zage = az*age/n
    DTT = 0.0
    DCMR = 0.0

  # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
      a = az+(1-az)*(i+0.5)/n
      adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
      DTT = DTT + 1./adot
      DCMR = DCMR + 1./(a*adot)

    DTT = (1.-az)*DTT/n
    DCMR = (1.-az)*DCMR/n
    age = DTT+zage
    
  # tangential comoving distance
    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if x > 0.1:
      if WK > 0:
        ratio =  0.5*(np.exp(x)-np.exp(-x))/x 
      else:
        ratio = np.sin(x)/x
    else:
      y = x*x
      if WK < 0: y = -y
      ratio = 1. + y/6. + y*y/120.
    DCMT = ratio*DCMR
    DA = az*DCMT
    DL = DA/(az*az)
    DL_Mpc = (c/H0)*DL
    
    return DL_Mpc * 3.08568e24
    
#%% Calculate the rest-frame F_lambda

def cal_L_lambda_rest(nu, F_nu, z_sys):
    """
    Function to calculate the rest-frame monochromatic flux density F_lambda, given an SED 

    Parameters
    ----------
    nu : float or 1d array
        Frequency.
    F_nu : float or 1d array
        Observed-frame Flux density F_nu.
    D_L : float or 1d array
        Luminosity distance.
    z_sys : float
        Systemic redshift of the quasar.

    Returns
    -------
    float or 1d array
        Returns rest-frame monochromatic flux density F_lambda in erg s^-1 cm^-1.

    """
    c = 2.9979e10
    D_L = luminosity_distance(z_sys, 70, 0.3, 0.7)
    F_lambda_obs = nu**2 * F_nu / c
    return F_lambda_obs * D_L**2 * (1+z_sys)

#%% Calculate hydrostatic equilibrium

import glob
import yt
def output_files(directory):
    """
    Function to output the name of all hdf5 files in a directory

    Parameters
    ----------
    directory : str
        Parent directory of simulation output.

    Returns
    -------
    lsdir : list of str
        List a series of outputs, sorted by their names.

    """
    lsdir = glob.glob(directory+'/*.athdf')
    lsdir.sort()
    return lsdir

def hash_vr_t(r0,lsout,units_override):
    """
    Function to retrieve radial velocity at r0 in every time step.

    Parameters
    ----------
    r0 : float
        Distance from cener, in Mpc.
    lsout : list
        List of output files.
    units_override : dict
        Specify the elementary units of the simulation.

    Returns
    -------
    v_r : 1d array
        Array of radial velocities at r0, as a function of time, in Mpc Myr^-1.
    t : 1d array
        Array of simulation time, in Myr

    """
    
    #load in first output to get radial profile, avoid doing this multiple times
    ds0 = yt.load(lsout[0],units_override=units_override)
    sp0 = ds0.sphere("c",(r0+0.02,"Mpc")) #reduce computing time by only making a sphere barely larger than r0
    rp0 = yt.create_profile(
        sp0,
        ("index", "radius"),
        ("gas", "radial_velocity"),
        units={("index", "radius"): "Mpc"},
        logs={("index", "radius"): False},
        )
    r = rp0.x.value
    dis_list = np.abs(r - r0)
    index_r0 = np.argmin(dis_list)
    
    t = np.zeros(len(lsout))
    v_r = np.zeros(len(lsout))
    k = 0
    
    for n in lsout:
        
        ds = yt.load(n,units_override=units_override)
        t[k] = float(ds.current_time.to("Myr"))
        
        sp = ds.sphere("c",(r0+0.02,"Mpc")) #reduce computing time by only making a sphere barely larger than r0
        rp = yt.create_profile(
            sp,
            ("index", "radius"),
            ("gas", "radial_velocity"),
            units={("index", "radius"): "Mpc"},
            logs={("index", "radius"): False},
            )
        v_r[k] = rp[("gas", "radial_velocity")].in_units("Mpc/Myr").value[index_r0]
        
        k +=1
    
    return v_r,t


def hash_press_rho_vr_r_t(lsout,units_override):
    """
    Function to retrieve gas density and radial velocity as a function of both r and t.

    Parameters
    ----------
    lsout : list
        List of output files.
    units_override : dict
        Specify the elementary units of the simulation.

    Returns
    -------
    press : 2d array
        Array of pressure at each r and t, in Msun Mpc^-1 Myr^-2.
    rho : 2d array
        Array of densities at each r and t, in Msun Mpc^-3.
    vr : 2d array
        Array of radial velocities at each r and t, in Mpc Myr^-1.
    t : 1d array
        Array of simulation time, in Myr
    r : 1d array
        Array of radial distances, in Mpc

    """
    
    r = []
    t = np.zeros(len(lsout))
    vr = []
    rho = []
    press = []
    k = 0

    for n in lsout:
        
        ds = yt.load(n,units_override=units_override)
        t[k] = float(ds.current_time.to("Myr"))
        
        sp = ds.sphere("c",(float(ds.r[('index','x')].to('Mpc').max()),"Mpc")) #reduce computing time by only making a sphere barely larger than r0
        rp = yt.create_profile(
            sp,
            ("index", "radius"),
            [("gas", "radial_velocity"),("gas","density"),("gas","pressure")],
            units={("index", "radius"): "Mpc"},
            logs={("index", "radius"): False},
            )
        vr.append(rp[("gas", "radial_velocity")].in_units("Mpc/Myr").value)
        r.append(rp.x.value)
        rho.append(rp[("gas", "density")].in_units("Msun/Mpc**3").value)
        press.append(rp[("gas", "pressure")].in_units("Msun/(Mpc*Myr**2)").value)
        k +=1
        
    vr = np.vstack(vr)
    rho = np.vstack(rho)
    r = np.vstack(r)
    press = np.vstack(press)
    
    return press, rho, vr, t, r[0,:]
    

def compute_dP_dr(P,r,r0):
    """
    Function to compute pressure gradient at some radial distance r0 (in kpc), using the secant method

    Parameters
    ----------
    P : 1d array
        Pressure array, in Msun Myr^-2 Mpc^-1.
    r : 1d array
        Radial distances array, in Mpc.
    r0 : float
        Distance from cener, in Mpc.

    Returns
    -------
    dP_dr : float
        Approximate dP/dr (r0), in Msun Myr^-2 Mpc^-2.

    """
    
    #find the nearest element to r0
    dis_list = np.abs(r - r0)
    index_r0 = np.argmin(dis_list)
    
    #Compute dP/dr using the secant method
    dP_dr = (P[index_r0+1] - P[index_r0-1]) / (r[index_r0+1] - r[index_r0-1])
    
    return dP_dr

def compute_g(v_r,t,t0):
    """
    Function to compute gravitational acceleration from radial velocity

    Parameters
    ----------
    v_r : 1d array
        Radial velocity at r0, as a function of t, in Mpc Myr^-1.
    t : 1d array
        Time array, in Myr.
    t0 : float
        Time of acceleration calculation, in Myr.

    Returns
    -------
    g : float
        Approximate gravitational acceleration g(r0,t0), in Mpc Myr^-2.

    """
    #find the nearest element to t0
    tim_list = np.abs(t - t0)
    index_t0 = np.argmin(tim_list)
    
    #Compute g using the secant method
    g = (v_r[index_t0+1] - v_r[index_t0-1]) / (t[index_t0+1] - t[index_t0-1])
    
    return g
    

def hydro_equil(directory,r0,t0,units_override):
    """
    Function to compute the difference between dP/dr and rho*g at some specified radial distance r0 and time t0

    Parameters
    ----------
    directory : str
        Parent directory of simulation output.
    r0 : float
        Distance from cener, in Mpc.
    t0 : float
        Time of simulation, in Myr.
    units_override : dict
        Specify the elementary units of the simulation.

    Returns
    -------
    dP_dr_rhog : float
        If system is in hydrodynamical equilibrium, this difference is zero.

    """
    
    lsout = output_files(directory)
    
    press,rho,v_r,t,r = hash_press_rho_vr_r_t(lsout, units_override)
    
    #find the nearest element to t0
    tim_list = np.abs(t - t0)
    index_t0 = np.argmin(tim_list)
    
    #find the nearest element to r0
    dis_list = np.abs(r - r0)
    index_r0 = np.argmin(dis_list)
    
    dP_dr = compute_dP_dr(press[index_t0,:], r, r0)
    
    g = compute_g(v_r[:,index_r0], t, t0)
    
    dP_dr_rhog = dP_dr - rho[index_t0,index_r0] * g
    
    return dP_dr_rhog