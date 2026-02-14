#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:14:51 2022

Script to generate random stars (or points) with associated error, then compute the VSF and plot it.

@author: Trung Ha
"""
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from numba import njit, prange

import warnings
warnings.filterwarnings("ignore")

def rand_sample(sample : pd.DataFrame, size : int, directory : str):
    """
    Function to generate a number of realizations of points (typically stars) based on a Gaussian distribution, then save each realization as a csv file at the designated path

    Parameters
    ----------
    sample : Pandas dataframe
        Any dataframe, must have these columns: 'ra','dec','dist','dist_err','vlsr','vlsrra','vlsrdec','vlsr_err','vlsrra_err','vlsrdec_err'. 'ra' and 'dec' are assumed as certain, so no error parameter is created.
    size : int
        The number of realizations you want to generate.
    directory : str
        Path to the saved folder.

    Returns
    -------
    None.

    """
    np.random.seed(2) #seed to ensure replicable generated dataset
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


from scipy.stats import norm
# Function to calculate any order of 3D VSFs up to 3rd
def VSF_3Dcart(X : np.ndarray,Y : np.ndarray,Z : np.ndarray,vx : np.ndarray,vy : np.ndarray,vz : np.ndarray,max_distance : float,n_bins=50,order=1,savemem=0):
    """
    Calculate the VSF given 3D velocity and 3D position.

    Parameters
    ----------
    X,Y,Z,vx,vy,vz : numpy 1d array
        x,y,z-positions, x,y,z-velocities.
    max_distance : float
        The maximum separation in pc.
    n_bins : int, optional
        Number of bins. The default is 50.
    order : int between 1 and 3, optional
        The order of the VSF to be calculated. The default is 1.
    savemem : int, either 0 (float64), 1 (float32), or 2 (float16), optional
        Indicate whether datatype should be reduced to save memory. The default is 0.

    Returns
    -------
    dist_array : 1d array
        The array of separations in pc.
    v_diff_mean : 1d array
        The array of VSF in km/s.
    np_dist : nxn matrix
        The distance matrix between all points.
    num_in_bin : int 1d array
        Number of pairs within each bin.

    """
    
    if savemem == 2: #default is float64, but it takes too much memory. So if needed, reduce to float32. Float16 is generally problematic, so avoid using if possible
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
    del dist_matrix, v_diff_matrix #reduce some memory by deleting big matrices
    
    good_dist = dist_half>0
    
    np_dist=dist_half[good_dist]
    np_v_diff=v_diff_half[good_dist]
    del good_dist
    
    # dist_array=np.logspace(np.log10(np.min(np_dist)),np.log10(max_distance),n_bins)
    dist_array=np.logspace(np.log10(1),np.log10(max_distance),n_bins)
    v_diff_mean=np.zeros(n_bins)
    v_diff_sigma=np.zeros(n_bins)
    num_in_bin = np.zeros(n_bins)
    
    if order == 2: #2nd order VSF
        np_v_diff_2=np_v_diff**2
        v_diff_mean2 = np.zeros(n_bins)
        v_diff_sigma2 = np.zeros(n_bins)
        for i in range(0,n_bins-1): 
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            (v_diff_mean2[i],v_diff_sigma2[i])=norm.fit(np_v_diff_2[this_bin])
        v_diff_mean = v_diff_mean2
        v_diff_sigma = v_diff_sigma2
    elif order == 3: #3rd order VSF
        np_v_diff_3=np_v_diff**3
        v_diff_mean3=np.zeros(n_bins)
        v_diff_sigma3=np.zeros(n_bins)
        for i in range(0,n_bins-1):
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            (v_diff_mean3[i],v_diff_sigma3[i])=norm.fit(np.abs(np_v_diff_3[this_bin]))
        v_diff_mean = v_diff_mean3
        v_diff_sigma = v_diff_sigma3
    else: #1st order VSF
         for i in range(0,n_bins-1):
            this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
            num_in_bin[i] = len(np_v_diff[this_bin])
            (v_diff_mean_zero,v_diff_sigma[i])=norm.fit(np_v_diff[this_bin])
            v_diff_mean[i]=np.nanmean(np.abs(np_v_diff[this_bin]))
    return dist_array, v_diff_mean


def vsf_from_sample(m,n_bins,max_distance,sample_size,file_loc):
    """
    Function to generate VSFs from individual realization

    Parameters
    ----------
    m : int
        Index of the realization.
    n_bins : int
        Number of bins.
    max_distance : float
        Maximum separation.
    sample_size : int
        Number of realization.
    file_loc : str
        Path to folder containing all realizations.

    Returns
    -------
    dist_array : 1d array
        The array of separations in pc.
    v_diff_mean : 1d array
        The array of VSF in km/s.
    num_in_bin : int 1d array
        Number of pairs within each bin.

    """
    i = int(m)
    tableq = pd.DataFrame()
    for k in range(len(file_loc)):
        table_read = pd.read_csv(file_loc[k]+'/generated_'+str(i)+'.csv',index_col=0)
        tableq = tableq.append(table_read,ignore_index=True)
     
    if i == 1:
        print("Number of stars : ",len(tableq))
    tableq = tableq.drop([np.random.randint(0,len(tableq))]) #drop 1 random element in each generated realization

    celes_pos = SkyCoord(ra=tableq.ra.values*u.deg, dec=tableq.dec.values*u.deg, distance=tableq.dist.values*u.pc, radial_velocity=tableq.vlsr.values*u.km/u.s, pm_ra_cosdec=tableq.vlsrra.values*u.mas/u.yr, pm_dec=tableq.vlsrdec.values*u.mas/u.yr, frame='icrs')
    galac_pos = celes_pos.galactic #convert to galatocentric frame
    
    galac_pos.representation_type = 'cartesian' #convert to Cartesian
    
    tableq['x_axis'] = np.asarray(galac_pos.u.data)
    tableq['y_axis'] = np.asarray(galac_pos.v.data)
    tableq['z_axis'] = np.asarray(galac_pos.w.data)
    tableq['v_x'] = np.asarray(galac_pos.U.data)
    tableq['v_y'] = np.asarray(galac_pos.V.data)
    tableq['v_z'] = np.asarray(galac_pos.W.data)
    
    # An additional criteria here to separate the dense cluster from the loose cluster
    # galac_pos.representation_type = 'spherical'
    # tableq['b'] = np.asarray(galac_pos.b.data)
    # tableq = tableq[tableq['b'] < 17.5]
    
        
    dist_array, v_diff_mean = VSF_3D(tableq.x_axis.values,tableq.y_axis.values,tableq.z_axis.values,tableq.v_x.values,tableq.v_y.values,tableq.v_z.values,max_distance=max_distance,n_bins=n_bins,order=1)
    return dist_array,v_diff_mean

@njit(parallel=True)
def VSF_3D(X : np.ndarray, Y : np.ndarray, Z : np.ndarray, vx : np.ndarray, vy : np.ndarray, vz : np.ndarray, min_distance=None, max_distance=None, n_bins=50, order=1):
    """Compute first-order velocity structure function (VSF) in 3D, with jit and parallelization

    Args:
        X (np.ndarray): x-coordinates of the data points (pc)
        Y (np.ndarray): y-coordinates of the data points (pc)
        Z (np.ndarray): z-coordinates of the data points (pc)
        vx (np.ndarray): x-components of the velocity vectors (km/s)
        vy (np.ndarray): y-components of the velocity vectors (km/s)
        vz (np.ndarray): z-components of the velocity vectors (km/s)
        max_distance (float, optional): starting distance (pc) in the bins. If None, function assumes 1 pc. Defaults to None.
        max_distance (float, optional): maximum distance (pc) in the bins. If None, function assumes cubic data and take the longest diagonal distance. Defaults to None.
        n_bins (int, optional): number of distance bins. Defaults to 50.
        order (int, optional): order of the VSF. Defaults to 1.

    Returns:
        (np.ndarray, np.ndarray): 1D arrays of the distance bins (pc) and the VSF (km/s) per bin
    """
    # find the maximum distance between any two points in the data
    # assuming the data is roughly of cubic shape (neccessary for the quick computation of max distance)
    if max_distance is None:
        max_distance = np.sqrt((X.max()-X.min())**2 + (Y.max()-Y.min())**2 + (Z.max()-Z.min())**2)

    if min_distance is None:
        min_distance = 1.

    # if order == 1:
    #     print("Calculating 1st order VSF")
    # elif order == 2:
    #     print("Calculating 2nd order VSF")
    # else:
    #     print("Order not valid, defaulting to 1st order VSF")
    #     order = 1
    if (order != 1) or (order != 2):
        order = 1

    # create bins of equal size in log space
    bins = 10.**np.linspace(np.log10(min_distance), np.log10(max_distance), n_bins)
    squared_bins = bins**2

    vsf_per_bin = np.zeros(n_bins-1)

    # loop through bins
    for this_bin_index in range(len(squared_bins)-1):
        # if (this_bin_index+1) % 20 == 0:
        #     print(f"bin {this_bin_index+1} of {len(squared_bins)-1} : START")
        # for each point in the data, find the distance to all other points, then choose only the distances that are in the same bin
        weights = np.zeros(len(X))
        mean_velocity_differences = np.zeros(len(X))

        for point_a in prange(len(X)):
            squared_distance_to_point_a = (X[point_a]-X)**2 + (Y[point_a]-Y)**2 + (Z[point_a]-Z)**2
            elements_in_this_bin = np.full(len(squared_distance_to_point_a), False)
            elements_in_this_bin[(squared_bins[this_bin_index] < squared_distance_to_point_a) & (squared_distance_to_point_a <= squared_bins[this_bin_index+1])] = True
            elements_in_this_bin[:point_a] = False # don't calculate the same point twice

            squared_velocity_difference_to_point_a = (vx[point_a]-vx[elements_in_this_bin])**2 + (vy[point_a]-vy[elements_in_this_bin])**2 + (vz[point_a]-vz[elements_in_this_bin])**2

            # calculate the mean of the velocity differences
            if order == 1:
                mean_velocity_differences[point_a] = np.mean(np.sqrt(squared_velocity_difference_to_point_a))
            else:
                mean_velocity_differences[point_a] = np.mean(squared_velocity_difference_to_point_a)
            weights[point_a] = len(squared_velocity_difference_to_point_a) # the number of points in the distance bin is the weight for the mean calculation later
        
        # if (this_bin_index+1) % 20 == 0:
        #     print(f"bin {this_bin_index+1} of {len(squared_bins)-1} : END")

        # calculate the mean of the velocity differences in this bin
        mean_velocity_differences[weights == 0] = 0. # set the mean to 0 if there are no points in the bin

        if np.max(weights) == 0: # if there are no points in the bin, set the VSF to 0
            vsf_per_bin[this_bin_index] = 0.
        else:
            vsf_per_bin[this_bin_index] = np.average(mean_velocity_differences, weights=weights)

    return bins, vsf_per_bin

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "stixgeneral",
    "font.sans-serif": ["Helvetica"]})
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
        Return the plot and its current axis so that further manipulations can be made.
    """
    plt.clf()
    f = plt.figure(figsize = (9,8), dpi=300)
    ax = f.add_subplot(111)

    plt.loglog(dist_array,v_diff_mean,marker="none",linestyle="-",markersize=10,color="C3",linewidth=2,label=legend,zorder=10)
    
    if Kolmogorov == True:
        y_expect=dist_array**(1.0/3)*2.8
        plt.loglog(dist_array,y_expect,linestyle="-.",label="1/3 (Kolmogorov)",color="black",linewidth=3,zorder=2)
    if Supersonic == True:
        y_expect2=dist_array**(1.0/2)*1.1
        plt.loglog(dist_array,y_expect2,label="1/2 (Supersonic)",color="lime",linestyle='--',linewidth=3,zorder=1)
    if linear == True:
        y_expect3=dist_array*0.5
        plt.loglog(dist_array,y_expect3,label="1 (linear)",color="indigo",linestyle='-.',linewidth=3,zorder=0)
    if Larson == True:
        y_larson= 1.1 * dist_array ** 0.38
        plt.loglog(dist_array,y_larson,label='0.38 (Larson\'s law)', color="blue",linewidth=3,zorder=3)
    if error_mean is not None:
        plt.errorbar(dist_array,v_diff_mean,yerr=error_mean,fmt='none',label=None,elinewidth=3,color='coral',zorder=5)

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


from functools import partial
import matplotlib.pyplot as plt
import multiprocessing
def parallel_runs (m, num_thread):
    """
    Function to parallelize the random sampling VSF calculation. You can run on one thread by modifying the code, but it might take a while with > 1000 points, 1000 realizations.

    Parameters
    ----------
    m : list of int
        A list from 0 to # of realizations.
    num_thread : int
        Number of threads being used.

    Returns
    -------
    result_parallel : 3-parameter list
        Output distance array, VSF, and number of pairs in each bin.

    """
    a_pool = multiprocessing.Pool(processes=num_thread) #parallelize the for loop.
    prod_m = partial(vsf_from_sample, n_bins=n_bins,max_distance=max_distance,sample_size=sample_size,file_loc=file_loc)
    result_parallel = a_pool.map(prod_m,m)
    return result_parallel

## Calculate the VSF and errors

n_bins = 40 #number of bins in the VSF calculation
max_distance = 200 # maximum separation in pc
cloud = r'Orion, no ONC' # naming of the cloud, for plotting
sample_size = 1000 #number of realizations
num_thread = 8 #number of threads used
file_loc = ['/Users/tvh0021/OneDrive - UNT System/RESEARCH_DOCS/VSF_Turbulence/MilkyWay/Gaia_allsky/LambdaOri/','/Users/tvh0021/OneDrive - UNT System/RESEARCH_DOCS/VSF_Turbulence/MilkyWay/Gaia_allsky/OrionB/','/Users/tvh0021/OneDrive - UNT System/RESEARCH_DOCS/VSF_Turbulence/MilkyWay/Gaia_allsky/OrionC/','/Users/tvh0021/OneDrive - UNT System/RESEARCH_DOCS/VSF_Turbulence/MilkyWay/Gaia_allsky/OrionD/','/Users/tvh0021/OneDrive - UNT System/RESEARCH_DOCS/VSF_Turbulence/MilkyWay/Gaia_allsky/LambdaOri/'] #put the path here, inside the square bracket. If there are multiple files, separate them with commas

m = list(range(0,sample_size))

vsf = np.zeros((sample_size,n_bins-1)) #1st order VSF will be saved here
num_in_bin = np.zeros((sample_size,n_bins-1)) # number of pairs in each bin will be saved here

if __name__ == '__main__':
    mapout = parallel_runs(m,num_thread) #run the program in parallel
    
    dist_array = mapout[sample_size-1][0]
    for i in range(0,sample_size):
        vsf[i,:] = mapout[i][1]
        # num_in_bin[i,:] = mapout[i][2]
    
    v_diff_mean = np.zeros(n_bins)
    error_mean = np.zeros(n_bins)
    # num_mean = np.zeros(n_bins)
    
    for k in range (0,n_bins-1):
        v_diff_mean[k] = np.nanmean(vsf[:,k])
        error_mean[k] = np.nanstd(vsf[:,k])
        # num_mean[k] = np.mean(num_in_bin[:,k])
    
    plot_vsf(dist_array,v_diff_mean,error_mean,x_bound=[0.9,200],y_bound=[1,12],name=cloud,Kolmogorov=False,Supersonic=False)
    plt.savefig('vsf_'+cloud+'.png',dpi=300)
    # plt.show()