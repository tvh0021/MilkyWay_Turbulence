#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:00:10 2021

Script to compute the velocity structure function of the Taurus star-forming region using APOGEE DR17 and Gaia EDR3 data.

@author: trungha
"""
# %% Defining functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

import warnings

warnings.filterwarnings("ignore")


# function to convert ra-dec coordinate and velocity to cartesian coordinate
def radec2cartesian(ra, dec, dist, rv, vra, vdec):
    scale_dist = 3.0857 * 10**13  # scale factor from pc to km, applied to dist
    scale_vel = (
        1.0 / (3.6 * 10**6) * 1.0 / (3.1536 * 10**7) * np.pi / 180.0
    )  # scale factor from mas to degrees, years to seconds, degrees to radians, applied to vdec and vra to convert from mas/yr to rad/s.
    x = dist * np.cos(dec) * np.cos(ra)
    y = dist * np.cos(dec) * np.sin(ra)
    z = dist * np.sin(dec)
    v_x = (
        rv * np.cos(dec) * np.cos(ra)
        - dist * scale_dist * np.sin(dec) * np.cos(ra) * vdec * scale_vel
        - dist * scale_dist * np.cos(dec) * np.sin(ra) * vra * scale_vel
    )
    v_y = (
        rv * np.cos(dec) * np.sin(ra)
        - dist * scale_dist * np.sin(dec) * np.sin(ra) * vdec * scale_vel
        + dist * scale_dist * np.cos(dec) * np.cos(ra) * vra * scale_vel
    )
    v_z = rv * np.sin(dec) + dist * scale_dist * np.cos(dec) * vdec * scale_vel
    return x, y, z, v_x, v_y, v_z


import scipy as scipy
from scipy.stats import norm


# function to calculate the velocity structure function of the 3d dataset
def VSF_3Dcartesian(X, Y, Z, vx, vy, vz, n_bins, d_max):

    # Velocity difference matrix
    vel_xa = np.reshape(vx, (vx.size, 1))
    vel_xb = np.reshape(vx, (1, vx.size))
    vel_ya = np.reshape(vy, (vy.size, 1))
    vel_yb = np.reshape(vy, (1, vy.size))
    vel_za = np.reshape(vz, (vz.size, 1))
    vel_zb = np.reshape(vz, (1, vz.size))
    v_diff_matrix = np.sqrt(
        (vel_xa - vel_xb) ** 2 + (vel_ya - vel_yb) ** 2 + (vel_za - vel_zb) ** 2
    )

    # distance matrix
    px_a = np.reshape(X, (X.size, 1))
    px_b = np.reshape(X, (1, X.size))
    py_a = np.reshape(Y, (Y.size, 1))
    py_b = np.reshape(Y, (1, Y.size))
    pz_a = np.reshape(Z, (Z.size, 1))
    pz_b = np.reshape(Z, (1, Z.size))
    dist_matrix = np.sqrt((px_a - px_b) ** 2 + (py_a - py_b) ** 2 + (pz_a - pz_b) ** 2)

    v_diff_half = np.ndarray.flatten(np.triu(v_diff_matrix, k=0))
    dist_half = np.ndarray.flatten(
        np.triu(dist_matrix, k=0)
    )  # this is still a 2D matrix, with the lower half values all set to 0

    good_dist = dist_half > 0

    np_dist = dist_half[good_dist]
    np_v_diff = v_diff_half[good_dist]
    np_v_diff_2 = np_v_diff**2
    np_v_diff_3 = np_v_diff**3

    # d_max=170
    # n_bins=40

    # dist_floor=np.floor(np_dist*100)
    # unique=np.unique(dist_floor)/100.
    dist_array = np.logspace(np.log10(0.4), np.log10(d_max), n_bins)
    # dist_array=np.append(unique[0:15],np.logspace(np.log10(unique[15]),np.log10(d_max),n_bins-15))
    v_diff_mean = np.zeros(n_bins)
    v_diff_sigma = np.zeros(n_bins)
    v_diff_mean2 = np.zeros(n_bins)
    v_diff_sigma2 = np.zeros(n_bins)
    v_diff_mean3 = np.zeros(n_bins)
    v_diff_sigma3 = np.zeros(n_bins)

    for i in range(0, n_bins - 1):
        this_bin = (np_dist >= dist_array[i]) & (np_dist < dist_array[i + 1])
        (v_diff_mean_zero, v_diff_sigma[i]) = scipy.stats.norm.fit(np_v_diff[this_bin])
        v_diff_mean[i] = np.mean(np.abs(np_v_diff[this_bin]))
        (v_diff_mean2[i], v_diff_sigma2[i]) = scipy.stats.norm.fit(
            np_v_diff_2[this_bin]
        )
        (v_diff_mean3[i], v_diff_sigma3[i]) = scipy.stats.norm.fit(
            np.abs(np_v_diff_3[this_bin])
        )
    return (
        dist_array,
        v_diff_mean,
        v_diff_matrix,
        dist_matrix,
        np_dist,
        v_diff_mean2,
        v_diff_mean3,
        v_diff_sigma,
    )


# Function to generate a random sample based on the measured data and its uncertainties


def rand_sample(sample, size):
    np.random.seed(2)  # seed to ensure replicable generated dataset
    # n_1_sample = sample.drop([np.random.randint(1,len(sample))])
    n_1_sample = sample.reset_index()  # just need to do this to avoid indexing conflict

    Pool = np.zeros((len(n_1_sample), 6, size))  # 3d np array that saves everything

    for i in range(
        0, len(n_1_sample)
    ):  # each iteration take the measurement and error of 1 star to generate "size" number of gaussian random number of the listed properties
        generate_rv = np.random.normal(n_1_sample.rv[i], n_1_sample.rv_err[i], size)
        generate_vra = np.random.normal(n_1_sample.vra[i], n_1_sample.vra_err[i], size)
        generate_vdec = np.random.normal(
            n_1_sample.vdec[i], n_1_sample.vdec_err[i], size
        )
        generate_dist = np.random.normal(
            n_1_sample.dist[i], n_1_sample.dist_err[i], size
        )
        for j in range(
            0, size
        ):  # each iteration assign the previously generated gaussian distribution into separate pool
            Pool[i, 0, j] = generate_dist[j]
            Pool[i, 1, j] = generate_rv[j]
            Pool[i, 2, j] = generate_vra[j]
            Pool[i, 3, j] = generate_vdec[j]
            Pool[i, 4, j] = n_1_sample.ra[i]
            Pool[i, 5, j] = n_1_sample.dec[i]
        print("Star " + str(i + 1) + " generated.")

    for k in range(0, size):
        frame = pd.DataFrame(
            Pool[:, :, k], columns=("dist", "rv", "vra", "vdec", "ra", "dec")
        )  # each of these "frame" is a set of randomly generated set of stars based on the normal distribution of observed stars in the set
        frame.to_csv("/home/trungha/Downloads/Taurus1000/generated_" + str(k) + ".csv")

    print("Done, files saved.")
    return


# %% Import Taurus catalog and eliminate undesirable objects
# UPDATE 01/14/2021: Update apogee_taurus with APOGEE DR17 and change proper motion to the lsr frame
from astropy.io import fits

fits_data = fits.open("Taurus/apogee_taurus_dr17.fits")
data = fits_data[1].data

ra = data["RAJ2000"].byteswap().newbyteorder()
dec = data["DEJ2000"].byteswap().newbyteorder()
sb2 = data["SB2"]
slope = data["Slope"].byteswap().newbyteorder()
chi2 = data["chi2"].byteswap().newbyteorder()
rv = np.float64(data["VHELIO_AVG"].byteswap().newbyteorder())
rv_err_s = np.float64(data["VSCATTER"].byteswap().newbyteorder())
rv_err = np.float64(data["VERR"].byteswap().newbyteorder())
parallax = np.float64(data["GAIAEDR3_PARALLAX"].byteswap().newbyteorder())
parallax_err = np.float64(data["GAIAEDR3_PARALLAX_ERROR"].byteswap().newbyteorder())
vra = np.float64(data["GAIAEDR3_PMRA"].byteswap().newbyteorder())
vra_err = np.float64(data["GAIAEDR3_PMRA_ERROR"].byteswap().newbyteorder())
vdec = np.float64(data["GAIAEDR3_PMDEC"].byteswap().newbyteorder())
vdec_err = np.float64(data["GAIAEDR3_PMDEC_ERROR"].byteswap().newbyteorder())


pull = {
    "ra": ra,
    "dec": dec,
    "rv": rv,
    "vra": vra,
    "vdec": vdec,
    "rv_err_s": rv_err_s,
    "rv_err": rv_err,
    "vlsrra_err": vra_err,
    "vlsrdec_err": vdec_err,
    "par": parallax,
    "par_err": parallax_err,
    "slope": slope,
    "chi2": chi2,
    "sb2": sb2,
}

apogee_taurus = pd.DataFrame(data=pull)

apogee_taurus["vlsr_err"] = 0.0
# sometimes only 1 epoch v_err is available (verr), so if vscatter = 0, use that instead
for i in range(len(apogee_taurus)):
    if apogee_taurus["rv_err_s"][i] == 0:
        apogee_taurus["vlsr_err"][i] = apogee_taurus["rv_err"][i]
    else:
        apogee_taurus["vlsr_err"][i] = apogee_taurus["rv_err_s"][i]

# remove all objects with sb2 = 2, slope > 4, chi2 < 16, 5 < rv < 25
apogee_taurus = apogee_taurus[apogee_taurus.sb2.values == 1]
apogee_taurus["chi2"] = apogee_taurus["chi2"].fillna(0)
apogee_taurus["slope"] = apogee_taurus["slope"].fillna(0)
apogee_taurus = apogee_taurus[apogee_taurus.chi2.values < 16]
apogee_taurus = apogee_taurus[apogee_taurus.slope.values < 4]
apogee_taurus = apogee_taurus.dropna()
apogee_taurus = apogee_taurus[apogee_taurus.rv.values < 25]
apogee_taurus = apogee_taurus[apogee_taurus.rv.values > 5]

apogee_taurus["dist"] = 1.0 / (
    apogee_taurus["par"] / 1000.0
)  # convert parralax to distance
apogee_taurus["dist_err"] = (
    1000.0 * apogee_taurus["par_err"] / (apogee_taurus["par"]) ** 2
)  # convert parralax error to distance error

apogee_taurus = apogee_taurus[apogee_taurus.dist < 200]
apogee_taurus = apogee_taurus.dropna().reset_index(drop=True)

from astropy.coordinates import SkyCoord, LSR
import astropy.units as u

c = SkyCoord(
    ra=apogee_taurus["ra"].values * u.degree,
    dec=apogee_taurus["dec"].values * u.degree,
    pm_ra_cosdec=apogee_taurus["vra"].values * u.mas / u.yr,
    pm_dec=apogee_taurus["vdec"].values * u.mas / u.yr,
    distance=apogee_taurus["dist"].values * u.pc,
    radial_velocity=apogee_taurus["rv"].values * u.km / u.s,
    frame="icrs",
)
l = c.transform_to(LSR())

apogee_taurus["vlsrra"] = l.pm_ra_cosdec.value
apogee_taurus["vlsrdec"] = l.pm_dec.value
apogee_taurus["vlsr"] = l.radial_velocity.value

apogee_taurus = apogee_taurus.drop(
    columns={
        "rv",
        "vra",
        "vdec",
        "sb2",
        "chi2",
        "slope",
        "par",
        "par_err",
        "rv_err_s",
        "rv_err",
    }
)
apogee_taurus = apogee_taurus[
    apogee_taurus["vlsrra"] < 100
]  # remove one star with ridiculous proper motion
apogee_taurus = apogee_taurus[apogee_taurus.vlsr_err < 5]

# %% Generate n number of random samples

import functions as fcs

sample_size = 1000
frame = fcs.rand_sample(
    apogee_taurus, sample_size, r"MilkyWay/Gaia_allsky/Taurus_edr3_dr17_test/"
)

# %% Import back to python and compute VSF

d_max = 100
n_bins = 50

dispersion1 = np.zeros((sample_size, n_bins))  # 1st order statistics
dispersion2 = np.zeros((sample_size, n_bins))  # 2nd order statistics
dispersion3 = np.zeros((sample_size, n_bins))  # 3rd order statistics

for i in range(0, sample_size):
    tableq = pd.read_csv(
        "/home/trungha/Downloads/Taurus1000/generated_" + str(i) + ".csv"
    )
    tableq = tableq.drop(
        [np.random.randint(0, len(tableq))]
    )  # drop 1 random element in each generated realization

    tableq.ra = tableq.ra * np.pi / 180.0
    tableq.dec = tableq.dec * np.pi / 180.0
    # tableq.rv = 0 # test effect on vsf if RV didn't exist

    (
        tableq["x_axis"],
        tableq["y_axis"],
        tableq["z_axis"],
        tableq["v_x"],
        tableq["v_y"],
        tableq["v_z"],
    ) = radec2cartesian(
        tableq.ra.values,
        tableq.dec.values,
        tableq.dist.values,
        tableq.rv.values,
        tableq.vra.values,
        tableq.vdec.values,
    )

    (
        dist_array,
        v_diff_mean,
        v_diff_matrix,
        dist_matrix,
        np_dist,
        v_diff_mean2,
        v_diff_mean3,
        v_diff_sigma,
    ) = VSF_3Dcartesian(
        tableq.x_axis.values,
        tableq.y_axis.values,
        tableq.z_axis.values,
        tableq.v_x.values,
        tableq.v_y.values,
        tableq.v_z.values,
        n_bins,
        d_max,
    )

    # if i == sample_size / #add progress bar here, tba later

    dispersion1[i, :] = v_diff_mean
    dispersion2[i, :] = v_diff_mean2
    dispersion3[i, :] = v_diff_mean3


v_diff_mean = np.zeros(n_bins)
error_mean = np.zeros(n_bins)
v_diff_mean2 = np.zeros(n_bins)
v_diff_mean3 = np.zeros(n_bins)

for k in range(0, n_bins):
    v_diff_mean[k] = np.nanmean(dispersion1[:, k])
    error_mean[k] = np.nanstd(dispersion1[:, k])
    v_diff_mean2[k] = np.nanmean(dispersion2[:, k])
    v_diff_mean3[k] = np.nanmean(dispersion3[:, k])

# cloud = 'All Orion, with ONC'
# v_diff_mean = tableVSF.AOriwONC_mean_v
# error_mean = tableVSF.AOriwONC_err

# y_expect=dist_array**(1.0/3)*2.1
# y_expect2=dist_array**(1.0/2)*1.7
y_larson = 1.1 * dist_array**0.38

# rough plot of Qian 2018 CVD measurements
dist_array_scale = dist_array * np.sqrt(3 / 2)
ys_qian = 0.85 * dist_array_scale ** (1.0 / 2) * np.sqrt(2) * np.sqrt(3)
yl_qian = 0.85 * dist_array_scale ** (1.0 / 3) * np.sqrt(2) * np.sqrt(3)
m = (yl_qian[23] - ys_qian[18]) / (dist_array_scale[23] - dist_array_scale[18])
b = ys_qian[18] - m * dist_array_scale[18]
y_btw = m * dist_array_scale + b

plt.clf()
f = plt.figure(figsize=(9, 8), dpi=300)
ax = f.add_subplot(111)
plt.loglog(
    dist_array[:-1],
    v_diff_mean[:-1],
    marker="o",
    linestyle="-",
    markersize=10,
    color="C3",
    linewidth=2,
)
plt.errorbar(
    dist_array,
    v_diff_mean,
    yerr=error_mean,
    fmt="none",
    label=None,
    elinewidth=3,
    color="coral",
)

# plt.loglog(dist_array[:],y_expect,linestyle="-.",label="1/3 (Kolmogorov)",color="black",linewidth=3)
# plt.loglog(dist_array[:],y_expect2,label="1/2 (Supersonic)",color="C6",linestyle='--',linewidth=3)
# plt.loglog(dist_array_scale[8:19],ys_qian[8:19],label="Taurus Gas VSF (1-3 pc)",color="C6",linestyle='None',marker='d',markersize=10)
# plt.loglog(dist_array_scale[23:31],yl_qian[23:31],label="Taurus Gas VSF (5-10 pc)",color="C14",linestyle='None',marker='d',markersize=10)
# plt.loglog(dist_array_scale[19:23],y_btw[19:23],color="C7",linestyle='None',marker='d',markersize=10)
plt.loglog(
    dist_array[:], y_larson, label="0.38 (Larson's law)", color="blue", linewidth=3
)
# plt.fill_between(dist_array[1:], v_diff_mean[1:] + error_mean[1:], v_diff_mean[1:] - error_mean[1:], alpha=0.2, color='C3')

plt.xlabel("$\ell$ (pc)", size=24)
plt.ylabel(r"$\langle|\delta v|\rangle\, \rm (km/s)$", size=24)
y_locs = np.array([0.5, 1, 2, 6, 14])
y_labels = np.array([0.5, 1, 2, 6, 14])
plt.yticks(y_locs, y_labels)
plt.legend(loc="lower right", prop={"size": 20})
plt.ylim(0, 18)
plt.xlim(0.35, 110)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

props = dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.3)
ax.text(0.4, 14, "Taurus, RV = 0", fontsize=24, bbox=props, fontweight="bold")
plt.grid()
plt.show()

# %% Scatter plot the Taurus cloud

# Scatter Plot stars and their position in the sky
fig = px.scatter_3d(
    tableq,
    x="x_axis",
    y="y_axis",
    z="z_axis",
    opacity=1,
    labels={"x_axis": "X (pc)", "y_axis": "Y (pc)", "z_axis": "Z (pc)"},
    title="3-D scatter plot of Taurus",
    hover_data={"x_axis", "y_axis", "z_axis"},
)
fig.show()

# %% Assu
