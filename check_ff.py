# %%
import healpy as hp 
import numpy as np 
import pysm3
import pysm3.units as u

import pymaster as nmt
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams["text.usetex"] = True

from icecream import ic

# %%
# Hyperparameters 

# Nside for all analysis (dgrade point source mask is onlt effect).
NSIDE = 256

# Frequencies to calculate model.
FREQS = {
    "100": 100. * u.GHz, 
    "143": 143. * u.GHz,
    }

# NaMaster parameters.
APOSIZE= 3.
APOTYPE="C1"

# %%

def mask_3pct(nside):
    gl,gb = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)), lonlat=True)
    r = hp.rotator.Rotator(coord=['G', 'C'])
    ra,dec = r(gl, gb, lonlat=True)
    mask = np.ones(hp.nside2npix(nside))
    mask[np.where(ra < -30.)] = 0.
    mask[np.where(ra > 30.)] = 0.
    mask[np.where(dec < -70.)] = 0.
    mask[np.where(dec > -42.)] = 0.
    return mask

def mask_gal(nside):
    gl,gb = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)), lonlat=True)
    mask = np.zeros(hp.nside2npix(nside))
    mask[np.where(gb < -15.)] = 1.
    mask[np.where(gb > 15.)] = 1.
    return mask

def ff_model_table4_planck150201588(EM, T_e, nu=100e9):
    T_4 = T_e / 1e4
    nu_9 = nu / 1e9
    g_ff = np.log(np.exp(5.960 - np.sqrt(3.) / np.pi * np.log(nu_9 * T_4 ** (-3/2))) + np.exp(1))
    tau = 0.05468 * T_e ** (-3. / 2.) * nu_9 ** -2 * EM * g_ff
    return 1e6 * T_e * (1. - np.exp(-tau))

def namaster_spec(arr, mask, nlb=50, is_Dell=True):
    nmtbin = nmt.NmtBin.from_nside_linear(NSIDE, nlb, is_Dell=is_Dell)
    f0 = nmt.NmtField(mask, [arr])
    bpws = nmt.compute_full_master(f0, f0, nmtbin)
    return nmtbin.get_effective_ells(), bpws[0]

# %%
# Compute Commander model at 100 GHz, 143 GHz.
maps = {}
maps["Commander"] = {}

EM, T_e = hp.read_map("COM_CompMap_freefree-commander_0256_R2.00.fits", field=(0, 3))
for (key, nu) in FREQS.items():
    arr = ff_model_table4_planck150201588(EM, T_e, nu=nu.to(u.Hz).value) * u.uK_RJ
    maps["Commander"][key] = arr.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu)) 
# %%
# Compute Hutschenreuter model at 100 GHz, 143 GHz.
maps["Heutschenreuter"] = {}

hdu = fits.open("components_w_ff.fits")
T_e = hp.read_map("COM_CompMap_freefree-commander_0256_R2.00.fits", field=3)
EM = np.exp(hdu[1].data["EPSILON_MEAN"])

for (key, nu) in FREQS.items():
    arr = ff_model_table4_planck150201588(EM, T_e, nu=nu.to(u.Hz).value) * u.uK_RJ
    maps["Heutschenreuter"][key] = arr.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu))                

# %%
# Compute masks.
masks = {
    "SPT3G": mask_3pct(NSIDE),
    "PointSources": np.floor(hp.ud_grade(hp.read_map("LFI_Mask_PointSrc_2048_R2.00.fits"), nside_out=NSIDE)),
    "Gal": mask_gal(NSIDE),
    "GalPointSources": mask_gal(NSIDE) * np.floor(hp.ud_grade(hp.read_map("LFI_Mask_PointSrc_2048_R2.00.fits"), nside_out=NSIDE)),
    "SPT3GPointSources": mask_3pct(NSIDE) * np.floor(hp.ud_grade(hp.read_map("LFI_Mask_PointSrc_2048_R2.00.fits"), nside_out=NSIDE)),
}

# %%
# Visualize masks.
def maskview(arr, mask, *args, **kwargs):
    masked_arr = np.copy(arr)
    masked_arr[mask == 0] = hp.UNSEEN
    return hp.mollview(masked_arr, **kwargs)

hp.mollview(maps["Heutschenreuter"]["100"], norm='log', min=0.1, max=3000, title=r"${\rm Heutschenreuter~Free-free~100~GHz}$")
hp.mollview(maps["Commander"]["100"], norm='log', min=0.1, max=3000, title=r"${\rm Commander~Free-free~100~GHz}$")

maskview(maps["Heutschenreuter"]["100"].value, masks["Gal"], min=0.1, max=100, title=r"${\rm Heutschenreuter~Free-free~100~GHz,~Gal~Mask}$")
maskview(maps["Commander"]["100"].value, masks["Gal"], min=0.1, max=100, title=r"${\rm Commander~Free-free~100~GHz,~Gal~Mask}$")

maskview(maps["Heutschenreuter"]["100"].value, masks["GalPointSources"], min=0.1, max=100, title=r"${\rm Heutschenreuter~Free-free~100~GHz,~Gal~Mask~+~Pnt~Src}$")
maskview(maps["Commander"]["100"].value, masks["GalPointSources"], min=0.1, max=100, title=r"${\rm Commander~Free-free~100~GHz,~Gal~Mask~+~Pnt~Src}$")

maskview(maps["Heutschenreuter"]["100"].value, masks["PointSources"], min=0.1, max=100, title=r"${\rm Heutschenreuter~Free-free~100~GHz,~Pnt~Src}$")
maskview(maps["Commander"]["100"].value, masks["PointSources"], min=0.1, max=100, title=r"${\rm Commander~Free-free~100~GHz,~Gal~Mask,~Pnt~Src}$")

maskview(maps["Heutschenreuter"]["100"].value, masks["SPT3GPointSources"], min=0.1, max=1, title=r"${\rm Heutschenreuter~Free-free~100~GHz,~SPT3G+Pnt~Src}$")
maskview(maps["Commander"]["100"].value, masks["SPT3GPointSources"], min=0.1, max=1, title=r"${\rm Commander~Free-free~100~GHz,~Gal~Mask,~SPT3G+Pnt~Src}$")

# %%
# Calculate spectra on all difference masks.
spectra = {}
for (k, arr) in maps.items():
    spectra[k] = {}
    for (masklabel, mask) in masks.items():
        spectra[k][masklabel] = {}
        mask_apo = nmt.mask_apodization(mask, APOSIZE, apotype=APOTYPE)
        for (nulabel, nuarr) in arr.items():
            ells, spectra[k][masklabel][nulabel] = namaster_spec(nuarr, mask)
# %%
# Plot Galaxy mask w/ and w/o point source mask.
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

plt.suptitle(r"${\rm Galaxy~+Pnt~Src~Masks}$")

ax[0].semilogy(ells, spectra["Commander"]["Gal"]["100"], "C0-")
ax[0].semilogy(ells, spectra["Commander"]["GalPointSources"]["100"], "C0--")
ax[0].semilogy(ells, spectra["Heutschenreuter"]["Gal"]["100"], "C1-")
ax[0].semilogy(ells, spectra["Heutschenreuter"]["GalPointSources"]["100"], "C1--")

ax[1].semilogy(ells, spectra["Commander"]["Gal"]["143"], "C0", label=r"${\rm Commander}$")
ax[1].semilogy(ells, spectra["Commander"]["GalPointSources"]["143"], "C0--", label=r"${\rm Commander+Pnt Src Mask}$")
ax[1].semilogy(ells, spectra["Heutschenreuter"]["Gal"]["143"], "C1-", label=r"${\rm Heutschenreuter}$")
ax[1].semilogy(ells, spectra["Heutschenreuter"]["GalPointSources"]["143"], "C1--", label=r"${\rm Heutschenreuter+Pnt Src Mask}$")

ax[0].set_xscale('log')
ax[1].set_xscale('log')

ax[1].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

ax[0].set_title(r"${\rm 100~GHz}$")
ax[1].set_title(r"${\rm 143~GHz}$")

ax[0].set_xlabel(r'$\ell$')
ax[1].set_xlabel(r'$\ell$')
ax[0].set_ylabel(r"$\mathcal{D}_\ell^{\rm TT}~[{\rm \mu K^2}]$")

ax[0].set_ylim(1e-2, 1e3)
ax[0].set_xlim(20, 800)
ax[1].set_xlim(20, 800)

# %%
# Plot SPT-3G patch w/ and w/o point source mask.
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

plt.suptitle(r"${\rm SPT~3G~patch}$")

ax[0].semilogy(ells, spectra["Commander"]["SPT3G"]["100"], "C0-")
ax[0].semilogy(ells, spectra["Commander"]["SPT3GPointSources"]["100"], "C0--")
ax[0].semilogy(ells, spectra["Heutschenreuter"]["SPT3G"]["100"], "C1")

ax[1].semilogy(ells, spectra["Commander"]["SPT3G"]["143"], "C0-", label=r"${\rm Commander}$")
ax[1].semilogy(ells, spectra["Commander"]["SPT3GPointSources"]["143"], "C0--", label=r"${\rm Commander+Pnt Src Mask}$")
ax[1].semilogy(ells, spectra["Heutschenreuter"]["SPT3G"]["143"], "C1", label=r"${\rm Heutschenreuter}$")
ax[1].semilogy(ells, spectra["Heutschenreuter"]["SPT3GPointSources"]["143"], "C1--", label=r"${\rm Heutschenreuter+PntSrcMask}$")

ax[0].set_xscale('log')
ax[1].set_xscale('log')

ax[1].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

ax[0].set_title(r"${\rm 100~GHz}$")
ax[1].set_title(r"${\rm 143~GHz}$")

ax[0].set_xlabel(r'$\ell$')
ax[1].set_xlabel(r'$\ell$')
ax[0].set_ylabel(r"$\mathcal{D}_\ell^{\rm TT}~[{\rm \mu K^2}]$")

ax[0].set_ylim(1e-5, 1e0)

ax[0].set_xlim(20, 800)
ax[1].set_xlim(20, 800)


