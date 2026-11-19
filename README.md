# The dark companion of J0008+0128 need not be a supermassive neutron star

Paper and code for Stevenson (2026, RNAAS, in prep.), a comment on Zhao et al. (2026,
Science Advances, 12, eaeg3307) about the red giant plus dark companion binary J0008+0128
(Gaia DR3 2546630699873483008).

Archived on Zenodo: [doi:10.5281/zenodo.24293381](https://doi.org/10.5281/zenodo.24293381)
(this DOI covers all versions). `CITATION.cff` gives the citation details.

## Contents

| Path | What it is |
|---|---|
| `paper/rnaas_j0008.tex` | The Research Note (AASTeX v7.0.1) |
| `paper/rnaas_j0008.pdf` | Compiled note |
| `paper/fig_m1m2.pdf` | Figure 1, as used by the note |
| `j0008_analysis.py` | Reproduces every number in the note and Figure 1 |
| `inputs.csv` | Every input value with its source |
| `fig_m1m2.pdf`, `fig_m1m2.png` | Figure 1, as written by `j0008_analysis.py` |
| `j0008_hrd.py` | MIST track masses quoted in the note, plus an HR and Kiel diagram |
| `j0008_hrd.png` | The HR and Kiel diagram written by `j0008_hrd.py` (not in the note) |
| `fig_m2_prior.png` | Implied prior on M2 and the effect of other priors, written by `j0008_analysis.py` (not in the note) |
| `j0008_isochrones.py` | Cross-check of Zhao et al.'s MIST evolutionary mass with the `isochrones` package (not in the note) |
| `isochrones_results.json` | Results written by `j0008_isochrones.py` |
| `requirements-isochrones.txt` | Separate environment for `j0008_isochrones.py` |
| `data/` | Catalogue exports behind `inputs.csv` (see below) |

## Reproducing the numbers and Figure 1

```
pip install -r requirements.txt
python j0008_analysis.py
```

The script prints the numbers quoted in the note and writes `fig_m1m2.pdf` and
`fig_m1m2.png`. It covers:

- the sensitivity of the companion mass M2 to the giant mass M1 and the inclination
- a check that Zhao et al.'s adopted M1 = 1.67 ± 0.15 and M2 = 2.20 ± 0.13 Msun match the
  inverse-variance means of their five estimates, and the companion mass you get by
  propagating M1 = 1.67 ± 0.15 Msun and i = 73 ± 11 deg directly
- gR^2/G giant masses at different distances, including the zero-point-corrected parallax
  distance with its uncertainty, with the APOGEE DR17 and SDSS-V log g
- the Lindegren et al. (2021) parallax zero point for this star
- the Teff implied by the Gaia FLAME luminosity and radius
- Martig et al. (2016) [C/N] masses and their sensitivities
- APOKASC-3 (Pinsonneault et al. 2025) seismic masses of Kepler giants with J0008's parameters:
  three broad selections (log g 3.0-3.3, [Fe/H] and [C/N] from -0.3 to 0), and boxes centred on each survey's
  corrected parameters in Zhao et al. Table S2 (medians 1.23-1.26 Msun)
- the Monte Carlo companion mass for M1 = 1.25 ± 0.15 Msun, i = 73 ± 11 deg and
  f(M) = 0.587 ± 0.017 Msun, and the same calculation for other choices of M1
- the APOKASC-3 mass used as a prior and updated with J0008's own log g and zero-point-corrected
  parallax (importance-weighted draws), giving M1 = 1.29 ± 0.13 Msun and M2 = 1.91 Msun with
  the DR17 log g
- a check that is not quoted in the note: the prior on M2 implied by the Monte Carlo
  (proportional to df/dM2), and the companion mass if it is replaced by a flat prior or by the
  neutron star mass distribution of Alsing, Silva & Berti (2018), written to `fig_m2_prior.png`

`gaiadr3-zeropoint` is optional. Without it, the script uses the zero point stored in
`inputs.csv`, and the output is identical. The Monte Carlo uses 100,000 draws and a fixed
random seed, so the numbers are exactly reproducible. It runs in a few seconds.

## MIST track masses

```
python j0008_hrd.py                 # downloads the MIST v1.2 [Fe/H] = -0.25 EEP tracks (~110 MB)
python j0008_hrd.py /path/to/EEPS   # or use an already unpacked EEPS directory
```

This prints the MIST track masses quoted in the note (interpolated at the observed Teff, at
fixed log g and at fixed luminosity for three distances) and writes `j0008_hrd.png`.

## isochrones cross-check (not in the note)

Zhao et al. obtain M1 = 1.45 (APOGEE DR17) and 1.52 (SDSS-V) Msun by fitting MIST isochrones
with the `isochrones` package to the APOGEE parameters, photometry and parallax.
`j0008_isochrones.py` repeats that fit with the package's default priors for several input
choices (the inputs are listed in the script):

```
python -m venv isoenv && isoenv/bin/pip install -r requirements-isochrones.txt
ISOCHRONES=/path/with/5GB/free isoenv/bin/python j0008_isochrones.py
```

`isochrones` 2.1 needs NumPy < 2 and pandas < 2, hence the separate environment. The MIST
isochrone grid (about 0.7 GB to download, 4.5 GB unpacked) is fetched on first use. The
package's URL for the MIST bolometric corrections no longer works, so the script fetches
them from `https://mist.science/BC_tables/v1/UBVRIplus.txz` first. The nine fits take a
few minutes and write `isochrones_results.json`; naming cases on the command line reruns just those.

With Zhao et al.'s stated DR17 inputs and the package defaults, the fit gives about 1.23 Msun
rather than 1.45 Msun. Their M, R = 5.34 ± 0.04 Rsun and age are approached only when the
distance is held essentially fixed at 1/parallax and Teff is allowed to drift to about 5010 K
(case `zhao_dr17_dfixed_teff100`).

## Rebuilding the note

To rebuild the note, copy `fig_m1m2.pdf` into `paper/` and compile `rnaas_j0008.tex` with
AASTeX v7.0.1 (`aastex701.cls`, available from the AAS and on Overleaf).

## Data

`inputs.csv` takes its catalogue values from these exports for 2M00080122+0128582 /
Gaia DR3 2546630699873483008:

| File | Source |
|---|---|
| `data/apogee_dr17_aspcap.csv` | SDSS DR17 SkyServer, `apogeeStar` joined with `aspcapStar` |
| `data/apogee_dr17_distmass.csv` | SDSS DR17 SkyServer, APOGEE DistMass value-added catalogue (Stone-Martinez et al. 2024) |
| `data/apogee_dr17_starhorse.csv` | SDSS DR17 SkyServer, APOGEE StarHorse value-added catalogue |
| `data/apokasc3_table4.csv` | VizieR J/ApJS/276/69 table 4 (APOKASC-3, Pinsonneault et al. 2025), selected columns: KIC, evolutionary state, catalogue, seismic parameters, mass, radius, APOGEE DR17 Teff, log g, [Fe/H], [C/Fe], [N/Fe] |
| `data/gaia_dr3_2mass.csv` | Gaia archive, `gaiadr3.gaia_source` with its 2MASS cross-match (`tmass_psc_xsc_best_neighbour`, `tmass_psc_xsc_join`, `gaiadr1.tmass_original_valid`): G, BP, RP, J, H, Ks for `j0008_isochrones.py` |
| `data/gaia_dr3_source_flame.vot.gz` | Gaia archive, `gaiadr3.gaia_source` joined with `gaiadr3.astrophysical_parameters` (the ADQL query is in the file header) |

Values from Zhao et al. (2026) are taken from the paper and its supplementary material, as
listed in `inputs.csv`.

## Licence

MIT. See `LICENSE`.
