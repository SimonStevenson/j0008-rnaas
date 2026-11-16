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
  distance with its uncertainty
- the Lindegren et al. (2021) parallax zero point for this star
- the Teff implied by the Gaia FLAME luminosity and radius
- Martig et al. (2016) [C/N] masses and their sensitivities
- the Monte Carlo companion mass for M1 = 1.20 ± 0.15 Msun, i = 73 ± 11 deg and
  f(M) = 0.587 ± 0.017 Msun, and the same calculation for other choices of M1

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
| `data/gaia_dr3_source_flame.vot.gz` | Gaia archive, `gaiadr3.gaia_source` joined with `gaiadr3.astrophysical_parameters` (the ADQL query is in the file header) |

Values from Zhao et al. (2026) are taken from the paper and its supplementary material, as
listed in `inputs.csv`.

## Licence

MIT. See `LICENSE`.
