"""Cross-check of the MIST evolutionary mass of the J0008+0128 giant with the `isochrones`
package (Morton 2015), the tool Zhao et al. (2026) used. Not quoted in the note.

Zhao et al. fit MIST isochrones to the APOGEE Teff, log g and [Fe/H], Gaia and 2MASS
photometry and the Gaia DR3 parallax, and obtain M1 = 1.45 (DR17) and 1.52 (SDSS-V) Msun.
This script repeats that fit with the package's default priors (Chabrier IMF through the EEP
prior, log-flat age, local-disc [Fe/H], distance^2 prior, flat A_V in 0-1) for several input
choices, and writes isochrones_results.json.

Sampling: emcee, driven directly through BasicStarModel.lnpost, started from the better of a
global (differential evolution) optimum and a first-ascent giant-branch optimum. Both optima
are recorded, because the global search alone can stall in a poorly fitting core-helium-burning
solution. Parameters: [EEP, log10 age, [Fe/H], distance (pc), A_V].

Needs its own environment (isochrones 2.1 requires numpy < 2 and pandas < 2):
    python -m venv isoenv && isoenv/bin/pip install -r requirements-isochrones.txt
    ISOCHRONES=/path/with/5GB/free isoenv/bin/python j0008_isochrones.py [CASE ...]
The MIST isochrone grid (~0.7 GB download, ~4.5 GB unpacked) is fetched on first use.
"""
import json
import lzma
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ISO_DIR = os.environ.setdefault("ISOCHRONES", os.path.join(os.path.expanduser("~"), ".isochrones"))


def ensure_bc_table():
    """isochrones 2.1 downloads the MIST bolometric corrections from a URL that now returns 404
    (the tables moved to BC_tables/v1/ on mist.science). Fetch them from the new location."""
    path = os.path.join(ISO_DIR, "BC", "mist", "UBVRIplus.txz")
    try:
        with lzma.open(path) as fh:
            fh.read(1)
        return
    except (OSError, lzma.LZMAError):
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = "https://mist.science/BC_tables/v1/UBVRIplus.txz"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


ensure_bc_table()

import emcee  # noqa: E402
from scipy.optimize import differential_evolution, minimize  # noqa: E402
from isochrones import get_ichrone  # noqa: E402
from isochrones.starmodel import SingleStarModel  # noqa: E402

BANDS = ["G", "BP", "RP", "J", "H", "K"]
MIST = get_ichrone("mist", bands=BANDS)

# Photometry (data/gaia_dr3_2mass.csv): Gaia DR3 G, BP, RP with a 0.01 mag floor (their
# catalogue errors are < 0.001 mag); 2MASS PSC J, H, Ks with catalogue errors (quality AAA).
PHOT = dict(G=(12.692175, 0.01), BP=(13.165074, 0.01), RP=(12.057319, 0.01),
            J=(11.287, 0.026), H=(10.779, 0.022), K=(10.672, 0.019))
PLX_DR3 = (0.6828, 0.0300)           # mas, as published
PLX_ZP = (0.6828 + 0.023, 0.0300)    # Lindegren et al. (2021) zero point (-0.023 mas) removed

# Spectroscopic inputs from Zhao et al. Table S2: catalogue rows and "corrected" rows.
CASES = {
    # As Zhao et al. describe: catalogue APOGEE DR17 parameters, photometry, published parallax
    "zhao_dr17": dict(Teff=(4955, 10), logg=(3.15, 0.03), feh=(-0.25, 0.01), parallax=PLX_DR3, **PHOT),
    # The same with the APOGEE SDSS-V catalogue parameters
    "zhao_sdss5": dict(Teff=(4913, 13), logg=(3.20, 0.03), feh=(-0.23, 0.01), parallax=PLX_DR3, **PHOT),
    # zhao_dr17 with a 0.03 mag floor on all photometry
    "zhao_dr17_phot03": dict(Teff=(4955, 10), logg=(3.15, 0.03), feh=(-0.25, 0.01), parallax=PLX_DR3,
                             **{k: (v[0], max(v[1], 0.03)) for k, v in PHOT.items()}),
    # zhao_dr17 with the corrected-row uncertainties (+/-40 K, +/-0.07 dex, +/-0.05 dex)
    "zhao_dr17_wide": dict(Teff=(4955, 40), logg=(3.15, 0.07), feh=(-0.25, 0.05), parallax=PLX_DR3, **PHOT),
    # zhao_dr17 with the distance effectively fixed at 1/parallax (parallax error 0.003 mas)
    "zhao_dr17_dfixed": dict(Teff=(4955, 10), logg=(3.15, 0.03), feh=(-0.25, 0.01), parallax=(0.6828, 0.003),
                             **PHOT),
    # ... and with a loose Teff constraint (+/-100 K): close to Zhao et al.'s M, R and age
    "zhao_dr17_dfixed_teff100": dict(Teff=(4955, 100), logg=(3.15, 0.03), feh=(-0.25, 0.01),
                                     parallax=(0.6828, 0.003), **PHOT),
    # Corrected DR17 parameters, photometry, published parallax
    "corr_plx": dict(Teff=(4964, 40), logg=(3.14, 0.07), feh=(-0.24, 0.05), parallax=PLX_DR3, **PHOT),
    # Corrected DR17 parameters, photometry, zero-point-corrected parallax
    "corr_plxzp": dict(Teff=(4964, 40), logg=(3.14, 0.07), feh=(-0.24, 0.05), parallax=PLX_ZP, **PHOT),
    # Corrected DR17 spectroscopy only (no photometry or parallax)
    "corr_spec": dict(Teff=(4964, 40), logg=(3.14, 0.07), feh=(-0.24, 0.05)),
}

BOUNDS = [(250, 700), (8.5, 10.13), (-1.0, 0.4), (500, 4000), (0.0, 1.0)]


def run(case, nwalkers=48, nsteps=6000, seed=1):
    kw = CASES[case]
    mod = SingleStarModel(MIST, **kw)
    has_dist = "parallax" in kw or "G" in kw
    bounds = BOUNDS if has_dist else BOUNDS[:3]
    ndim = len(bounds)

    def lnpost(p):
        if not has_dist:  # no photometry or parallax: distance and A_V do not enter
            p = list(p) + [1400.0, 0.1]
        v = mod.lnpost(np.asarray(p, dtype=float))
        return v if np.isfinite(v) else -np.inf

    def neg(p):
        v = lnpost(p)
        return -v if np.isfinite(v) else 1e30

    t0 = time.time()
    de = differential_evolution(neg, bounds, seed=seed, tol=1e-8, maxiter=2000, polish=False)
    # First-ascent giant-branch start (MIST EEP 454-605) matching Teff and log g.
    T0, g0, best = kw["Teff"][0], kw["logg"][0], None
    for logage in np.arange(9.0, 10.1, 0.01):
        for eep in np.arange(454, 605, 0.5):
            T, g = MIST.interp_value(np.array([eep, logage, kw["feh"][0], 1400.0, 0.1]), ["Teff", "logg"])
            if np.isfinite(g):
                c = ((T - T0) / 30) ** 2 + ((g - g0) / 0.05) ** 2
                if best is None or c < best[0]:
                    best = (c, eep, logage)
    d0 = 1000.0 / kw["parallax"][0] if "parallax" in kw else 1400.0
    starts = {"global": de.x, "giant_branch": np.array([best[1], best[2], kw["feh"][0], d0, 0.1])[:ndim]}
    opt = {k: minimize(neg, x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8))
           for k, x in starts.items()}
    best_key = min(opt, key=lambda k: opt[k].fun)
    xbest = opt[best_key].x

    rng = np.random.default_rng(seed)
    scale = np.array([2.0, 0.01, 0.01, 10.0, 0.01])[:ndim]
    p0 = xbest + scale * rng.standard_normal((nwalkers, ndim))
    for i in range(nwalkers):
        while not np.isfinite(lnpost(p0[i])):
            p0[i] = xbest + 0.3 * scale * rng.standard_normal(ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    sampler.run_mcmc(p0, nsteps, progress=False)
    tau = sampler.get_autocorr_time(tol=0)
    burn = int(max(1000, 5 * np.max(tau)))
    chain = sampler.get_chain(discard=burn, thin=max(1, int(np.max(tau) / 2)), flat=True)
    full = chain if has_dist else np.column_stack([chain, np.full(len(chain), 1400.0), np.full(len(chain), 0.1)])
    derived = np.array([MIST.interp_value(p, ["mass", "radius", "Teff", "logg", "logL"]) for p in full])

    pct = lambda x, nd=3: [round(float(v), nd) for v in np.percentile(x, [16, 50, 84])]
    out = {"case": case, "inputs": {k: list(v) for k, v in kw.items()},
           "mass": pct(derived[:, 0]), "radius": pct(derived[:, 1], 2), "Teff": pct(derived[:, 2], 0),
           "logg": pct(derived[:, 3]), "logL": pct(derived[:, 4]), "age_Gyr": pct(10 ** chain[:, 1] / 1e9, 2),
           "feh": pct(chain[:, 2])}
    if has_dist:
        out["distance_pc"] = pct(chain[:, 3], 0)
        out["AV"] = pct(chain[:, 4])
    out.update({"optima_lnpost": {k: round(float(-o.fun), 2) for k, o in opt.items()}, "start_used": best_key,
                "n_samples": len(chain), "tau_max": round(float(np.max(tau)), 1),
                "acceptance": round(float(np.mean(sampler.acceptance_fraction)), 2),
                "seconds": round(time.time() - t0, 1)})
    return out


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1] != "all" else list(CASES)
    results = [run(c) for c in names]
    for r in results:
        d = f", d = {r['distance_pc'][1]:.0f} pc" if "distance_pc" in r else ""
        print(f"{r['case']:18s} M = {r['mass'][1]:.2f} +{r['mass'][2] - r['mass'][1]:.2f} "
              f"-{r['mass'][1] - r['mass'][0]:.2f} Msun, Teff = {r['Teff'][1]:.0f} K, "
              f"R = {r['radius'][1]:.2f} Rsun{d}")
    out_path = os.path.join(HERE, "isochrones_results.json")
    merged = {}
    if os.path.exists(out_path):  # running named cases updates just those entries
        merged = {r["case"]: r for r in json.load(open(out_path))}
    merged.update({r["case"]: r for r in results})
    with open(out_path, "w") as fh:
        json.dump([merged[c] for c in CASES if c in merged], fh, indent=1)
    print("Wrote isochrones_results.json")
