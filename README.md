# PROLITE

**PSF Reconstructor of Off-axis LBT Images via Tiptop for Extragalactic observations**

PROLITE is a Python package that reconstructs the point spread function (PSF) across the whole field of an image taken with a single-conjugate adaptive optics (SCAO) system. It was developed for SOUL–LUCI at the Large Binocular Telescope as part of a Master's thesis at the University of Pisa.

In SCAO the correction is optimal only along the line of sight of the guide star, and the PSF degrades with distance from it. PROLITE needs only the reduced focal-plane image and a few off-axis reference stars, with no AO telemetry. It uses the analytic simulator [TIPTOP](https://github.com/astro-tiptop/TIPTOP) to estimate the atmospheric and instrumental parameters that best reproduce the observed PSFs:

- seeing;
- a multi-layer Cn² profile with wind speed and direction for each layer;
- optionally, non-common-path aberrations and residual jitter.

With these parameters, TIPTOP can then produce the PSF at any position of the field.

## How it works

1. **Read the image**: load the FITS image and extract the observing conditions from its header (elevation, filter, AO loop frequency, WFS counts).
2. **Find stars**: detect the AO guide star and the field stars, then choose the reference stars.
3. **Cut regions**: extract a square data region around each reference star.
4. **Estimate parameters**: run an optimisation loop. At each cycle, PROLITE:
   1. converts a trial parameter vector into a TIPTOP configuration;
   2. simulates the PSFs on the GPU;
   3. matches each simulated PSF to the observed one in position and flux;
   4. evaluates a merit function, the relative RMS of the residuals inside a circle around each star.

   Available optimisers are dual annealing, differential evolution, Nelder–Mead and a genetic algorithm (pyGAD).
5. **Post-process**: compare reconstructed and observed PSFs, including on control stars not used in the fit. The output is FWHM, flux-error and residual plots plus LaTeX tables, for one or several analyses.

## Repository structure

Modules are listed in the order in which they are first used by the example script `data_analysis.py`:

- `headerReader.py`: extracts the observing conditions from the FITS header.
- `findStar.py`: detects the AO guide star and the field stars.
- `pixelUtiles.py`: provides circular masks and region boundaries on pixel grids.
- `sensorPosition.py`: converts star positions between pixels, arcsec and TIPTOP polar coordinates.
- `dataRegions.py`: cuts the square data regions around the reference stars.
- `graphics.py`: produces all the static and real-time plots.
- `format_and_control.py`: packs and unpacks the parameter vector and builds the parameter bounds.
- `files.py`: reads and writes `.ini` configuration files.
- `estimator.py`: wraps the optimisers and records the history of the estimation.
- `merit.py`: runs TIPTOP, matches simulated to observed PSFs and computes the merit function.
- `saver.py`: saves the configuration of a run to `save_status.ini`.
- `plot_results.py`: post-processes one or more analyses into plots and LaTeX tables; also a command-line tool.
- `simulation.py`: generates synthetic focal-plane images with TIPTOP for validation (not used in the example).

## Requirements

Python 3 with `numpy`, `scipy`, `astropy`, `photutils`, `matplotlib`, `pygad` and `tqdm`, plus a working TIPTOP installation (GPU recommended). Plots use LaTeX rendering, so a LaTeX distribution must be installed.

## Usage

PROLITE is used as a library: the classes above are combined in your own script to build the analysis you need. The script `data_analysis.py` is an example of this, showing the complete workflow used on a SOUL–LUCI image of the Palomar 10 globular cluster. It is a good starting point for your own analysis:

1. Copy `data_analysis.py` and adapt the settings at the top:
   - image path;
   - star-finder parameters;
   - indices and radii of the reference stars;
   - layer heights and parameter bounds;
   - optimiser.
2. Run it:

   ```bash
   python my_analysis.py
   ```

3. Re-run or compare the post-processing on one analysis, a folder of analyses, or several branches:

   ```bash
   python plot_results.py Output/Analysis_1
   python plot_results.py -h   # all options
   ```

PROLITE was developed as part of the Master's thesis:
*F. Casucci, Development and on-sky validation of PROLITE: a hybrid tool for off-axis PSF modelling in single-conjugate adaptive optics, University of Pisa, 2026.*
