#!/usr/bin/env python3
"""
analyze_ao.py -- Post-processing for AO Cn2/NCPA estimation runs.

Compares the measured PSF field (FWHM, flux) against the model reconstructed
from an estimator's best-fit solution, for one or several estimation runs,
and writes LaTeX tables + plots summarizing the residuals, parameters and
parameter correlations.

Star positions are not hardcoded: each run's save_status.ini should point to
the star-position files (in [Regions] or [Project]):
    fpath-all-pos = path/to/all_pos_field.txt   2-column (x, y), every star
    fpath-ao-pos  = path/to/ao_pos.txt          2-column (x, y), the AO star
and optionally, either as an inline list or a file:
    avoid-stars = [0, 1, 7, ...]                indices to exclude, inline
    avoid-stars-fpath = path/to/avoid.txt       indices to exclude, one per line
These can also be passed/overridden on the command line (see -h).

Mode (single analysis / one project / several branches to compare) is
auto-detected from the directory structure -- pass --mode to override:
    .../Branch/Analysis/     -> 'analysis': that one estimation run
    .../Branch/              -> 'single':   its numbered Analysis_N run folders
    .../Parents/             -> 'branches': each Branch's numbered run folders,
                                 plus a comparison across branches
Detection works by looking for save_status.ini itself, then one level down,
then two levels down, below whatever path is given.

Usage (auto-detected mode, all default plots)
-------------------------------------------
    python analyze_ao.py Output/PalomarJ_run/

Usage (only recompute the FWHM/flux plots)
--------------------------------------------------------------------------
    python analyze_ao.py Output/PalomarJ_run/ --plots fwhm flux

Usage (forcing branch mode: several parent folders, each with several runs)
-----------------------------------------------------------------------------
    python analyze_ao.py Output/AllBranches/ --mode branches

Usage (field paths not yet in save_status.ini)
-----------------------------------------------------------------------------
    python analyze_ao.py Output/PalomarJ_run/ --all-pos-fpath star_finder/all_pos_field.txt \
                                                --ao-pos-fpath star_finder/ao_pos.txt

Run `python analyze_ao.py -h` for the full option list.
"""
import argparse
from math import floor, log10
from types import SimpleNamespace
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import files as ff
import format_and_control as fcu
from graphics import Graphics
from astropy.io import fits
from sensorPosition import SensorPosition
from dataRegions import dataRegions
from merit import Merit

plt.rcParams['text.usetex'] = True

Z_NAME_LIST = ['Tip', 'Tilt', 'Defocus', 'Astigmastism sin', 'Astigmastism cos',
               'Coma sin', 'Coma cos', 'Trefoil sin', 'Trefoil cos', 'Spherical']

# The estimator fits NCPA as 6 physical groups, not as 10 independent
# Zernike coefficients: two orthogonal terms of a group (e.g. astigmatism
# sin/cos) are fitted together as one (amplitude, phase) pair, while
# Defocus and Spherical are single signed scalars. 'max-ncpa-vec' in
# save_status.ini has one entry per GROUP (in this order) -- a zero entry
# means that group was not fitted -- and the estimator's 'bounds' list
# gives one bound per fitted group's free parameter(s): 1 for a scalar
# group, 2 (amplitude, then phase) for a pair group. This mapping is
# inferred from the bounds structure (matching e.g. a signed [-60, 60]
# defocus bound, a [0, 180] astigmatism phase bound reflecting its 180-deg
# symmetry, [0, 360] for coma, [0, 120] for trefoil's 3-fold symmetry) --
# double check against your estimator's actual parametrization if in doubt.
NCPA_GROUPS = [
    ('Tip-Tilt', (0, 1), 'pair'),
    ('Defocus', (2,), 'scalar'),
    ('Astigmatism', (3, 4), 'pair'),
    ('Coma', (5, 6), 'pair'),
    ('Trefoil', (7, 8), 'pair'),
    ('Spherical', (9,), 'scalar'),
]

DEFAULT_COLORS = [
    "#000000", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
    "#F0E442", "#004488", "#EE7733", "#228833", "#AA3377", "#9B9A9A", "#332288",
    "#88CCEE", "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#117733",
    "#DDCC77", "#CC6677", "#AA4499", "#DDDDDD", "#222255", "#55A868", "#C44E52",
    "#8172B3", "#937860",
]

PLOT_CHOICES = ['regions', 'fwhm', 'flux', 'merit', 'field', 'all', 'none']


# ===========================================================================
#  LaTeX / formatting helpers
# ===========================================================================

def escape_underscores_outside_math(s):
    """
    Escape underscores in `s` for LaTeX, but leave anything inside $...$
    math segments untouched (so '$\\phi_\\mathrm{wind}$' keeps its '_',
    while 'layer_0' outside math becomes 'layer\\_0').
    """
    s = str(s)
    parts = s.split('$')
    for i in range(0, len(parts), 2):  # even indices are outside math
        parts[i] = parts[i].replace('_', '\\_')
    return '$'.join(parts)


def sig_format(value, sig=3):
    """
    Format a number to `sig` significant digits: no scientific notation for
    ordinary magnitudes, no unnecessary trailing zeros. Falls back to
    scientific notation only for very small/large magnitudes, and passes
    non-numeric values through unchanged.
    """
    if value is None:
        return ''
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return str(val)
    if val == 0:
        return '0'
    exponent = int(floor(log10(abs(val))))
    if exponent > 5 or exponent < -4:
        return f"{val:.{max(sig - 1, 0)}e}"
    decimals = max(sig - exponent - 1, 0)
    formatted = f"{val:.{decimals}f}"
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    return formatted if formatted not in ('', '-', '-0') else '0'


def sig_format_pair(lo, hi, sig=3):
    """Format a [lo, hi] bound pair as 'lo, hi' with shared significant digits."""
    return f"{sig_format(lo, sig)}, {sig_format(hi, sig)}"


def format_val_pm_err(val, err, cap_decimals=2, fallback_decimals=3):
    """
    Format 'value $\\pm$ error' with a number of decimals set by the first
    significant digit of the error (no point printing 3 decimals if the
    error is already uncertain at the 1st or 2nd decimal). Capped at
    cap_decimals (default 2), floored at 1. If err is not usable, fall back
    to a plain value with fallback_decimals.
    """
    if val is None or np.isnan(val):
        return ''
    if err is None or not np.isfinite(err) or err <= 0:
        return '%.*f' % (fallback_decimals, val)
    exponent = int(floor(log10(abs(err))))
    decimals = max(1, -exponent)
    decimals = min(decimals, cap_decimals)
    return '%.*f $\\pm$ %.*f' % (decimals, val, decimals, err)


def write_latex_table(header, rows, filename, col_format=None, hline_after=None, sig=3):
    """
    Single generic LaTeX table writer used everywhere in this module.

    header: list of column-header strings, OR a raw LaTeX header string
        (col_format must be given explicitly in that case, since the number
        of columns can't be inferred from raw LaTeX).
    rows: list of row lists. Numeric cells are formatted to `sig`
        significant digits; string cells are passed through underscore
        escaping (outside any $...$ math segments).
    hline_after: set/list of 0-based row indices after which an \\hline is
        inserted (e.g. to mark a section boundary), independent of \\hline
        already placed once right after the header.
    """
    if isinstance(header, str):
        if col_format is None:
            raise ValueError("col_format must be given when header is raw LaTeX.")
        header_block = header if header.endswith('\n') else header + '\n'
    else:
        header = [escape_underscores_outside_math(h) for h in header]
        ncols = len(header)
        if col_format is None:
            col_format = 'l' + 'c' * (ncols - 1)
        header_block = " & ".join('\\textbf{' + h + '}' for h in header) + " \\\\\n"

    hline_after = set(hline_after or [])
    formatted_rows = []
    for row in rows:
        formatted_row = []
        for cell in row:
            if isinstance(cell, str):
                formatted_row.append(escape_underscores_outside_math(cell))
            else:
                formatted_row.append(escape_underscores_outside_math(sig_format(cell, sig)))
        formatted_rows.append(formatted_row)

    with open(filename, 'w') as f:
        f.write("\\centering\n")
        f.write("\\begin{tabular}{" + col_format + "}\n")
        f.write(header_block)
        f.write("\\hline\n")
        for i, row in enumerate(formatted_rows):
            f.write(" & ".join(row) + " \\\\\n")
            if i in hline_after:
                f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def write_latex_table_chunked(header, rows, filename_base, col_format=None, hline_after=None,
                               sig=3, max_rows=25):
    """
    Like write_latex_table, but splits `rows` into files of at most
    `max_rows` rows each -- for long, per-star tables meant to be printed
    landscape on A4, a single file with hundreds of rows isn't usable.
    filename_base is used as-is (with '.txt' appended) when everything fits
    in one file; otherwise as 'filename_base_partN.txt' for each chunk.
    Returns the list of file paths written.
    """
    if len(rows) <= max_rows:
        path = filename_base + '.txt'
        write_latex_table(header, rows, path, col_format=col_format, hline_after=hline_after, sig=sig)
        return [path]

    n_chunks = -(-len(rows) // max_rows)  # ceil division
    paths = []
    for c in range(n_chunks):
        chunk = rows[c * max_rows:(c + 1) * max_rows]
        # hline_after indices are relative to the whole table; keep only the
        # ones that fall inside this chunk, and shift them accordingly.
        chunk_hlines = {h - c * max_rows for h in (hline_after or []) if c * max_rows <= h < (c + 1) * max_rows}
        path = '%s_part%d.txt' % (filename_base, c + 1)
        write_latex_table(header, chunk, path, col_format=col_format, hline_after=chunk_hlines, sig=sig)
        paths.append(path)
    return paths


def latex_label_for_key(key):
    """Map an internal parameter-dict key to a compact LaTeX math label."""
    if key.startswith('Wind dir. L') or key.startswith('Wind vel. L'):
        kind, layer = ('phi_{\\mathrm{wind,\\,L%s}}', key.split('L')[-1]) if 'dir' in key \
            else ('v_{\\mathrm{wind,\\,L%s}}', key.split('L')[-1])
        return '$\\%s$' % (kind % layer) if 'phi' in kind else '$%s$' % (kind % layer)
    if key.startswith('Cn2 weight L'):
        layer = key.split('L')[-1]
        return r'$C_{n,\,\mathrm{L%s}}^2$' % layer
    if key == 'Seeing':
        return r'$\mathrm{seeing}$'
    if key == 'Jitter rot.':
        return r'$\phi_\mathrm{jit}$'
    if key.endswith(' phase'):
        base = key[:-len(' phase')]
        short_map = {'Astigmatism': 'ast', 'Coma': 'com', 'Trefoil': 'tref', 'Tip-Tilt': 'tt'}
        short = short_map.get(base, base.lower()[:3])
        return r'$\phi_\mathrm{%s}$' % short
    return key


# ===========================================================================
#  Star-position file loaders
# ===========================================================================



def find_ini_key(status, *key_names):
    """
    Look up any of `key_names` across every [section] of a parsed ini dict,
    returning the first match found (searching [Regions] and [Project]
    first, since that's where these paths are expected, then any other
    section). Returns None if none of the keys are present anywhere.
    """
    preferred = ['Regions', 'Project']
    sections = preferred + [s for s in status if s not in preferred]
    for section in sections:
        cfg = status.get(section)
        if not isinstance(cfg, dict):
            continue
        for key in key_names:
            if key in cfg:
                return cfg[key]
    return None


def load_positions_file(fpath):
    """Load a whitespace-separated 2-column (x, y) star-position text file."""
    pos = np.loadtxt(fpath)
    return np.atleast_2d(pos)


def load_avoid_indices_file(fpath):
    """
    Load an optional file listing star indices to exclude from both reference
    and control stars (whitespace/newline-separated integers, one or many per
    line). Returns an empty array if `fpath` is None or the file is empty.
    """
    if fpath is None:
        return np.array([], dtype=int)
    try:
        return np.atleast_1d(np.loadtxt(fpath, dtype=int))
    except Exception:
        return np.array([], dtype=int)


# ===========================================================================
#  Per-run data loading
# ===========================================================================

class RunData:
    """
    The setup (save_status.ini), the optimizer's parameter/merit history, and
    the best-fit solution for a single estimation run.

    save_status.ini may optionally define (searched across all sections,
    [Regions]/[Project] first):
        fpath-all-pos = <path>       2-column (x, y) file with every star in the field
        fpath-ao-pos  = <path>       2-column (x, y) file with the AO star position
        avoid-stars = [i, j, ...]    inline list of star indices to exclude
        avoid-stars-fpath = <path>   or the same, as a whitespace/newline-separated file
    `avoid-stars` (inline) takes priority over `avoid-stars-fpath` if both are
    given. Avoided indices are removed from both the reference stars
    (select-stars) and the control stars (everything else); see
    get_star_selection(). These are only used when the caller doesn't
    already supply avoid_indices,
    and/or when the field geometry is auto-detected (i.e. when AOAnalysis is
    created without explicit all_pos_fpath/ao_pos_fpath).
    """

    def __init__(self, estimator_outdir, save_status_ftag, avoid_indices=None):
        self.save_status_fpath = os.path.join(estimator_outdir, save_status_ftag + '.ini')
        self.status = ff.read_ini(self.save_status_fpath)
        self.status_estimator = self.status['Estimator']
        self.radii = self.status_estimator['radii']
        self.Cn2Heights = self.status_estimator['cn2Heights']
        self.data_ini_filepath = self.status_estimator['data-ini-fpath']
        # PROLITE run configuration, for the per-analysis summary table.
        self.estimator_type = self.status_estimator.get('estimator-type')
        self.psf_fit_type_cfg = self.status_estimator.get('psf-fit-type')
        self.merit_shape = self.status_estimator.get('merit-shape')
        self.data_fpath = self.status['Project']['data-fpath']
        self.regions_fpath = self.status['Regions']['regions-fpath']
        self.box_edge = self.status['Regions']['box_edge']
        self.bounds = np.array(self.status_estimator['bounds'])
        self.max_abberation_vec = self.status_estimator['max-ncpa-vec']
        self.max_jitter_vec = self.status_estimator.get('max-jitter-vec')
        self.do_jitter = self.max_jitter_vec is not None
        self.rng_seed = self.status_estimator.get('rng_seed')
        self.sel_stars = self.status['Project'].get('select-stars')

        self.all_pos_fpath = find_ini_key(self.status, 'fpath-all-pos', 'all-pos-fpath')
        self.ao_pos_fpath = find_ini_key(self.status, 'fpath-ao-pos', 'ao-pos-fpath')
        avoid_inline = find_ini_key(self.status, 'avoid-stars')
        avoid_fpath = find_ini_key(self.status, 'avoid-stars-fpath', 'fpath-avoid-stars')
        if avoid_indices is not None:
            self.avoid_indices = np.asarray(avoid_indices, dtype=int)
        elif avoid_inline is not None:
            self.avoid_indices = np.asarray(avoid_inline, dtype=int)
        else:
            self.avoid_indices = load_avoid_indices_file(avoid_fpath)
        self._warned_oob = False  # print the out-of-bounds warning (if any) only once per run

        param_res_ftag = self.status_estimator['cycle-status-ftag']
        self.param_res_fpath = os.path.join(estimator_outdir, param_res_ftag + '.txt')
        cycle_res = np.loadtxt(self.param_res_fpath, comments='#')
        self.xv_gens = cycle_res[:, 1:-1]
        self.gens_num = cycle_res[:, 0]
        self.merits_gens = cycle_res[:, -1]

        if self.merits_gens[-1] < 0:
            # pyGAD fitness (negative, cubed)
            self.merit = (-self.merits_gens[-1]) ** (1 / 3)
            best_x = self.xv_gens[-1]
        else:
            # plain merit (DE / minimize)
            best_idx = np.argmin(self.merits_gens)
            self.merit = self.merits_gens[best_idx]
            best_x = self.xv_gens[best_idx]
        self.est_parm_dic = fcu._unpack_param(best_x, self.Cn2Heights, self.max_abberation_vec, self.do_jitter)

        self.tel_res = 2048
        self.pixscale = 0.015
        run_time = self.status_estimator.get('run-time')
        self.run_time = run_time / 3600 if run_time is not None else np.nan
        self.n_iterations = self.gens_num[-1] if self.gens_num.size > 0 else np.nan

    def get_star_selection(self, n_total_stars):
        """
        Reference (estimation) vs control (comparison) star indices. All
        stars are reference (no control stars) when no selection was saved.
        select-stars/avoid-stars indices at or beyond n_total_stars (e.g. a
        list generated for a larger field than this particular run's) are
        dropped before use, rather than raising -- np.delete would error on
        an out-of-bounds index.
        """
        if self.sel_stars is None:
            return np.arange(n_total_stars), np.array([], dtype=int)
        select_stars = np.asarray(self.sel_stars, np.int32)
        avoid_indices = np.asarray(self.avoid_indices, np.int32)

        select_valid = select_stars[(select_stars >= 0) & (select_stars < n_total_stars)]
        avoid_valid = avoid_indices[(avoid_indices >= 0) & (avoid_indices < n_total_stars)]
        n_dropped = (select_stars.size - select_valid.size) + (avoid_indices.size - avoid_valid.size)
        if n_dropped and not self._warned_oob:
            print('  Warning: dropped %d select-star/avoid-star index(es) outside '
                  'the field (0..%d)' % (n_dropped, n_total_stars - 1))
            self._warned_oob = True

        remove = np.concatenate((select_valid, avoid_valid))
        remaining = np.delete(np.arange(n_total_stars), remove, axis=0)
        return select_valid, remaining

    def fitted_ncpa_groups(self):
        """Names of the NCPA groups (see NCPA_GROUPS) that were actually
        fitted, i.e. have a nonzero max amplitude in max-ncpa-vec and
        therefore bounds entries."""
        max_vec = np.atleast_1d(self.max_abberation_vec)
        return [name for (name, _, _), m in zip(NCPA_GROUPS, max_vec) if m != 0]


class TempDataRegion:
    def __init__(self, data, star_pos, box_edge):
        self.data_regions = data
        self.star_pos = star_pos
        self.box_edge = box_edge


def detect_mode(path, save_status_ftag='save_status', outplot_dir_prefix='Outplot'):
    """
    Auto-detect which processing mode fits a given directory, by exploring
    downward from it until a '<save_status_ftag>.ini' is found:
      - path itself has it                -> 'analysis' (one estimation run;
                                               .../Branch/Analysis/)
      - a subfolder of path has it        -> 'single'   (one project/branch
                                               of numbered run folders;
                                               .../Branch/)
      - a subfolder of a subfolder has it -> 'branches' (several projects/
                                               branches to compare;
                                               .../Parents/)
    Raises ValueError if `path` isn't a directory, or if no ini is found
    within two levels below it.
    """
    ini_name = save_status_ftag + '.ini'
    if not os.path.isdir(path):
        raise ValueError("Not a directory: %s" % path)
    if os.path.isfile(os.path.join(path, ini_name)):
        return 'analysis'

    def subdirs(p):
        try:
            return [e.path for e in os.scandir(p) if e.is_dir() and not e.name.startswith(outplot_dir_prefix)]
        except (FileNotFoundError, NotADirectoryError):
            return []

    level1 = subdirs(path)
    if any(os.path.isfile(os.path.join(d, ini_name)) for d in level1):
        return 'single'

    for d in level1:
        if any(os.path.isfile(os.path.join(dd, ini_name)) for dd in subdirs(d)):
            return 'branches'

    raise ValueError(
        "Could not find any '%s' within two levels below %s -- pass --mode "
        "explicitly, or double check the path." % (ini_name, path))


# ===========================================================================
#  Main pipeline
# ===========================================================================

class AOAnalysis:
    """
    All the processing steps for one AO project: computing per-run merits
    (FWHM / flux residuals w.r.t. the model), building LaTeX summary and
    correlation tables, and looping over runs (single project) or branches
    (several projects to compare against each other).

    All plotting is delegated to a `Graphics` instance.
    """

    OUTPLOT_PREFIX = 'Outplot'  # any folder starting with this is treated as an
                                 # output folder and excluded from subfolder scans,
                                 # regardless of which threshold (if any) it's for.

    @staticmethod
    def _format_threshold_for_dirname(value):
        """'10000' for 10000.0, '12345p6' for 12345.6 (dot-free, filesystem-friendly)."""
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return ('%g' % value).replace('.', 'p')

    def __init__(self, project_dir, psf_fit_type='lmsq',
                 save_status_ftag='save_status', plots=('all',), showflag=False, colors=None,
                 all_pos_fpath=None, ao_pos_fpath=None, avoid_fpath=None, sim_dic=None,
                 flux_threshold=None):
        self.project_dir = project_dir
        self.psf_fit_type = psf_fit_type
        self.save_status_ftag = save_status_ftag
        self.showflag = showflag
        self.colors = colors or DEFAULT_COLORS

        # Field geometry (every star's position, the AO star's position) is
        # resolved lazily from the first run's save_status.ini -- or the
        # --all-pos-fpath/--ao-pos-fpath overrides below, which take
        # priority -- see _resolve_field().
        self._all_pos_override = all_pos_fpath
        self._ao_pos_override = ao_pos_fpath
        self.AO_pos = None
        self.all_pos_field = None
        self.avoid_indices = load_avoid_indices_file(avoid_fpath) if avoid_fpath else None
        self.sim_dic = sim_dic  # optional known ground truth, e.g. for a simulated dataset

        # Optional flux threshold for CONTROL stars: every quantity computed
        # and plotted for control stars (RMS, tables, plots) considers only
        # the ones whose fitted flux is >= this value; reference stars are
        # never filtered. Since flux isn't known until the Merit fit has
        # actually run, this can't skip the fit itself -- only what's
        # reported/plotted afterward -- see _apply_flux_threshold(). When
        # set, results go to a distinctly-named output folder instead of
        # the default, so filtered and unfiltered runs never collide.
        self.flux_threshold = flux_threshold
        if flux_threshold is not None:
            self.OUTPLOT_DIR = '%s_flux_%s' % (
                self.OUTPLOT_PREFIX, self._format_threshold_for_dirname(flux_threshold))
        else:
            self.OUTPLOT_DIR = self.OUTPLOT_PREFIX

        plots = set(plots)
        self.do_regions = 'all' in plots or 'regions' in plots
        self.do_fwhm = 'all' in plots or 'fwhm' in plots
        self.do_flux = 'all' in plots or 'flux' in plots
        self.do_merit = 'all' in plots or 'merit' in plots
        self.do_field = 'all' in plots or 'field' in plots

        self.graphics = Graphics(showflag=showflag)

    # ------------------------------------------------------------------
    #  Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def polar_distance(zen_a, azi_a, zen_b, azi_b, degrees=False):
        zen_a, azi_a, zen_b, azi_b = (np.asarray(x) for x in (zen_a, azi_a, zen_b, azi_b))
        if degrees:
            azi_a, azi_b = np.deg2rad(azi_a), np.deg2rad(azi_b)
        return np.sqrt(zen_a ** 2 + zen_b ** 2 - 2 * zen_a * zen_b * np.cos(azi_a - azi_b))

    def star_distances(self, star_pos, tel_res=2048, pixscale=0.01495):
        sensor = SensorPosition(tel_res, pixscale)
        zen, azi = sensor.px2polar(star_pos)
        AO_zen, AO_azi = sensor.px2polar(self.AO_pos)
        return self.polar_distance(zen, azi, AO_zen, AO_azi, degrees=True)

    def _resolve_field(self, run):
        """
        Resolve the field geometry (every star's position, and the AO star's
        position) for THIS run, from the paths given in its own
        save_status.ini -- or the CLI overrides, if given, which take
        priority. Re-read for every run (no caching across analyses), since
        each Analysis_N folder carries its own record of which star files it
        used, and different runs/branches can point at different fields.
        """
        all_pos_fpath = self._all_pos_override or run.all_pos_fpath
        if all_pos_fpath is None:
            raise ValueError(
                "Need the path to a 2-column (x, y) file listing every star "
                "in the field. Add 'fpath-all-pos = <path>' to %s "
                "(in [Regions] or [Project]), or pass --all-pos-fpath." % run.save_status_fpath)
        self.all_pos_field = load_positions_file(all_pos_fpath)

        ao_pos_fpath = self._ao_pos_override or run.ao_pos_fpath
        self.AO_pos = load_positions_file(ao_pos_fpath) if ao_pos_fpath else None
        if self.AO_pos is None:
            print("Warning: no AO star position found for %s (add 'fpath-ao-pos' "
                  "to save_status.ini or pass --ao-pos-fpath) -- distance-from-AO "
                  "quantities will be unavailable." % run.save_status_fpath)

        select_stars, remaining = run.get_star_selection(self.all_pos_field.shape[0])
        print('  Field: %d reference, %d control star(s) (from %s)' % (
            len(select_stars), len(remaining), all_pos_fpath))

    def compute_dist(self, run):
        select_stars, remaining = run.get_star_selection(self.all_pos_field.shape[0])
        dist_ref = self.star_distances(self.all_pos_field[select_stars])
        if remaining.size > 0:
            dist_ctrl = self.star_distances(self.all_pos_field[remaining])
        else:
            dist_ctrl = np.array([np.nan])
        return dist_ref, dist_ctrl

    # ------------------------------------------------------------------
    #  Per-run merit computation (FWHM / flux residuals vs. the model)
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_star_arrays(select_stars, remaining_indices, ref_vals, ctrl_vals, n_total):
        """Merge per-star values from reference + control stars into one
        (n_total,) array indexed by original star index; uncovered stars are NaN."""
        combined = np.full(n_total, np.nan)
        if len(select_stars) > 0:
            combined[select_stars] = np.asarray(ref_vals)
        if len(remaining_indices) > 0:
            combined[remaining_indices] = np.asarray(ctrl_vals)
        return combined

    @staticmethod
    def _convert_std_vec(std_vec, pixscale):
        std_vec[:, 0] *= pixscale * 1e3
        std_vec[:, 1] *= pixscale * 1e3
        return std_vec

    def compute_run_merits(self, run, estimator_output_dir, saveflag_estimation=True):
        """Run the Merit estimator for reference + (if any) control stars.
        Returns a SimpleNamespace with all per-star results for this run."""
        select_stars, remaining = run.get_star_selection(self.all_pos_field.shape[0])
        has_control = remaining.size > 0
        sensor = SensorPosition(run.tel_res, run.pixscale)

        ref = self._run_merit_for_stars(
            run, estimator_output_dir, select_stars, sensor,
            outdir_name='Reference_stars', ini_name='region_data_config.ini',
            ini_tag='temp_config_reg', sim_tag='temp_TIPTOP_sim_output_reg',
            saveflag=saveflag_estimation)

        if has_control:
            ctrl = self._run_merit_for_stars(
                run, estimator_output_dir, remaining, sensor,
                outdir_name='Control_stars', ini_name='rem_data_config.ini',
                ini_tag='temp_config_rem', sim_tag='temp_TIPTOP_sim_output_rem',
                saveflag=saveflag_estimation, in_temp_subdir=True)
            regions_mf_fpath = os.path.join(estimator_output_dir, 'Control_stars', 'merit_for_remaining_stars.txt')
            np.savetxt(regions_mf_fpath, ctrl['posteriors'], fmt='%.5f', delimiter='\t', newline='\n')
        else:
            shape = ref['data_regions'].data_regions.shape
            ctrl = dict(
                data_regions=None, radii=np.array([]),
                rec_regions=np.full((1,) + shape[1:], np.nan),
                map_regions=np.empty((0,) + shape[1:]),
                std_vec=np.full((1, 3), np.nan), rec_std_vec=np.full((1, 3), np.nan),
                err_flux=np.array([]), tot_flux=np.array([]), posteriors=np.array([]),
            )

        result = SimpleNamespace(
            has_control_stars=has_control,
            select_stars=select_stars, remaining_indices=remaining,
            radii=ref['radii'],
            data_regions=ref['data_regions'], rec_data_regions=ref['rec_regions'], map_regions=ref['map_regions'],
            rem_data_regions=ctrl['data_regions'], rem_radii=ctrl['radii'],
            rec_remain_regions=ctrl['rec_regions'], rem_map_regions=ctrl['map_regions'],
            data_std_vec=self._convert_std_vec(ref['std_vec'], run.pixscale),
            reconstruct_std_vec=self._convert_std_vec(ref['rec_std_vec'], run.pixscale),
            data_remain_std_vec=self._convert_std_vec(ctrl['std_vec'], run.pixscale),
            reconstruct_remain_std_vec=self._convert_std_vec(ctrl['rec_std_vec'], run.pixscale),
            err_flux_ref=ref['err_flux'], tot_flux_ref=ref['tot_flux'],
            err_flux_ctrl=ctrl['err_flux'], tot_flux_ctrl=ctrl['tot_flux'],
            posteriors_regions=ref['posteriors'], rem_posteriors_regions=ctrl['posteriors'],
        )
        return result

    def _run_merit_for_stars(self, run, estimator_output_dir, star_idx, sensor,
                              outdir_name, ini_name, ini_tag, sim_tag, saveflag, in_temp_subdir=False):
        star_pos = self.all_pos_field[star_idx]
        star_zen, star_azi = sensor.px2polar(star_pos)
        outdir = os.path.join(estimator_output_dir, outdir_name)
        os.makedirs(outdir, exist_ok=True)
        work_dir = os.path.join(outdir, 'Temp') if in_temp_subdir else outdir
        if in_temp_subdir:
            os.makedirs(work_dir, exist_ok=True)
        ini_fpath = os.path.join(work_dir, ini_name)

        if run.data_fpath is None:
            data = np.load(run.regions_fpath)
            data_regions = TempDataRegion(data, star_pos, run.box_edge)
            radii = run.radii
        else:
            with fits.open(run.data_fpath) as hdul:
                data = hdul[0].data
            if data.ndim > 2:
                data = data[0]
            radii = fcu._format_radii(run.radii[0], star_pos.shape[0]) if in_temp_subdir else run.radii
            data_regions = dataRegions(data, star_pos, run.box_edge)

        star_param = [
            ("sources_science", "Zenith", np.array2string(star_zen, separator=", ", precision=8).replace('\n', '')),
            ("sources_science", "Azimuth", np.array2string(star_azi, separator=", ", precision=8).replace('\n', '')),
        ]
        ff.write_ini(file_in=run.data_ini_filepath, file_out=ini_fpath, param_dic=star_param)

        merit = Merit(data_regions.data_regions, radii, outdir, work_dir, ini_tag, sim_tag, ini_fpath,
                       psf_fit_type=self.psf_fit_type, stars_idx=star_idx, do_err_flux=True)
        std_vec = merit.data_fwhm_vec
        _, posteriors, map_regions, rec_std_vec, rec_regions, err_flux, tot_flux = merit.make_estimation(
            run.est_parm_dic, show_estimators=self.showflag, save_estimators=saveflag,
            show_regions=False, save_regions=False, gaussResults=True)
        merit.close_pool()

        return dict(radii=radii, data_regions=data_regions, rec_regions=rec_regions, map_regions=map_regions,
                    std_vec=std_vec, rec_std_vec=rec_std_vec, err_flux=err_flux, tot_flux=tot_flux,
                    posteriors=posteriors)

    # ------------------------------------------------------------------
    #  RMS residual summaries
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_fwhm_rows(arr):
        arr = np.atleast_2d(arr)
        return arr[~np.isnan(arr).any(axis=1)]

    @classmethod
    def rms_relative_fwhm_single(cls, data_vec, reconstruct_vec):
        """RMS of the relative FWHM residual, separately for major/minor
        axis, for a single set of stars (reference-only or control-only)."""
        d, r = cls._valid_fwhm_rows(data_vec), cls._valid_fwhm_rows(reconstruct_vec)
        if d.shape[0] == 0:
            return np.nan, np.nan
        rel = (d - r) / d
        return (np.sqrt(np.nanmean(rel[:, 0] ** 2)), np.sqrt(np.nanmean(rel[:, 1] ** 2)))

    @classmethod
    def rms_relative_fwhm(cls, data_std_vec, reconstruct_std_vec, data_remain_std_vec, reconstruct_remain_std_vec):
        """RMS of the relative FWHM residual, separately for major/minor axis,
        combining reference + control stars (dropping the NaN placeholder row)."""
        d_ref, r_ref = cls._valid_fwhm_rows(data_std_vec), cls._valid_fwhm_rows(reconstruct_std_vec)
        d_ctrl, r_ctrl = cls._valid_fwhm_rows(data_remain_std_vec), cls._valid_fwhm_rows(reconstruct_remain_std_vec)
        d = np.vstack([d_ref, d_ctrl]) if d_ctrl.shape[0] else d_ref
        r = np.vstack([r_ref, r_ctrl]) if r_ctrl.shape[0] else r_ref
        rel = (d - r) / d
        return (np.sqrt(np.nanmean(rel[:, 0] ** 2)), np.sqrt(np.nanmean(rel[:, 1] ** 2)))

    def combined_err_flux(self, result, n_total):
        return self._combine_star_arrays(result.select_stars, result.remaining_indices,
                                          result.err_flux_ref, result.err_flux_ctrl, n_total)

    def _apply_flux_threshold(self, result, dist_ctrl):
        """
        If self.flux_threshold is set, keep only the control stars whose
        fitted flux is >= the threshold -- every RMS/table/plot downstream
        of this call then only ever sees those stars. Reference stars are
        never filtered. This can't skip the Merit fit itself (flux isn't
        known until it has run), only what's reported afterward.
        """
        if self.flux_threshold is None or not result.has_control_stars:
            return result, dist_ctrl

        mask = result.tot_flux_ctrl >= self.flux_threshold
        if mask.all():
            return result, dist_ctrl

        print('  Flux threshold >= %g: keeping %d/%d control star(s)' %
              (self.flux_threshold, int(mask.sum()), mask.size))

        result.remaining_indices = result.remaining_indices[mask]
        result.data_remain_std_vec = result.data_remain_std_vec[mask]
        result.reconstruct_remain_std_vec = result.reconstruct_remain_std_vec[mask]
        result.err_flux_ctrl = result.err_flux_ctrl[mask]
        result.tot_flux_ctrl = result.tot_flux_ctrl[mask]
        if result.rem_posteriors_regions.size:
            result.rem_posteriors_regions = result.rem_posteriors_regions[mask]
        if result.rem_data_regions is not None:
            result.rem_data_regions.data_regions = result.rem_data_regions.data_regions[mask]
        if result.rec_remain_regions.shape[0] == mask.size:
            result.rec_remain_regions = result.rec_remain_regions[mask]
        if result.rem_map_regions.shape[0] == mask.size:
            result.rem_map_regions = result.rem_map_regions[mask]
        rem_radii_arr = np.asarray(result.rem_radii)
        if rem_radii_arr.ndim and rem_radii_arr.shape[0] == mask.size:
            result.rem_radii = rem_radii_arr[mask]

        result.has_control_stars = result.remaining_indices.size > 0
        if dist_ctrl is not None and len(dist_ctrl) == mask.size:
            dist_ctrl = dist_ctrl[mask]
        if not result.has_control_stars:
            dist_ctrl = np.array([np.nan])
        return result, dist_ctrl

    # ------------------------------------------------------------------
    #  Parameter tables: a per-analysis comparison table, and a separate
    #  mean/std/boundaries summary table (NCPA shown as raw Zernike
    #  coefficients, with bounds derived from the estimator's actual
    #  amplitude/phase-per-group parametrization -- see NCPA_GROUPS)
    # ------------------------------------------------------------------

    @staticmethod
    def _display_dict(parm_dic, fitted_groups, seed=None, merit=None):
        """
        Flatten a raw parameter dict into display quantities, one entry per
        table row: seeing, per-layer cn2/wind, each raw Zernike coefficient
        belonging to a fitted NCPA group, jitter, plus optional seed/merit
        pseudo-rows.
        """
        d = {}
        if seed is not None:
            d['__seed__'] = seed
        d['seeing'] = parm_dic['seeing']
        for ii, hei in enumerate(parm_dic['cn2hei']):
            d['layer%d_height' % ii] = hei
            d['layer%d_weight' % ii] = parm_dic['cn2wei'][ii]
            d['layer%d_vel' % ii] = parm_dic['winvel'][ii]
            d['layer%d_dir' % ii] = parm_dic['windir'][ii]
        if 'zCoefStaticOn' in parm_dic:
            zc = parm_dic['zCoefStaticOn']
            for name, idxs, kind in NCPA_GROUPS:
                if name not in fitted_groups:
                    continue
                for zi in idxs:
                    d['zc_%d' % zi] = zc[zi]
        if 'jitter' in parm_dic:
            d['jitter_major'], d['jitter_minor'], d['jitter_rot'] = parm_dic['jitter']
        if merit is not None:
            d['__merit__'] = merit
        return d

    @staticmethod
    def _row_defs(n_layers, fitted_groups, has_jitter, has_seed, has_merit):
        """Ordered (key, row_label) pairs matching _display_dict's keys."""
        defs = []
        if has_seed:
            defs.append(('__seed__', 'rng-seed'))
        defs.append(('seeing', 'seeing [arcsec]'))
        for ii in range(n_layers):
            defs.append(('layer%d_height' % ii, 'layer %d height [m]' % ii))
            defs.append(('layer%d_weight' % ii, 'layer %d cn2 weight' % ii))
            defs.append(('layer%d_vel' % ii, 'layer %d wind velocity [m/s]' % ii))
            defs.append(('layer%d_dir' % ii, 'layer %d wind direction [deg]' % ii))
        for name, idxs, kind in NCPA_GROUPS:
            if name not in fitted_groups:
                continue
            for zi in idxs:
                defs.append(('zc_%d' % zi, 'Z%d %s [nm]' % (zi + 2, Z_NAME_LIST[zi])))
        if has_jitter:
            defs.append(('jitter_major', 'Jitter FWHM major [mas]'))
            defs.append(('jitter_minor', 'Jitter FWHM minor [mas]'))
            defs.append(('jitter_rot', 'Jitter rotation [rad]'))
        if has_merit:
            defs.append(('__merit__', 'Merit Function'))
        return defs

    @staticmethod
    def _bounds_for_display(bounds, n_layers, fitted_groups, has_jitter):
        """
        {display_key: [lo, hi]} aligned with _row_defs' keys, matching the
        estimator's actual bounds layout: seeing, then per-layer cn2
        weight/wind velocity/wind direction, then one bound per fitted NCPA
        group's free parameter(s) -- 1 (signed amplitude) for a scalar group
        (Defocus, Spherical), 2 (amplitude, then phase) for a pair group
        (Tip-Tilt, Astigmatism, Coma, Trefoil) -- then jitter (last 3).

        A pair group's amplitude/phase bounds don't translate into an exact
        rectangular bound on its two raw (sin, cos) coefficients -- only
        the amplitude bound does, as a symmetric envelope [-A, A] each
        coefficient individually can't exceed (the phase bound further
        restricts the *combination*, but that can't be shown as a single
        range per coefficient).
        """
        b = {'seeing': bounds[0]}
        idx = 1
        for ii in range(n_layers):
            if ii < n_layers - 1:
                b['layer%d_weight' % ii] = bounds[idx]; idx += 1
            b['layer%d_vel' % ii] = bounds[idx]; idx += 1
            b['layer%d_dir' % ii] = bounds[idx]; idx += 1
        for name, idxs, kind in NCPA_GROUPS:
            if name not in fitted_groups:
                continue
            if kind == 'pair':
                amp_bound = bounds[idx]; idx += 1
                idx += 1  # phase bound: consumed (keeps offset correct) but not directly displayable per coefficient
                amp_max = max(abs(amp_bound[0]), abs(amp_bound[1]))
                for zi in idxs:
                    b['zc_%d' % zi] = [-amp_max, amp_max]
            else:
                b['zc_%d' % idxs[0]] = bounds[idx]; idx += 1
        if has_jitter:
            b['jitter_major'], b['jitter_minor'], b['jitter_rot'] = bounds[-3], bounds[-2], bounds[-1]
        return b

    def save_parameter_table(self, entries, filename, bounds=None, fitted_groups=None,
                              n_layers=None, has_jitter=False, already_display=False):
        """
        entries: list of (label, parm_dic, seed, merit) tuples, one per
        column (e.g. one per analysis, or 'Simulated'). If `already_display`
        is True, `parm_dic` is instead an already-flattened display dict
        (see _display_dict/mean_std_display) rather than a raw parameter
        dict -- used for the Mean/Std summary table, since circular phase
        statistics can't be recovered from raw sin/cos values alone.
        Writes one row per parameter, one column per entry (+ a Boundaries
        column if `bounds` is given).
        """
        fitted_groups = fitted_groups or []
        n_layers = n_layers if n_layers is not None else len(entries[0][1]['cn2hei'])
        has_seed = any(e[2] is not None for e in entries)
        has_merit = any(e[3] is not None for e in entries)
        row_defs = self._row_defs(n_layers, fitted_groups, has_jitter, has_seed, has_merit)

        header = ['Parameter']
        if bounds is not None:
            header.append('Boundaries')
        header += [label for label, *_ in entries]

        bound_dict = self._bounds_for_display(bounds, n_layers, fitted_groups, has_jitter) \
            if bounds is not None else {}
        if already_display:
            display_dicts = []
            for _, disp, seed, merit in entries:
                d = dict(disp)
                if seed is not None:
                    d['__seed__'] = seed
                if merit is not None:
                    d['__merit__'] = merit
                display_dicts.append(d)
        else:
            display_dicts = [self._display_dict(parm_dic, fitted_groups, seed, merit)
                              for _, parm_dic, seed, merit in entries]

        section_bounds = set()
        rows = []
        prev_section = None
        for i, (key, row_label) in enumerate(row_defs):
            section = key.split('_')[0] if key[0].islower() else key
            if prev_section is not None and section != prev_section and i > 0:
                section_bounds.add(i - 1)
            prev_section = section
            row = [row_label]
            if bounds is not None:
                b = bound_dict.get(key)
                row.append(sig_format_pair(*b) if b is not None else '')
            for d in display_dicts:
                row.append(d.get(key, ''))
            rows.append(row)

        col_format = 'l' + ('|c' if bounds is not None else '') + 'c' * len(entries)
        write_latex_table(header, rows, filename, col_format=col_format, hline_after=section_bounds)

    @staticmethod
    def mean_std_display(display_dicts):
        """
        Nan-aware mean/std across per-analysis display dicts (see
        _display_dict). NCPA rows are raw (sin, cos) Zernike coefficients
        here, i.e. Cartesian components rather than angles, so a plain
        arithmetic mean/std is the correct statistic (no wrap-around to
        handle, unlike averaging a phase angle directly would have).
        """
        if not display_dicts:
            raise ValueError("Input list is empty")
        mean_d, std_d = {}, {}
        for key in display_dicts[0]:
            values = np.array([d[key] for d in display_dicts], dtype=float)
            mean_d[key] = np.nanmean(values)
            std_d[key] = np.nanstd(values)
        return mean_d, std_d

    # ------------------------------------------------------------------
    #  Correlation tables
    # ------------------------------------------------------------------

    @staticmethod
    def zernike_amplitude_phase(zCoef_list):
        """
        Decompose each fitted sin/cos Zernike pair into amplitude + phase
        [deg]; pairs never actually fitted (amplitude ~ 0 everywhere) are
        skipped. Returns {label: (amplitudes, phases_deg)}.
        """
        pairs = [(0, 1, 'Tip-Tilt'), (3, 4, 'Astigmatism'), (5, 6, 'Coma'), (7, 8, 'Trefoil')]
        zCoef_arr = np.array(zCoef_list)
        results = {}
        for i_sin, i_cos, label in pairs:
            s, c = zCoef_arr[:, i_sin], zCoef_arr[:, i_cos]
            amplitude = np.sqrt(s ** 2 + c ** 2)
            if np.allclose(amplitude, 0.0, atol=1e-6):
                continue
            results[label] = (amplitude, np.degrees(np.arctan2(s, c)))
        return results

    @staticmethod
    def circular_mean_std_deg(angles_deg):
        rad = np.radians(angles_deg)
        C, S = np.mean(np.cos(rad)), np.mean(np.sin(rad))
        mean_deg = np.degrees(np.arctan2(S, C))
        R = min(np.sqrt(C ** 2 + S ** 2), 1.0)
        std_deg = np.degrees(np.sqrt(-2.0 * np.log(R))) if R > 0 else np.nan
        return mean_deg, std_deg

    def save_zernike_amp_phase_table(self, zern_dict, filename):
        header = ['Zernike pair', 'Amplitude [nm]', 'Phase [deg]']
        rows = []
        for label, (amp, phase) in zern_dict.items():
            amp_mean = np.mean(amp)
            amp_std = np.std(amp, ddof=1) if len(amp) > 1 else np.nan
            phase_mean, phase_std = self.circular_mean_std_deg(phase)
            rows.append([label, format_val_pm_err(amp_mean, amp_std), format_val_pm_err(phase_mean, phase_std)])
        write_latex_table(header, rows, filename, col_format='l|c|c')

    @staticmethod
    def save_correlation_table(value_dict, filename, sig=2, bold_threshold=0.9):
        """
        Square Pearson correlation matrix table (all entries of value_dict,
        same rows and columns). Cells with |r| >= bold_threshold are
        highlighted in bold (None disables highlighting).
        """
        labels = list(value_dict.keys())
        data = np.array([value_dict[l] for l in labels])
        corr = np.corrcoef(data)
        latex_labels = [latex_label_for_key(l) for l in labels]

        def fmt_cell(value):
            s = sig_format(value, sig)
            if bold_threshold is not None and abs(value) >= bold_threshold:
                return '\\textbf{%s}' % s
            return s

        header = [''] + latex_labels
        rows = []
        for i in range(len(labels)):
            row = [latex_labels[i]] + [fmt_cell(1.0 if i == j else corr[i, j]) for j in range(len(labels))]
            rows.append(row)
        write_latex_table(header, rows, filename, col_format='l|' + 'c' * len(labels), sig=sig)

    def build_angle_dict(self, mem_param_dics, zern_dict):
        windir_arr = np.array([d['windir'] for d in mem_param_dics])
        angle_dict = {'Wind dir. L%d' % ii: windir_arr[:, ii] for ii in range(windir_arr.shape[1])}
        for label, (_, phase) in zern_dict.items():
            angle_dict['%s phase' % label] = phase
        if all('jitter' in d for d in mem_param_dics):
            jitter_arr = np.array([d['jitter'] for d in mem_param_dics])
            if jitter_arr.shape[1] >= 3:
                angle_dict['Jitter rot.'] = jitter_arr[:, 2]
        return angle_dict

    @staticmethod
    def build_magnitude_dict(mem_param_dics):
        """Seeing, per-layer Cn2 weight and per-layer wind velocity, for the
        velocity / cn2 / seeing correlation table. The last layer's Cn2
        weight is omitted: the weights sum to 1, so it is fully determined
        by the others and adds no independent information."""
        d = {'Seeing': np.array([m['seeing'] for m in mem_param_dics])}
        n_layers = len(mem_param_dics[0]['cn2hei'])
        for ii in range(n_layers):
            if ii < n_layers - 1:
                d['Cn2 weight L%d' % ii] = np.array([m['cn2wei'][ii] for m in mem_param_dics])
            d['Wind vel. L%d' % ii] = np.array([m['winvel'][ii] for m in mem_param_dics])
        return d

    def save_all_correlation_tables(self, mem_param_dics, project_output_dir):
        if len(mem_param_dics) < 4:
            print('Not enough analyses for meaningful correlation tables: skipping.')
            return

        zcoef_list = [d['zCoefStaticOn'] for d in mem_param_dics if 'zCoefStaticOn' in d]
        if zcoef_list:
            zern_dict = self.zernike_amplitude_phase(zcoef_list)
            if zern_dict:
                self.save_zernike_amp_phase_table(
                    zern_dict, os.path.join(project_output_dir, 'Zernike_amplitude_phase_table.txt'))
                angle_dict = self.build_angle_dict(mem_param_dics, zern_dict)
                if len(angle_dict) >= 2:
                    self.save_correlation_table(
                        angle_dict, os.path.join(project_output_dir, 'Correlation_table_wind_zernike_jitter.txt'))

        magnitude_dict = self.build_magnitude_dict(mem_param_dics)
        self.save_correlation_table(
            magnitude_dict, os.path.join(project_output_dir, 'Correlation_table_velocity_cn2_seeing.txt'))

        print('Saved correlation and Zernike amplitude/phase tables.')

    # ------------------------------------------------------------------
    #  Per-analysis / branch summary tables
    # ------------------------------------------------------------------

    @staticmethod
    def save_analysis_summary_table(rows, filename):
        header = "\\multirow{2}{4em}{\\textbf{Analysis}} & \\multicolumn{2}{c|}{\\textbf{FWHM rel. res.} }" \
                 "& \\multirow{2}{4em}{\\textbf{Flux error}}& \\multirow{2}{4em}{\\textbf{Merit}}" \
                 "& \\multirow{2}{4em}{\\textbf{Iteration}}& \\multirow{2}{4em}{\\textbf{Time [hr]}} \\\\ & major & minor \\\\"
        write_latex_table(header, rows, filename, col_format="l|cc|cc|cc")

    @staticmethod
    def save_flux_error_table(flux_error_rows, filename_base):
        """
        Transposed: one row per star actually used (as reference or control
        in at least one analysis -- avoided/unused stars are dropped rather
        than showing as a column of NaNs), one column per analysis. Split
        into <=25-row chunks for landscape-A4 printing.
        """
        analysis_labels = [str(num) for num, _ in flux_error_rows]
        data = np.array([errs for _, errs in flux_error_rows])  # (n_analyses, n_total)
        used_stars = np.nonzero(~np.all(np.isnan(data), axis=0))[0]

        header = ['Star'] + analysis_labels
        rows = [[str(s)] + list(data[:, s]) for s in used_stars]
        write_latex_table_chunked(header, rows, filename_base,
                                   col_format='l' + 'c' * len(analysis_labels))

    @staticmethod
    def save_branch_summary_table(branch_results, filename):
        header = "\\multirow{2}{4em}{\\textbf{Branch}} & \\multicolumn{2}{c|}{\\textbf{FWHM rel. res.} }" \
                 "& \\multirow{2}{4em}{\\textbf{Flux error}}& \\multirow{2}{4em}{\\textbf{Merit}}" \
                 "& \\multirow{2}{4em}{\\textbf{Iteration}}& \\multirow{2}{4em}{\\textbf{Time [hr]}} \\\\ & major & minor \\\\"
        rows = []
        for branch_name, res in branch_results.items():
            rows.append([
                branch_name,
                np.nanmean(res.get('rms_major_list', [np.nan])) * 100,
                np.nanmean(res.get('rms_minor_list', [np.nan])) * 100,
                np.nanmean(res.get('mean_flux_err_list', [np.nan])) * 100,
                np.nanmean(res.get('merit_list', [np.nan])),
                np.nanmean(res.get('n_iter_list', [np.nan])),
                np.nanmean(res.get('run_time_list', [np.nan])),
            ])
        write_latex_table(header, rows, filename, col_format="l|cc|cc|cc")

    # ------------------------------------------------------------------
    #  Subfolder discovery
    # ------------------------------------------------------------------

    def _discover_subdirs(self, project_dir, subdir_tag, subdir_numbers):
        subfolders_all = [f.name for f in os.scandir(project_dir) if f.is_dir()]
        subfolders_all = [s for s in subfolders_all if not s.startswith(self.OUTPLOT_PREFIX)]

        if len(subfolders_all) == 1:
            subdir_tag = subfolders_all[0][:-1]
            return subdir_tag, [int(subfolders_all[0][-1])]

        if subdir_tag is None:
            subdir_tag = os.path.commonprefix(subfolders_all)
            subfolders = subfolders_all
        else:
            subfolders = [s for s in subfolders_all if s.startswith(subdir_tag)]

        if subdir_numbers is None:
            prefix_len = len(subdir_tag)
            subdir_numbers = []
            for subname in subfolders:
                digits = ''.join(c for c in subname[prefix_len:] if c.isdigit())
                subdir_numbers.append(int(digits))
        return subdir_tag, sorted(subdir_numbers)

    # ------------------------------------------------------------------
    #  Per-run-folder processing (shared by run_analysis, run_single)
    # ------------------------------------------------------------------

    def _process_run_folder(self, estimator_dir, estimator_output_dir, label, saveflag_estimation=True):
        """
        Process ONE estimation-run folder (a folder that contains
        save_status.ini directly): load it, resolve the field, compute
        distances, produce the merit-trend / FWHM / flux / region / field
        plots and the per-star results table, and print the per-run console
        summary. Returns a SimpleNamespace with everything a caller needs to
        aggregate across several runs (run, result, dist_ref, dist_ctrl,
        rms_major/minor, mean_flux_err, err_flux_combined, run_time_val).
        """
        os.makedirs(estimator_output_dir, exist_ok=True)
        run = RunData(estimator_dir, self.save_status_ftag, self.avoid_indices)
        self._resolve_field(run)
        dist_ref, dist_ctrl = self.compute_dist(run)

        if self.do_merit:
            self.graphics.plot_merit_trend(run.merits_gens, label,
                                            os.path.join(estimator_output_dir, 'Merit_trend.png'))

        result = self.compute_run_merits(run, estimator_output_dir, saveflag_estimation=saveflag_estimation)
        result, dist_ctrl = self._apply_flux_threshold(result, dist_ctrl)

        rms_major, rms_minor = self.rms_relative_fwhm(
            result.data_std_vec, result.reconstruct_std_vec,
            result.data_remain_std_vec, result.reconstruct_remain_std_vec)
        rms_major_ref, rms_minor_ref = self.rms_relative_fwhm_single(
            result.data_std_vec, result.reconstruct_std_vec)
        flux_err_ref = np.nanmean(np.abs(result.err_flux_ref)) * 100
        if result.has_control_stars:
            rms_major_ctrl, rms_minor_ctrl = self.rms_relative_fwhm_single(
                result.data_remain_std_vec, result.reconstruct_remain_std_vec)
            flux_err_ctrl = np.nanmean(np.abs(result.err_flux_ctrl)) * 100
        else:
            rms_major_ctrl = rms_minor_ctrl = flux_err_ctrl = np.nan

        print('  Merit function:                     %s' % sig_format(run.merit, 5))
        print('  Flux error   (reference):            %6.3f %%' % flux_err_ref)
        if result.has_control_stars:
            print('  Flux error   (control):              %6.3f %%' % flux_err_ctrl)
        print('  FWHM RMS rel. residual (reference):  major=%6.3f %%  minor=%6.3f %%' %
              (rms_major_ref * 100, rms_minor_ref * 100))
        if result.has_control_stars:
            print('  FWHM RMS rel. residual (control):    major=%6.3f %%  minor=%6.3f %%' %
                  (rms_major_ctrl * 100, rms_minor_ctrl * 100))
        print('  FWHM RMS rel. residual (all):        major=%6.3f %%  minor=%6.3f %%' %
              (rms_major * 100, rms_minor * 100))

        n_stars = self.all_pos_field.shape[0]
        err_flux_combined = self.combined_err_flux(result, n_stars)
        mean_flux_err = np.nanmean(np.abs(err_flux_combined))
        try:
            run_time_val = float(run.run_time)
        except (TypeError, ValueError):
            run_time_val = np.nan

        self.save_run_summary_table(run, estimator_output_dir, rms_major, rms_minor,
                                     rms_major_ref, rms_minor_ref, flux_err_ref,
                                     rms_major_ctrl, rms_minor_ctrl, flux_err_ctrl,
                                     result.has_control_stars, run_time_val)

        if self.do_regions:
            self.graphics.plot_star_regions(estimator_output_dir, result)
        if self.do_fwhm:
            self.graphics.plot_fwhm(dist_ref, dist_ctrl, result.data_std_vec, result.reconstruct_std_vec,
                                     result.data_remain_std_vec, result.reconstruct_remain_std_vec, estimator_output_dir)
        if self.do_flux:
            self.graphics.plot_flux(result, dist_ref, dist_ctrl, estimator_output_dir)
            self.graphics.plot_fwhm_error_vs_flux(result, estimator_output_dir)
        if self.do_field:
            self.graphics.plot_field_positions(self.all_pos_field, result.select_stars, result.remaining_indices,
                                                self.AO_pos, estimator_output_dir,
                                                pixscale=run.pixscale, tel_res=run.tel_res)
        self.save_analysis_results_table(result, dist_ref, dist_ctrl, estimator_output_dir)
        plt.close('all')

        return SimpleNamespace(run=run, result=result, dist_ref=dist_ref, dist_ctrl=dist_ctrl,
                                rms_major=rms_major, rms_minor=rms_minor, n_stars=n_stars,
                                mean_flux_err=mean_flux_err, err_flux_combined=err_flux_combined,
                                run_time_val=run_time_val)

    def run_analysis(self, analysis_dir=None, saveflag_estimation=True):
        """
        Process a single estimation-run folder directly -- one that
        contains save_status.ini itself, rather than a parent of several
        numbered run folders. Just that run's own plots/tables; no
        cross-analysis comparison or correlation tables, since there's
        nothing to compare against.
        """
        analysis_dir = analysis_dir or self.project_dir
        output_dir = os.path.join(analysis_dir, self.OUTPLOT_DIR)
        print()
        print('=' * 70)
        print('Single analysis -- %s' % analysis_dir)
        print('=' * 70)
        self._process_run_folder(analysis_dir, output_dir, 'Analysis', saveflag_estimation)
        print()
        print('-' * 70)
        print('Done: %s' % analysis_dir)
        print('-' * 70)
        return output_dir

    # ------------------------------------------------------------------
    #  Single-project loop
    # ------------------------------------------------------------------

    def run_single(self, project_dir=None, subdir_tag=None, subdir_numbers=None,
                    saveflag_estimation=True, branch_accumulator=None):
        project_dir = project_dir or self.project_dir
        # Field geometry (star positions) is read fresh per analysis folder,
        # from that folder's own save_status.ini (see _resolve_field), so
        # nothing is cached at the project/branch level here.
        subdir_tag, subdir_numbers = self._discover_subdirs(project_dir, subdir_tag, subdir_numbers)
        project_output_dir = os.path.join(project_dir, self.OUTPLOT_DIR)
        os.makedirs(project_output_dir, exist_ok=True)

        mem_param_dics, entries = [], []
        data_std_vecs, rem_std_vecs = [], []
        analysis_summary_rows, flux_error_rows = [], []
        dist_ref = dist_ctrl = None
        bounds = None
        fitted_groups, has_jitter, n_layers = [], False, None

        n_total = len(subdir_numbers)
        for i, subdir_number in enumerate(subdir_numbers):
            label = 'Analysis %d' % subdir_number
            print()
            print('=' * 70)
            print('[%d/%d] %s -- %s' % (i + 1, n_total, label, project_dir))
            print('=' * 70)
            estimator_dir = os.path.join(project_dir, subdir_tag + str(subdir_number))
            estimator_output_dir = os.path.join(project_output_dir, subdir_tag + str(subdir_number))

            r = self._process_run_folder(estimator_dir, estimator_output_dir, label, saveflag_estimation)
            run, result, dist_ref, dist_ctrl = r.run, r.result, r.dist_ref, r.dist_ctrl

            mem_param_dics.append(run.est_parm_dic)
            entries.append((label, run.est_parm_dic, run.rng_seed, run.merit))
            bounds, fitted_groups, has_jitter = run.bounds, run.fitted_ncpa_groups(), run.do_jitter
            n_layers = len(run.est_parm_dic['cn2hei'])

            analysis_summary_rows.append([str(subdir_number), r.rms_major * 100, r.rms_minor * 100,
                                           r.mean_flux_err * 100, run.merit, run.n_iterations, r.run_time_val])
            flux_error_rows.append((subdir_number, r.err_flux_combined))

            if branch_accumulator is not None:
                for key, val in [('rms_major_list', r.rms_major), ('rms_minor_list', r.rms_minor),
                                  ('mean_flux_err_list', r.mean_flux_err), ('merit_list', run.merit),
                                  ('n_iter_list', run.n_iterations), ('run_time_list', r.run_time_val)]:
                    branch_accumulator.setdefault(key, []).append(val)
                mf = self._combine_star_arrays(result.select_stars, result.remaining_indices,
                                                result.posteriors_regions, result.rem_posteriors_regions, r.n_stars)
                fwhm_maj = self._combine_star_arrays(result.select_stars, result.remaining_indices,
                                                      result.reconstruct_std_vec[:, 0], result.reconstruct_remain_std_vec[:, 0], r.n_stars)
                fwhm_min = self._combine_star_arrays(result.select_stars, result.remaining_indices,
                                                      result.reconstruct_std_vec[:, 1], result.reconstruct_remain_std_vec[:, 1], r.n_stars)
                data_fwhm_maj = self._combine_star_arrays(result.select_stars, result.remaining_indices,
                                                           result.data_std_vec[:, 0], result.data_remain_std_vec[:, 0], r.n_stars)
                data_fwhm_min = self._combine_star_arrays(result.select_stars, result.remaining_indices,
                                                           result.data_std_vec[:, 1], result.data_remain_std_vec[:, 1], r.n_stars)
                for key, val in [('mf', mf), ('fwhm_major', fwhm_maj), ('fwhm_minor', fwhm_min),
                                  ('data_fwhm_major', data_fwhm_maj), ('data_fwhm_minor', data_fwhm_min)]:
                    branch_accumulator.setdefault(key, []).append(val)

            data_std_vecs.append(result.reconstruct_std_vec)
            rem_std_vecs.append(result.reconstruct_remain_std_vec)
            plt.close('all')

        # --- Per-analysis comparison table (NCPA shown as amplitude/phase) ---
        comparison_entries = list(entries)
        if self.sim_dic is not None:
            comparison_entries = [('Simulated', self.sim_dic, None, None)] + comparison_entries
        self.save_parameter_table(comparison_entries, os.path.join(project_output_dir, 'Table_comparison.txt'),
                                   fitted_groups=fitted_groups, n_layers=n_layers, has_jitter=has_jitter)

        # --- Mean/std summary table, with the estimator's actual boundaries ---
        display_dicts = [self._display_dict(d, fitted_groups) for d in mem_param_dics]
        mean_disp, std_disp = self.mean_std_display(display_dicts)
        mean_std_entries = [('Mean', mean_disp, None, None), ('Std', std_disp, None, None)]
        self.save_parameter_table(mean_std_entries, os.path.join(project_output_dir, 'Table_mean_std.txt'),
                                   bounds=bounds, fitted_groups=fitted_groups, n_layers=n_layers,
                                   has_jitter=has_jitter, already_display=True)

        if data_std_vecs:
            rec_mean_fwhm = np.nanmean(np.stack(data_std_vecs), axis=0)
            rec_std_fwhm = np.nanstd(np.stack(data_std_vecs), axis=0)
            rec_remain_mean_fwhm = np.nanmean(np.stack(rem_std_vecs), axis=0)
            rec_remain_std_fwhm = np.nanstd(np.stack(rem_std_vecs), axis=0)
            if self.do_fwhm:
                self.graphics.plot_fwhm(dist_ref, dist_ctrl, result.data_std_vec, rec_mean_fwhm,
                                         result.data_remain_std_vec, rec_remain_mean_fwhm, project_output_dir,
                                         reconstruct_std_err=rec_std_fwhm, reconstruct_remain_std_err=rec_remain_std_fwhm)
        plt.close('all')

        if analysis_summary_rows:
            self.save_analysis_summary_table(analysis_summary_rows, os.path.join(project_output_dir, 'Analysis_summary_table.txt'))
        if flux_error_rows:
            self.save_flux_error_table(flux_error_rows, os.path.join(project_output_dir, 'Flux_error_table'))
        if mem_param_dics:
            self.save_all_correlation_tables(mem_param_dics, project_output_dir)

        if branch_accumulator is not None:
            branch_accumulator['dist'] = self.star_distances(self.all_pos_field)
            branch_accumulator['mf_mean'] = np.nanmean(np.stack(branch_accumulator['mf']), axis=0)
            branch_accumulator['fwhm_major_mean'] = np.nanmean(np.stack(branch_accumulator['fwhm_major']), axis=0)
            branch_accumulator['fwhm_major_std'] = np.nanstd(np.stack(branch_accumulator['fwhm_major']), axis=0)
            branch_accumulator['fwhm_minor_mean'] = np.nanmean(np.stack(branch_accumulator['fwhm_minor']), axis=0)
            branch_accumulator['fwhm_minor_std'] = np.nanstd(np.stack(branch_accumulator['fwhm_minor']), axis=0)
            # Data FWHM is the same physical measurement regardless of estimation
            # run, so the mean across runs just averages out any per-run NaN gaps.
            branch_accumulator['data_fwhm_major'] = np.nanmean(np.stack(branch_accumulator['data_fwhm_major']), axis=0)
            branch_accumulator['data_fwhm_minor'] = np.nanmean(np.stack(branch_accumulator['data_fwhm_minor']), axis=0)

        overall_major = np.nanmean([r[1] for r in analysis_summary_rows]) if analysis_summary_rows else np.nan
        overall_minor = np.nanmean([r[2] for r in analysis_summary_rows]) if analysis_summary_rows else np.nan
        print()
        print('-' * 70)
        print('Done: %s' % project_dir)
        print('Overall mean RMS FWHM relative residual: major=%.3f %%, minor=%.3f %%' % (overall_major, overall_minor))
        print('-' * 70)
        return project_output_dir

    def save_run_summary_table(self, run, estimator_output_dir, rms_major, rms_minor,
                                rms_major_ref, rms_minor_ref, flux_err_ref,
                                rms_major_ctrl, rms_minor_ctrl, flux_err_ctrl,
                                has_control_stars, run_time_val):
        """
        Per-analysis summary table -- the same quantities as the console
        summary block (merit function, flux error, FWHM RMS relative
        residual for reference/control/all, at the same fixed precision),
        plus the iteration count and computation time -- saved to
        Run_summary_table.txt in this run's own output folder.
        """
        n_iter = run.n_iterations
        rows = [['Merit function', sig_format(run.merit, 5)],
                ['Flux error (reference) [\\%]', '%.3f' % flux_err_ref]]
        if has_control_stars:
            rows.append(['Flux error (control) [\\%]', '%.3f' % flux_err_ctrl])
        rows += [
            ['FWHM RMS rel. residual (reference) major [\\%]', '%.3f' % (rms_major_ref * 100)],
            ['FWHM RMS rel. residual (reference) minor [\\%]', '%.3f' % (rms_minor_ref * 100)],
        ]
        if has_control_stars:
            rows += [
                ['FWHM RMS rel. residual (control) major [\\%]', '%.3f' % (rms_major_ctrl * 100)],
                ['FWHM RMS rel. residual (control) minor [\\%]', '%.3f' % (rms_minor_ctrl * 100)],
            ]
        rows += [
            ['FWHM RMS rel. residual (all) major [\\%]', '%.3f' % (rms_major * 100)],
            ['FWHM RMS rel. residual (all) minor [\\%]', '%.3f' % (rms_minor * 100)],
            ['Number of cycles', '%d' % int(n_iter) if not np.isnan(n_iter) else 'N/A'],
            ['Computation time [hr]', '%.3f' % run_time_val if not np.isnan(run_time_val) else 'N/A'],
        ]
        # PROLITE configuration: a section of its own, set off by an hline.
        config_start = len(rows)
        rows += [
            ['Estimation Type', run.estimator_type if run.estimator_type is not None else 'N/A'],
            ['PSF Fitting', run.psf_fit_type_cfg if run.psf_fit_type_cfg is not None else 'N/A'],
            ['MF shape', run.merit_shape if run.merit_shape is not None else 'N/A'],
        ]
        write_latex_table(['Quantity', 'Value'], rows,
                           os.path.join(estimator_output_dir, 'Run_summary_table.txt'),
                           col_format='l|c', hline_after={config_start - 1})

    def save_analysis_results_table(self, result, dist_ref, dist_ctrl, fpath):
        header = ('&&&& \\multicolumn{3}{c|}{\\textbf{FWHM Major}}& \\multicolumn{3}{c|}{\\textbf{FWHM Minor}}'
                   '& \\multicolumn{3}{c|}{\\textbf{FWHM Angle}} \\\\ \n'
                   '\\textbf{Star id} & \\textbf{Dist. AO} & \\textbf{Flux $F$} & \\textbf{$\\Delta F/F$} & '
                   '\\textbf{data} & \\textbf{rec} & \\textbf{error} & \\textbf{data} & \\textbf{rec} & \\textbf{error} & '
                   '\\textbf{data} & \\textbf{rec} & \\textbf{error}\\\\ \n  & [arcsec] & [ADU] & [\\%] & [mas] & [mas] & '
                   '[\\%] & [mas] & [mas] & [\\%] & [rad] & [rad] & [rad] \\\\')

        def rows_for(idx, dist, data_std_vec, reconstruct_std_vec, tot_flux, err_flux):
            rows = []
            for row in range(len(idx)):
                r = [idx[row], dist[row], tot_flux[row], err_flux[row] * 100]
                for axis in (0, 1):
                    d, rec = data_std_vec[row, axis], reconstruct_std_vec[row, axis]
                    r += [d, rec, (d - rec) / d * 100]
                d, rec = data_std_vec[row, 2], reconstruct_std_vec[row, 2]
                r += [d, rec, d - rec]
                rows.append(r)
            return rows

        write_latex_table_chunked(header, rows_for(result.select_stars, dist_ref, result.data_std_vec,
                                                     result.reconstruct_std_vec, result.tot_flux_ref, result.err_flux_ref),
                                   os.path.join(fpath, 'results_reference_stars'), col_format='l|c|cc|ccc|ccc|ccc|')
        if result.has_control_stars:
            write_latex_table_chunked(header, rows_for(result.remaining_indices, dist_ctrl, result.data_remain_std_vec,
                                                          result.reconstruct_remain_std_vec, result.tot_flux_ctrl, result.err_flux_ctrl),
                                       os.path.join(fpath, 'results_control_stars'), col_format='l|c|cc|ccc|ccc|ccc|')

    # ------------------------------------------------------------------
    #  Branch loop (compare several projects/branches against each other)
    # ------------------------------------------------------------------

    def run_branches(self, grandparent_dir=None, branch_tag=None, saveflag_estimation=False):
        grandparent_dir = grandparent_dir or self.project_dir
        grand_output_dir = os.path.join(grandparent_dir, self.OUTPLOT_DIR)
        os.makedirs(grand_output_dir, exist_ok=True)

        branch_names = sorted(f.name for f in os.scandir(grandparent_dir) if f.is_dir())
        branch_names = [b for b in branch_names if not b.startswith(self.OUTPLOT_PREFIX)]
        if branch_tag is not None:
            branch_names = [b for b in branch_names if b.startswith(branch_tag)]
        if not branch_names:
            print('No branch (parent) folders found in %s' % grandparent_dir)
            return {}

        branch_results = {}
        for branch_name in branch_names:
            print('=== Branch: %s ===' % branch_name)
            branch_accumulator = {}
            self.run_single(os.path.join(grandparent_dir, branch_name),
                             saveflag_estimation=saveflag_estimation, branch_accumulator=branch_accumulator)
            branch_results[branch_name] = branch_accumulator

        self.graphics.plot_branch_comparison(branch_results, grand_output_dir, self.colors)
        self.save_branch_summary_table(branch_results, os.path.join(grand_output_dir, 'Branch_comparison_table.txt'))
        plt.close('all')
        print('Done: all branches.')
        return branch_results


# ===========================================================================
#  CLI
# ===========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Post-process AO Cn2/NCPA estimation runs: FWHM/flux residuals, '
                     'parameter tables and correlations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('project_dir', help='Project directory (single mode) or grandparent '
                                              'directory of branches (branch mode).')
    parser.add_argument('--all-pos-fpath', default=None,
                         help="Path to a 2-column (x, y) file listing every star in the "
                              "field. Overrides 'fpath-all-pos' in save_status.ini, if "
                              "that's also set; one of the two is required.")
    parser.add_argument('--ao-pos-fpath', default=None,
                         help="Path to the AO star's (x, y) position file. Overrides "
                              "'fpath-ao-pos' in save_status.ini, if also set.")
    parser.add_argument('--avoid-fpath', default=None,
                         help='Optional file of star indices to exclude from both reference '
                              'and control stars (whitespace/newline-separated integers), '
                              "applied to every run. Without this, each run falls back to "
                              "'avoid-stars-fpath' in its own save_status.ini, if present.")
    parser.add_argument('--flux-threshold', type=float, default=None,
                         help='Only consider control stars with fitted flux >= this value, '
                              'for every RMS/table/plot (reference stars are never filtered). '
                              "Results then go to 'Outplot_flux_<value>' instead of 'Outplot', "
                              'so filtered and unfiltered runs never collide.')
    parser.add_argument('--mode', choices=['analysis', 'single', 'branches'], default=None,
                         help="'analysis': one estimation-run folder (has save_status.ini "
                              "directly). 'single': one project/branch of numbered run "
                              "subfolders. 'branches': several such projects to compare. "
                              "If omitted, auto-detected from the directory structure by "
                              "looking for save_status.ini up to two levels below project_dir.")
    parser.add_argument('--psf-fit-type', default='lmsq', help='PSF fit type passed to Merit.')
    parser.add_argument('--plots', nargs='+', choices=PLOT_CHOICES, default=['all'],
                         help='Which plots to produce.')
    parser.add_argument('--save-status-ftag', default='save_status',
                         help='Basename (no extension) of the save-status ini file.')
    parser.add_argument('--subdir-tag', default=None,
                         help='Common prefix of the numbered run subfolders (single mode only).')
    parser.add_argument('--branch-tag', default=None,
                         help='Common prefix of the branch folders (branch mode only).')
    parser.add_argument('--no-estimation-save', action='store_true',
                         help='Skip saving the individual Merit estimator plots/files.')
    parser.add_argument('--show', action='store_true', help='Also display plots interactively.')
    return parser.parse_args(argv)


def plot_results(project_dir, mode=None, psf_fit_type='lmsq', save_status_ftag='save_status',
                  plots=('all',), showflag=False, colors=None,
                  all_pos_fpath=None, ao_pos_fpath=None, avoid_fpath=None, sim_dic=None,
                  flux_threshold=None, subdir_tag=None, branch_tag=None, saveflag_estimation=True):
    """
    Library entry point: everything the command line does, callable
    directly from another Python script, e.g.:

        from analyze_ao import plot_results
        plot_results('/path/to/Branch_or_Analysis_or_Parents_dir')

    project_dir can point at any of the three levels (a single analysis
    folder with save_status.ini directly, a project/branch folder of
    numbered Analysis_N/ subfolders, or a grandparent folder of several
    such branches) -- mode is auto-detected the same way as the CLI unless
    `mode` is given explicitly ('analysis', 'single' or 'branches'). Every
    other argument mirrors an AOAnalysis constructor argument or a
    run_analysis/run_single/run_branches argument; see their docstrings.

    Returns whatever the underlying run_analysis/run_single/run_branches
    call returns: an output-folder path for 'analysis'/'single' mode, or a
    {branch name: accumulator} dict for 'branches' mode.
    """
    if mode is None:
        mode = detect_mode(project_dir, save_status_ftag)
        print("Auto-detected mode: '%s'" % mode)

    analysis = AOAnalysis(
        project_dir=project_dir,
        psf_fit_type=psf_fit_type,
        save_status_ftag=save_status_ftag,
        plots=plots,
        showflag=showflag,
        colors=colors,
        all_pos_fpath=all_pos_fpath,
        ao_pos_fpath=ao_pos_fpath,
        avoid_fpath=avoid_fpath,
        sim_dic=sim_dic,
        flux_threshold=flux_threshold,
    )
    if mode == 'branches':
        return analysis.run_branches(branch_tag=branch_tag, saveflag_estimation=saveflag_estimation)
    elif mode == 'analysis':
        return analysis.run_analysis(saveflag_estimation=saveflag_estimation)
    else:
        return analysis.run_single(subdir_tag=subdir_tag, saveflag_estimation=saveflag_estimation)


def main(argv=None):
    args = parse_args(argv)
    plot_results(
        args.project_dir,
        mode=args.mode,
        psf_fit_type=args.psf_fit_type,
        save_status_ftag=args.save_status_ftag,
        plots=args.plots,
        showflag=args.show,
        all_pos_fpath=args.all_pos_fpath,
        ao_pos_fpath=args.ao_pos_fpath,
        avoid_fpath=args.avoid_fpath,
        flux_threshold=args.flux_threshold,
        subdir_tag=args.subdir_tag,
        branch_tag=args.branch_tag,
        saveflag_estimation=not args.no_estimation_save,
    )


if __name__ == '__main__':
    main()