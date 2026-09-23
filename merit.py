import numpy as np
import os
import queue
import subprocess

import pixelUtiles as pxu
import format_and_control as fcu
import files as ff
from graphics import Graphics

import tiptop.tiptop as tiptop
from scipy.ndimage import shift
from astropy.modeling import models, fitting
from multiprocessing import Pool

from scipy.optimize import curve_fit

import matplotlib.pylab as plt
# Use TeX in the plots
#plt.rcParams['figure.dpi'] = 100
#plt.rcParams['savefig.dpi'] = 600
plt.rcParams['text.usetex'] = True


def gaussian_weighted_average(data, sigma, epsilon=1e-3):
    # get shape
    h, w = data.shape
    # Get the central points
    cy, cx = h // 2, w // 2
    # Create coordinate grid 
    y, x = np.indices((h, w)) 
    # Cretae the Gaussian kernel
    kernel = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    # Add a costant
    kernel += epsilon
    # Normalize kernel
    kernel /= kernel.sum()
    # Weighted average
    wei_avg = np.nansum(data * kernel)
    return wei_avg

def merit_func(data, sim, radius, shape='gauss', std=None, do_err_flux=False):
    residuals_no_mask = data - sim
    # Applying a circular mask
    residuals = residuals_no_mask.copy()
    mask = pxu.inverse_circular_mask(residuals, [residuals.shape[0]/2,residuals.shape[0]/2], radius)
    residuals[mask] = np.nan
    # Mask data
    data_mask = data.copy()
    data_mask[mask] = np.nan
    # Compute the flux error 
    err_flux = np.nansum(residuals)/np.nansum(data_mask)
    tot_flux = np.nansum(data_mask)
    # Compute the average
    if shape=='linear':
        # Compute the residuals sum
        sum_residuals = np.nansum((residuals)**2)
        # Get data sum
        sum_data = np.nansum((data_mask)**2)
    elif shape=='gauss':
        # Check if there is no input std
        if std is None:
            std = radius/3
        # Average the residuals
        sum_residuals = gaussian_weighted_average(residuals**2, std)
        # Average the data
        sum_data = gaussian_weighted_average(data_mask**2, std)
    else:
        raise ValueError('Invalid merit shape %s', shape)
    # compute merit function
    mf_i = 100*np.sqrt(sum_residuals / sum_data)
    # Create live-plot map
    live_map = residuals_no_mask
    if do_err_flux:
        return mf_i, live_map, err_flux, tot_flux
    else:
        return mf_i, live_map

def combine_mf(merits_vec):
    mf = np.sum(merits_vec)/merits_vec.shape[0]
    return mf

def _calc_pos_and_flux(regions_original, radii, fwhmResults=False):
    # Copy data to preserve it
    regions = regions_original.copy()
    # Check if there is just one region or multipole
    if regions.ndim == 2:
        has_one_region = True
        regions = np.array([regions])
        radii = np.array([radii])
    elif regions.ndim == 3:
        has_one_region = False 
    else:
        raise ValueError('InvalidShape: must be a 1D or 2D np.array, given with shape ', regions.shape)
    # Create array to store the informations
    stars_pos = np.empty((regions.shape[0], 2))
    stars_flux = np.empty(regions.shape[0])
    std_vec = np.empty((regions.shape[0], 3))
    # Get the position and flux for each star
    for ii, region in enumerate(regions):
        # Must has a odd shape for box axes
        if region.shape[0]%2 == 0:
            region = region[:-1, :]
        if region.shape[1]%2 == 0:
            region = region[:, :-1]
        # Create a mask to cut out all the outside information
        mask = pxu.inverse_circular_mask(region, [region.shape[0]/2, region.shape[1]/2], radii[ii])
        # Gaussian fit that works with the nans
        # Define the grid
        y, x = np.mgrid[:region.shape[0], :region.shape[1]]
        # Remove the non-finite elements
        finite = np.isfinite(region)
        # Apply mask
        x_finite = x[finite]
        y_finite = y[finite]
        region_finite = region[finite]
        # Define the models to be fitted
        gauss_init = models.Gaussian2D(amplitude=np.nanmax(region), x_mean=region.shape[1]/2,
            y_mean=region.shape[0]/2,
            x_stddev=5,
            y_stddev=5
        )
        # Set the minimum to be positive
        gauss_init.amplitude.min = 0
        # Define the fitter
        fit = fitting.LevMarLSQFitter()
        # Make the fitting
        g_fit = fit(gauss_init, x_finite, y_finite, region_finite)
        # Get the stars relative position and save
        x0 = g_fit.x_mean.value
        y0 = g_fit.y_mean.value
        stars_pos[ii] = [x0 , y0]
        # Compute the stars flux and save
        A = g_fit.amplitude.value
        std_x = g_fit.x_stddev.value
        std_y = g_fit.y_stddev.value
        theta = g_fit.theta.value
        stars_flux[ii] = A * 2 * np.pi * std_x * std_y
        # Fix a correct order for the fwhm
        if std_x >= std_y:
            fwhm_major = 2.355 * std_x
            fwhm_minor = 2.355 * std_y
        else:
            fwhm_major = 2.355 * std_y
            fwhm_minor = 2.355 * std_x
            theta += 1.5708
        std_vec[ii] = [fwhm_major, fwhm_minor, theta] 
    # Return star position and flux
    if fwhmResults:
        if has_one_region:
            return stars_pos[0], stars_flux[0], std_vec[0]
        else:
            return stars_pos, stars_flux, std_vec
    else:
        if has_one_region:
            return stars_pos[0], stars_flux[0]
        else:
            return stars_pos, stars_flux

def _pos_flux_fit_lmsq(region_original, data_region_mask, region_radius):
    # Copy the data 
    region = region_original.copy()
    # Set the limits:
    max_data = np.nanmax(data_region_mask)
    max_sim = np.nanmax(region)
    max_amplitude = max_data / max_sim
    # Define the bounderies as ([mins], [maxs]) for the curve fit
    lmsq_bounds = ([-2, -2, max_amplitude*5e-2, -200], [2, 2, max_amplitude*1e2, 100])
    # Perform the curve fit
    try:
        popt, _ = curve_fit(lambda img_array, x_shift, y_shift, height, bias: _shift_scale_biased_ravel(img_array, x_shift, y_shift, height, bias, region_radius),
                                region, data_region_mask, bounds=lmsq_bounds)
        #curve_fit(_shift_scale_biased_ravel, region, data_region_mask, bounds=lmsq_bounds)
        # Apply the best-fit transform to the simulation
        sim_region_fitted = _shift_scale_biased(region_original, *popt)
    except:
        print('WARNING: LSMQ fitting failed!!!')
        sim_region_fitted = region
    return sim_region_fitted

def _shift_scale_biased(img, x_shift, y_shift, height, bias):
    # Shift the image
    final_img = shift(img.copy(), (x_shift, y_shift))
    # Change the scale of the PSF
    final_img *= height
    # Add the bias
    final_img += bias
    # Return the image
    return final_img

def _shift_scale_biased_ravel(img_array, x_shift, y_shift, height, bias, region_radius):
    # Copy the array to not corrupt the original
    img = img_array.copy()
    # Call the function
    final_img = _shift_scale_biased(img, x_shift, y_shift, height, bias)
    # Apply the circular mask
    if region_radius is not None:
        if region_radius > 0:
            # Create the mask around the center
            mask = pxu.circular_mask(final_img, [final_img.shape[0]//2, final_img.shape[1]//2], region_radius)
            # Remov the elements outside the radius
            final_img = final_img[mask]
        else:
            print('Radius must be positive')
    # Squezee from two dimention to one
    return np.ravel(final_img)


## Multipool process function to compute the merit in each region
def _pool_merit_region(args):
    (ii, sim_region, psf_fit_type, radii, data_stars_pos, data_stars_flux, data_std, data_regions, data_regions_mask, merit_shape, gaussResult) = args

    std_val = None 
    if psf_fit_type == 'gauss':
        # Find the 
        sim_star_pos, sim_star_flux, std_val = _calc_pos_and_flux(sim_region, radii[ii], True)

        sim_region = shift(sim_region, (data_stars_pos[ii, 1] - sim_star_pos[0], data_stars_pos[ii, 0] - sim_star_pos[1]) )

        sim_region *= data_stars_flux[ii] / sim_star_flux

    elif psf_fit_type == 'lmsq':
        if gaussResult:
            _, _, std_val = _calc_pos_and_flux(sim_region, radii[ii], True)

        sim_region = _pos_flux_fit_lmsq(sim_region, data_regions_mask[ii], radii[ii])

    value, map_region = merit_func(data_regions[ii], sim_region, radii[ii], merit_shape, data_std[ii])


    return ii, value, map_region, std_val


class Merit:
    def __init__(self, data_regions, radii, outdir, temp_dir, temp_ini_filetag, temp_sim_filetag, data_ini_filepath, psf_fit_type='gauss', merit_shape='linear', stars_idx=None, do_err_flux=False, verbose=True):
        # Store the input data
        self.data_regions = data_regions
        self.outdir = outdir
        self.temp_dir = temp_dir
        self.temp_ini_filetag = temp_ini_filetag
        self.temp_sim_filetag = temp_sim_filetag
        self.data_ini_filepath = data_ini_filepath
        self.psf_fit_type = psf_fit_type
        # Compute stars number
        self.n_star  = self.data_regions.shape[0]
        if stars_idx is None:
            self.stars_idx = np.arange(self.n_star)
        else:
            stars_idx = np.array(stars_idx)
            if stars_idx.shape[0] == self.n_star:
                self.stars_idx = stars_idx
            else:
                raise ValueError('stars_idx must have the same lenght of data_regions.shape[0]')
        self.data_regions_mask = []
        # Format the radii for estimation
        self.update_radii(radii)
        # Create the queue that store the GPU order
        self.gpu_queue = None
        # parallel PSF matching
        self.use_parallel_PSF_matching = True
        if self.psf_fit_type == 'lmsq':
            if verbose:
                print('######\tDo curve fit')
        elif self.psf_fit_type == 'gauss':
            if verbose:
                print('######\tDo Gauss fit')
        elif self.psf_fit_type == 'fix':
            if verbose:
                print('######\t No fit - fix PSF position and scale')
            self.use_parallel_PSF_matching = False
        # Raise an error
        else:
            raise ValueError('psf_fit_type is not well defined: used %s, expected gauss, lmsq or fix' %self.psf_fit_type)
        
        # Start the pooling
        self.MAX_process = 5
        self.pool = None
        self.merit_shape = merit_shape
        self.do_err_flux = do_err_flux
        if self.use_parallel_PSF_matching:
            self.setup_pool()

    def setup_pool(self, pool_number = None):
        # define the correct pool size
        if pool_number is None:
            pool_size = min(self.n_star, self.MAX_process)
        else:
            pool_size = min(int(pool_number), self.MAX_process)
        # Start the pools
        self.pool = Pool(processes=pool_size)

    # Close all the pools to avoid memory leaking
    def close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        self.close_pool()
        
## TODO: calcolare in modo dinamico la dimensione del processo, controllare se il numero di threads e la lista delle GPU hanno la stessa dimensione.    
    def set_multi_threads(self, n_threads=1, process_size_GB = 1, gpu_order_list=None):
        if gpu_order_list is None:
            self.gpu_queue = self._find_best_gpu_order(n_threads, process_size_GB)
        else:
            gpu_order_list = np.array(gpu_order_list)
            self.gpu_queue = queue.Queue(maxsize=gpu_order_list.shape[0])
            for gpu_idx in gpu_order_list:
                self.gpu_queue.put(int(gpu_idx))
    
    def update_radii(self, radii):
        # Format the radii for estimation
        self.radii = fcu._format_radii(radii, self.n_star)
        # Compute the data stars flux and positions for Gaussian Fit
        self.data_stars_pos, self.data_stars_flux, self.data_fwhm_vec = _calc_pos_and_flux(self.data_regions, self.radii, True)
        # Create an empty array to store the std used in weighted average of MF
        self.data_std = np.zeros(self.n_star)
        # Check if the values is smaller than the radius thirds
        for ii, fwhm_vec in enumerate(self.data_fwhm_vec):
            if fwhm_vec[0] <= self.radii[ii]/2:
                self.data_std[ii] = fwhm_vec[0]
            elif fwhm_vec[0] <= 2*self.radii[ii]:
                self.data_std[ii] = self.radii[ii]/2
            else:
                self.data_std[ii] = self.radii[ii]
                print('WARINING: the FWHM of data is bigger than the radius')
        # Create the data region for curve-fit
        if self.psf_fit_type == 'lmsq':
            for ii, data_region in enumerate(self.data_regions):
                # mask the data_region
                mask = pxu.circular_mask(data_region, [data_region.shape[0]//2, data_region.shape[1]//2], self.radii[ii])
                # Remove the elements outside the radius
                data_region_mask = data_region[mask]
                # Squezee into one dimension
                data_region_mask = np.ravel(data_region_mask)
                # Append to the list
                self.data_regions_mask.append(data_region_mask)
        
    def make_estimation(self, est_parm_dic, radii=None, thread_id=None, show_estimators=False, save_estimators=False, show_regions=False, save_regions=False, fullResults=False, outdir=None, gaussResults=False):
        if not thread_id is None:
            temp_ini_filepath = os.path.join(self.temp_dir, self.temp_ini_filetag+"_"+str(thread_id)+'.ini')
            temp_ini_filetag = self.temp_ini_filetag+'_'+str(thread_id)
            temp_sim_filetag = self.temp_sim_filetag+'_'+str(thread_id)
        else:
            temp_ini_filepath = os.path.join(self.temp_dir, self.temp_ini_filetag+'.ini')
            temp_ini_filetag = self.temp_ini_filetag
            temp_sim_filetag = self.temp_sim_filetag
        # Set the radii
        if not radii is None:
            self.update_radii(radii)
        # Create the graphics 
        if show_estimators or save_estimators or show_regions or save_regions:
            graphics = Graphics(outdir=self.outdir)
        # Prepare dictionary
        atmo_param = [
            ("atmosphere", "Seeing",        str(est_parm_dic['seeing']) ),
            ("atmosphere", "Cn2Weights",    np.array2string(est_parm_dic['cn2wei'], separator=", ").replace('\n', '') ),
            ("atmosphere", "Cn2Heights",    np.array2string(est_parm_dic['cn2hei'], separator=", ").replace('\n', '') ),
            ("atmosphere", "WindSpeed",     np.array2string(est_parm_dic['winvel'], separator=", ").replace('\n', '') ),
            ("atmosphere", "WindDirection", np.array2string(est_parm_dic['windir'], separator=", ").replace('\n', '') ),
        ]
        # Check if there are the NCPA:
        if "zCoefStaticOn" in est_parm_dic:
            atmo_param.append(("telescope", "zCoefStaticOn", np.array2string(est_parm_dic['zCoefStaticOn'], separator=", ").replace('\n', '') ))
        # Check if there is Jitter:
        if 'jitter' in est_parm_dic:
            atmo_param.append(("telescope", "jitter_FWHM", np.array2string(est_parm_dic['jitter'], separator=", ").replace('\n', '') ))
        # Write temporary configuration files
        ff.write_ini(temp_ini_filepath, atmo_param, self.data_ini_filepath)
        # Select the GPU
        if self.gpu_queue is None:
            tiptop.gpuSelect(0)
        else:
            gpu_idx = self.gpu_queue.get()
            tiptop.gpuSelect(gpu_idx)
        # Run TIPTOP
        simulation = tiptop.baseSimulation(self.temp_dir, temp_ini_filetag, self.temp_dir, temp_sim_filetag)
        simulation.doOverallSimulation()
        sim_regions = simulation.cubeResultsArray
        # Put the gpu_idx into the gpu_queue so could be used onemore time
        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_idx)
            self.gpu_queue.task_done()
        # Create the empty arrays
        posteriors_regions = np.empty((self.n_star))
        if self.do_err_flux:
            err_flux = np.empty((self.n_star))
            tot_flux = np.empty((self.n_star))
        map_regions = np.empty_like(self.data_regions)
        std_vec = np.empty((self.n_star, 3))
        # Use a serial implementation 
        if self.pool is None or show_estimators or save_estimators or self.do_err_flux:
            # Loop on each star
            for ii, sim_region in enumerate(sim_regions):
                # Use the gaussian fit
                if self.psf_fit_type == 'gauss':
                    # Find centroid and flux
                    sim_star_pos, sim_star_flux, std_vec[ii] = _calc_pos_and_flux(sim_region, self.radii[ii], True)
                    # shift center position
                    sim_region = shift(sim_region, (self.data_stars_pos[ii,1]-sim_star_pos[0], self.data_stars_pos[ii,0]-sim_star_pos[1]))
                    # rescale
                    sim_region *= self.data_stars_flux[ii]/sim_star_flux
                # Use the curve fit
                elif self.psf_fit_type == 'lmsq':
                    if gaussResults:
                        _, _, std_vec[ii] = _calc_pos_and_flux(sim_region, self.radii[ii], True)
                    sim_region = _pos_flux_fit_lmsq(sim_region, self.data_regions_mask[ii], self.radii[ii])
                # Just for the simualtion, no shift and scale
                elif self.psf_fit_type == 'fix':
                    if gaussResults:
                        _, _, std_vec[ii] = _calc_pos_and_flux(sim_region, self.radii[ii], True)
                    sim_region = sim_region
                # Raise an error
                else:
                    raise ValueError('psf_fit_type is not well defined: used %s, expected gauss, lmsq or fix' %self.psf_fit_type)
                # Compute the merit function
                if self.do_err_flux == False:
                    value, map = merit_func(self.data_regions[ii], sim_region, self.radii[ii], self.merit_shape, self.data_std[ii], do_err_flux=False)
                else:
                    value, map, err_flux[ii], tot_flux[ii] = merit_func(self.data_regions[ii], sim_region, self.radii[ii], self.merit_shape, self.data_std[ii], do_err_flux=True)
                if show_estimators or save_estimators:
                    graphics.plot_estimator(self.data_regions[ii], sim_region, map, star_idx=self.stars_idx[ii], radius=self.radii[ii], showfigure=show_estimators, saveflag=save_estimators, outdir=outdir)
                # Add value to the posterior
                posteriors_regions[ii] = value
                # Save Merit map to update plot
                map_regions[ii] = map
        else:
            # Define the array list 
            args_list = [(ii, sim_region, self.psf_fit_type, self.radii, self.data_stars_pos,
                            self.data_stars_flux, self.data_std, self.data_regions, self.data_regions_mask, self.merit_shape, gaussResults)
                            for ii, sim_region in enumerate(sim_regions) ]
            # Start the pool 
            results = self.pool.map(_pool_merit_region, args_list)
            # Unpack the pool results
            for ii, posterior, map_region, std_val in results:
                posteriors_regions[ii] = posterior
                map_regions[ii] = map_region
                std_vec[ii] = std_val
        # Plot regions image
        if show_regions or save_regions:
            graphics.plot_regions(map_regions, self.radii, is_residuals=True, outdir=outdir, showflag=show_regions, saveflag=save_regions)
        # Return the combining of diffent stars' merit
        if fullResults:
            if self.do_err_flux:
                return combine_mf(posteriors_regions), posteriors_regions, map_regions, err_flux, tot_flux
            else:
                return combine_mf(posteriors_regions), posteriors_regions, map_regions
        elif gaussResults:
            if self.do_err_flux:
                return combine_mf(posteriors_regions), posteriors_regions, map_regions, std_vec, sim_regions, err_flux, tot_flux
            else:
                return combine_mf(posteriors_regions), posteriors_regions, map_regions, std_vec, sim_regions
        else:
            return combine_mf(posteriors_regions)


    def _find_best_gpu_order(self, n_threads, process_size_GB):
        # Query nvidia-smi to get information on GPUs usage
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,nounits,noheader"
            ],
            encoding="utf-8"
        )
        # Create an empty list
        gpus = []
        # Cast values into the list
        for line in result.strip().splitlines():
            total, used, free = map(int, line.split(", "))
            gpus.append({
                "total": total/1024, #GB
                "used": used/1024,   #GB
                "free": free/1024,   #GB
            })
        # create an array for the free memory
        free_GB = np.empty(len(gpus))
        # Cast free memory into array
        for i, gpu in enumerate(gpus):
            free_GB[i] = gpu['free']
        # View how many process each GPU could run
        n_proc_gpu = np.floor(free_GB.copy() / process_size_GB)
        # Find the best solution
        if n_threads > 0:
            gpus_used = self._smallest_or_subset_with_indices(n_proc_gpu, n_threads)
        else:
            print(n_threads)
        # Check if there is a solution:
        if gpus_used is None:
            raise ValueError('Requesteed a higher number of threads than the GPU could hold')
        # Un pack the gpu_idx and add to the queue
        gpu_queue = queue.Queue(maxsize=n_threads)
        n_threads_left = n_threads
        for gpu_idx in gpus_used:
            for _ in range(int(n_proc_gpu[int(gpu_idx)])):
                if n_threads_left > 0:
                    gpu_queue.put(int(gpu_idx))
                    n_threads_left -= 1
        # Return the GPU queue
        return gpu_queue

    def _smallest_or_subset_with_indices(self, arr_original, target):
        arr = arr_original.copy().astype(np.int32)
        # Check if a single element > target
        greater_idx = np.where(arr >= target)[0]
        if greater_idx.size > 0:
            i = greater_idx[np.argmin(arr[greater_idx])]
            return [int(i)]
        # If there are not a single gpu free enough try with multipole selectio
        max_sum = arr.sum()

        dp = [False] * (max_sum + 1)
        parent = [None] * (max_sum + 1)  # (previous_sum, index)

        dp[0] = True

        for idx, num in enumerate(arr):
            for s in range(max_sum, num - 1, -1):
                if dp[s - num] and not dp[s]:
                    dp[s] = True
                    parent[s] = (s - num, idx)

        # Find smallest achievable sum >= target
        for s in range(target, max_sum + 1):
            if dp[s]:
                indices = []
                cur = s
                while cur != 0:
                    prev, idx = parent[cur]
                    indices.append(idx)
                    cur = prev
                return indices[::-1]
        # In case no solution as been found return None
        return None