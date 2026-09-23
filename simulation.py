import tiptop.tiptop as tiptop
from tiptop.tiptop import plot_directions  # to make PSF simulation
import numpy as np # for the mathematical operation
import os # to operate with path strings
from astropy.io import fits     # to handle fits files
from scipy.ndimage import shift

from files import write_ini
import pixelUtiles as pxu
from sensorPosition import SensorPosition

class Simluation:
    def __init__(self, AO_star_zenazi, stars_zenazi, atmo_param_dic, tele_hdr, generic_ini_filepath, outdir, cn2hei = None, isPositionInPixels=False, wavelength=2.161e-06, wavelenght_weight=None):
        self.tel_hdr = tele_hdr
        self.tel_res = tele_hdr['tel_res']
        self.generic_ini_filepath = generic_ini_filepath
        self.outdir = outdir
        self.wavelength = wavelength
        self.wavelenght_weight = wavelenght_weight

        # Check the Cn2 heights values
        if not 'cn2hei' in atmo_param_dic:
            if not cn2hei is None:
                atmo_param_dic['cn2hei'] = cn2hei
            else:
                raise ValueError('Need the Cn2 heights to perform the simulation')
        
        # Prepare usefull arrays
        self.image = np.zeros((self.tel_res, self.tel_res))
        self.data_regions = None
        # Store the position
        sensor = SensorPosition(self.tel_res, tele_hdr['pxscale'])
        if isPositionInPixels:
            self.star_pos = stars_zenazi
            self.AO_star_pos = AO_star_zenazi
            stars_zenazi = np.array(sensor.px2polar(stars_zenazi))
            AO_star_zenazi = np.array(sensor.px2polar(AO_star_zenazi))
        else:
            self.star_pos = np.array(sensor.polar2px(stars_zenazi))
            self.AO_star_pos = np.array(sensor.polar2px(AO_star_zenazi))

        # Create dictionary for write TIPTOP configuration file
        self.param_dic = [
            ("telescope", "ZenithAngle",    90-tele_hdr['telescope_alt']),   # degrees
            ("sources_HO", "Zenith",        np.array2string(AO_star_zenazi[0], separator=", ")), # arcsec
            ("sources_HO", "Azimuth",       np.array2string(AO_star_zenazi[1], separator=", ")), # degrees
            ("sources_science", "Wavelength",wavelength), # arcsec
            ("sources_science", "Zenith",   stars_zenazi[0] ), # arcsec
            ("sources_science", "Azimuth",  stars_zenazi[1] ), # degrees
            ("telescope", "Resolution",     200),             # pixels
            ("RTC", "SensorFrameRate_HO",   tele_hdr['AO_freq']),           # Hz
            ("sensor_HO", "NumberPhotons",  [tele_hdr['n_photo']]),
            ("atmosphere", "Seeing",        atmo_param_dic['seeing'] ),
            ("atmosphere", "Cn2Weights",    atmo_param_dic['cn2wei'] ),
            ("atmosphere", "Cn2Heights",    atmo_param_dic['cn2hei'] ),
            ("atmosphere", "WindSpeed",     atmo_param_dic['winvel']  ),
            ("atmosphere", "WindDirection", atmo_param_dic['windir'] ),
        ]
        if "zCoefStaticOn" in atmo_param_dic:
            self.param_dic.extend([
                ("telescope", "zCoefStaticOn", atmo_param_dic['zCoefStaticOn']),
            ])
            print('Add Non-common-path errors')
        if "jitter" in atmo_param_dic:
            self.param_dic.extend([
                ("telescope", "jitter_FWHM", atmo_param_dic['jitter'] ),
            ])
        # Create the configuartion file for the AO star
        self.param_dic_ao = [
            ( 'sources_science', 'Zenith', np.array2string(AO_star_zenazi[0], separator=", ") ),
            ( 'sources_science', 'Azimuth', np.array2string(AO_star_zenazi[1], separator=", ") ),
        ]
        seed = os.getenv('GUIDE_RANDOM_SEED', None)

        if seed is not None:
            seed = int(seed)
        self.noise_rng = np.random.default_rng(seed)

        self.n_star = stars_zenazi.shape[1]

    def combine_psf_over_wavelength(self, cube, weights):
        # Check if there is more wavelenght
        if len(cube.shape) == 4 :
            # Get number of wavelenghts and stars
            n_wave, n_star = cube.shape[0], cube.shape[1]
            if weights is None:
                # Plain (unweighted) sum over wavelength -- all weights = 1
                combined = np.einsum('ws...->s...', cube)
            else:
                if weights.shape == (n_wave,):
                    # Same weight per wavelength, shared across all stars
                    combined = np.einsum('w,ws...->s...', weights, cube)
                elif weights.shape == (n_wave, n_star):
                    # Independent weight per (wavelength, star)
                    combined = np.einsum('ws,ws...->s...', weights, cube)
                else:
                    raise ValueError(
                        f'weights must be None, shape (Nw,)=({n_wave},), or shape '
                        f'(Nw, Ns)=({n_wave}, {n_star}); got shape {weights.shape}'
                    )   
            return combined      
        elif len(cube.shape) == 3:
            return cube
        else:
            raise ValueError('Unexpected output shape of the cube array from TIPTIP')

    def simulate_no_background(self, generic_ini_filepath=None, outdir=None, sim_shift=None, sim_maxhei=None, doAOstar=True ):
        # Generate random shift to the stars
        if sim_shift is None:
            sim_shift = 2*(np.random.rand(self.n_star, 2)-0.5)
        # Generate random flux to the stars
        if sim_maxhei is None:
            sim_maxhei = 1e4*(1+10*np.random.rand(self.n_star))
        if sim_maxhei is False:
            sim_maxhei = np.full(self.n_star, 1e4)
        # Set file names
        if outdir is None:
            outdir = self.outdir
        temp_dir = os.path.join(outdir, 'Temp')
        sim_data_filename = 'sim_data'
        temp_ini_filename = 'temp_config'
        temp_ini_filename_ao = 'temp_config_ao'
        if generic_ini_filepath is None:
            generic_ini_filepath = self.generic_ini_filepath
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        # Construct the full path
        temp_ini_file = os.path.join(temp_dir, temp_ini_filename+'.ini')
        temp_ini_file_ao = os.path.join(temp_dir, temp_ini_filename_ao+'.ini')
        # Make Field stars regions        
        # Write configuration file
        write_ini(file_in=generic_ini_filepath, file_out=temp_ini_file, param_dic=self.param_dic)
         # Crate TIPTOP instanse
        tiptop.gpuSelect(0)
        simulation = tiptop.baseSimulation(temp_dir, temp_ini_filename, outdir, sim_data_filename)
        # Run TIPTOP simulation
        simulation.doOverallSimulation()
        # Get data
        self.data_regions = self.combine_psf_over_wavelength(simulation.cubeResultsArray, self.wavelenght_weight)
        # Save Population to file
        with open(os.path.join(self.outdir, 'simulated_regions_no_shift_no_scale.npy'), 'wb') as file:
            np.save(file, self.data_regions, allow_pickle=False)
        # Manipulate each data regions
        for ii in range(self.n_star):
            # Shift center position
            if not sim_shift is False:
                self.data_regions[ii] = shift(self.data_regions[ii], (sim_shift[ii, 1], sim_shift[ii, 0]))
            # rescale
            self.data_regions[ii] = sim_maxhei[ii]/np.nanmax(self.data_regions[ii]) * self.data_regions[ii]
        # Save the data regions
        with open(os.path.join(self.outdir, 'simulated_regions.npy'), 'wb') as file:
            np.save(file, self.data_regions, allow_pickle=False)
        # Add the Poisson error on just the 
        poisson_regions = self.poisson_noise(self.data_regions)
        # Save the data regions
        with open(os.path.join(self.outdir, 'simulated_regions_poisson.npy'), 'wb') as file:
            np.save(file, poisson_regions, allow_pickle=False)
        # Add the bais
        bais_vec = np.random.uniform(0, 100, self.n_star)
        bais_regions = self.data_regions + bais_vec[:, np.newaxis, np.newaxis]
        poisson_bais_regions = self.poisson_noise(bais_regions)
        # Save the data regions
        with open(os.path.join(self.outdir, 'simulated_regions_bais_poisson.npy'), 'wb') as file:
            np.save(file, poisson_bais_regions, allow_pickle=False)
        # Make Ao star region
        if doAOstar:
            # Write configuration file
            write_ini(file_in=temp_ini_file, file_out=temp_ini_file_ao, param_dic=self.param_dic_ao)
            # Run TipTop simulation
            simulation = tiptop.baseSimulation(temp_dir, temp_ini_filename_ao, outdir, sim_data_filename)
            simulation.doOverallSimulation()
            # Get data
            self.AO_region = self.combine_psf_over_wavelength(simulation.cubeResultsArray, self.wavelenght_weight)[0]
            if not sim_shift is False:
                self.AO_region = shift(self.AO_region, (self.noise_rng.normal(0, 3, 1), self.noise_rng.normal(0, 3, 1)))
            # rescale
            self.AO_region = 35*np.nanmax(self.data_regions)/np.nanmax(self.AO_region) * self.AO_region
        # return
        return self.data_regions

    def create_image(self, doError=False):

        self.simulate_no_background()

        region_dim = self.data_regions.shape[1]
        tel_res = self.tel_res
        stars_im = np.zeros_like(self.image)
        for ii in range(self.n_star):
            xx = int(self.star_pos[ii, 0])
            yy = int(self.star_pos[ii, 1])
            # Find regions limits
            MA_lim, mi_lim = pxu.find_bound_regions(xx, yy, tel_res, region_dim)
            # Sum count
            stars_im[MA_lim[1,0]:MA_lim[1,1], MA_lim[0,0]:MA_lim[0,1]] += self.data_regions[ii, mi_lim[1,0]:mi_lim[1,1], mi_lim[0,0]:mi_lim[0,1]]
        # add AO star
        xx = int(self.AO_star_pos[0,0])
        yy = int(self.AO_star_pos[0,1])
        # Find regions limits
        MA_lim, mi_lim = pxu.find_bound_regions(xx, yy, tel_res, region_dim)
        # Sum count
        stars_im[MA_lim[1,0]:MA_lim[1,1], MA_lim[0,0]:MA_lim[0,1]] += self.AO_region[mi_lim[1,0]:mi_lim[1,1], mi_lim[0,0]:mi_lim[0,1]]
        
        self.image = stars_im.copy()

        if doError:
            # Generate bais Current
            bias_val = 0
            n_col = 0
            bias_im = self.bias(bias_val, n_col)
            # Generate dark Current
            current = 0
            exp_time = 0
            dark_im = self.dark_current(current, exp_time)
            # Define a flat image
            flat_im = self.flat()
            # Define hot pixel
            rate = 1e-5
            hot_im = self.hot_pixels(rate)
            # Add sky sky_background
            sky_counts = 0
            grad_count = 0
            grad_dir = 0
            sky_im = self.sky_background(sky_counts, grad_count, grad_dir)
            # Generate Readout noise
            read_noise_value = 5
            noise_im = self.read_noise(read_noise_value)

            # Combine images and apply the
            th_image = bias_im + dark_im + flat_im * (stars_im)
            self.image = noise_im + self.noise_rng.poisson(th_image)

        return self.image

    def poisson_noise(self, images, gain=1.0, seed=None):
        # Reshape as an np.array
        images = np.asarray(images)
        # Check the dimension of the array to work whit 3D
        if images.ndim == 2:
            images = images[np.newaxis, ...]
            squeeze_output = True
        elif images.ndim == 3:
            squeeze_output = False
        else:
            raise ValueError(f'Expected a 2D or 3D array, got shape {images.shape}')
        # Start the random generator
        rng = np.random.default_rng(seed)
        # Shot noise requires non-negative photon/electron counts
        images_clipped = np.clip(images, 0, None)
        # compute the electron number 
        electrons = images_clipped * gain
        # Apply Noise (is not an adding as in the Gaussian case)
        noisy_electrons = rng.poisson(electrons)
        # Convert back to ADU
        noisy_images = noisy_electrons / gain
        # Return the 2D image 
        if squeeze_output:
            return noisy_images[0]
        # Return the 3D vector of images
        return noisy_images

    def read_noise(self, amount, gain=1):
        noise = self.noise_rng.normal(scale=amount/gain, size=(self.tel_res, self.tel_res))
        return noise

    def bias(self, bg_value, n_col=0):
        bias_im = np.full_like(self.image, bg_value)
        if n_col>0:
            # This MUST not change
            rng = np.random.RandomState(seed=3648)
            columns = rng.randint(0, self.tel_res, size=n_col)
            # This adds a little random-looking noise into the data.
            col_pattern = rng.randint(0, int(0.1 * bg_value), size=self.tel_res)
            # Apply to image
            for c in columns:
                bias_im[:, c] += col_pattern
        return bias_im

    def dark_current(self, current, exposure_time, gain=1.0):
        # Calc the base current in the ccd
        dark_im = current * exposure_time / gain
        return dark_im

    def hot_pixels(self, rate=1e-5):
        # Find a number for hot pixels
        y_max, x_max = self.image.shape
        n_hot = int(1e-5 * x_max * y_max)
        # The position MUST not change
        rng = np.random.RandomState(358476)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)
        # Ue an extremily high value for their counts
        hat_val = max(100*np.nanmax(self.image), 1e6)
        # Map the hot pixels
        hot_im = np.zeros_like(self.image)
        hot_im[(hot_y, hot_x)] = hat_val
        return hot_im

    def sky_background(self, sky_counts, gradient_count=0, gradient_dir=0, gain=1):
        sky_im = np.full_like(self.image, sky_counts * gain)
        ### TODO: Implementare i gradienmti nel cielo
        if gradient_count > 0:
            print('Da fare')
        return sky_im

    def flat(self):
        flat_im = np.full_like(self.image, 1)
        return flat_im


    def save_image(self, outdir=None, fname='simulated_images'):
        if outdir is None:
            outdir = self.outdir
        # header
        params = {
            "PIXSCALE" : self.tel_hdr['pxscale'],
            "AO_FREQ"  : self.tel_hdr['AO_freq'],
            "AOCOUNTS" : self.tel_hdr['n_photo'],
            "TELALT"   : self.tel_hdr['telescope_alt']
        }
        # New ImageHDU with final images
        primary_hdu = fits.PrimaryHDU(data=self.image)
        # Casting valus in header
        for k, v in params.items():
            primary_hdu.header[k] = v
        # Creation of a HDUList
        hdul = fits.HDUList(primary_hdu)
        # saving
        filepath = os.path.join(outdir, fname+'.fits')
        hdul.writeto(filepath, overwrite=True)
