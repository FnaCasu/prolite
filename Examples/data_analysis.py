import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#--- IMPORT LIBRARIES  ---#
import numpy as np # for the mathematical operation
import os # to operate with path strings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # to make plots
from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
from astropy.io import fits

from estimator import Estimator
import files as ff
import format_and_control as fcu
from sensorPosition import SensorPosition
from dataRegions import dataRegions
from graphics import Graphics
from saver import Saver
from merit import Merit
from findStar import findStar
from headerReader import headerReader
from plot_results import plot_results

if __name__ == "__main__":
    def print_log (msg):
        print(msg)

    outdir = 'Output'
    image_dir = '../Palomar10/Data_2026_red'
    image_ftag = 'luci2.20260610.0065.1.p.sks'

    AO_param = {
        'fwhm': 20,
        'thr' : 55000,
        'min_sep': 15
    }
    field_param = {
        'fwhm' : 8,
        'thr' : 50,
        'min_sep' : 30,
    }
    nField = 100

    box_edge = 200


    # Estimator setup
    request_cn2_height = np.array([-100, 100, 5000, 14000])

    bounds = np.array([[0.5, 1.7],   # Seeing
                    [0.01, 0.95], # Cn2 weight ground
                    [0.05, 15],    # Wind Velocity ground
                    [0, 360],   # Wind direction ground
                    [0.05, 0.95], # Ratio between mid and high
                    [2, 30],  # Wind velocity mid
                    [0, 360],   # Wind direction mid
                    [0.005, 0.95], # Ratio between mid2 and high
                    [2, 40],  # Wind velocity mid2
                    [0, 360],   # Wind direction mid2
                    [15, 60],  # Wind velocity high
                    [0, 360],   # Wind direction high
                    ])
    max_abberation_vec = np.array([0, 100, 200, 80, 80, 0])
    max_jitter_vec = np.array([80, 20])

    psf_fit_type = 'gauss'    # lmsq, gauss
    merit_shape  = 'gauss'   # fix, gauss, linear
    est_type     = 'dual'    # dual, minimize, diff, pyGAD

    estimator_setup_dictionary ={  
                'method' : 'Nelder-Mead',
                'initial_temp' : 4000, 
                'temp_ratio' : 5e-3, 
                'visit' : 2.9, 
                'accept' : -7, 
                'maxfun' : 10, 
                'tol' : 5e-2, 
                'maxiter' : 10, 
                'no_local_search' : False
                }


    # Default file names
    outdir_findstar_name = 'FindStar'
    all_stars_pos_fname = 'all_pos_field'
    ao_star_pos_fname = 'pos_ao'

    # Create the path
    image_fpath = os.path.join(image_dir, image_ftag+'.fits')
    outdir_findStar = os.path.join(outdir, outdir_findstar_name)

    # Create the directory
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if not os.path.isdir(outdir_findStar):
        os.makedirs(outdir_findStar)

    #============= DATA IMPORT =======================#
    # Read data
    print_log('Import data')
    data_hdu_list = fits.open(image_fpath)
    data = data_hdu_list[0].data
    if len(data.shape)>2:
        data = data[0]

    # Read Header
    hdr = headerReader(data_hdu_list[0].header)
    data_hdu_list.close()

    tel_res = hdr.tel_res
    pixscale = hdr.pixscale
    telescope_alt = hdr.telescope_alt
    AO_freq = hdr.AO_freq
    n_photo =  hdr.n_photo / 20.
    wavelenght = hdr.wavelenght


    # Save Image
    print_log('Save original figure')
    fig = plt.figure(figsize=(6.5,6))
    im = plt.imshow(np.log(data), origin='lower', cmap='inferno', aspect=1, vmin=0)
    plt.suptitle(r'\textbf{Original data on '+image_ftag+'}')
    plt.ylabel(r'y position [px]')
    plt.xlabel(r'x position [px]')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.ylabel(r'photons counts [log (ADU)]')
    fig.savefig(os.path.join(outdir_findStar, 'Original_image.png'), dpi=400, bbox_inches='tight')


    #================ FIND STARS ======================#
    # Find stars
    print_log('Run findstar')
    stars = findStar(data, AO_param, field_param, nField, outdir=outdir_findStar)

    ## DEVE ANDARE TUTTA DENTRO A FIND STAR MODULE, forse anche l'immagine dei dati
    if True:
        # Save star position
        print_log('Write position on file')
        np.savetxt(os.path.join(outdir_findStar, all_stars_pos_fname+'.txt'), stars.pos_field, fmt='%.5f')
        np.savetxt(os.path.join(outdir_findStar, ao_star_pos_fname+'.txt'), stars.pos_AO, fmt='%.5f')

        # Plot star position
        print_log('Plot star position')
        sensor_position = SensorPosition(tel_res, pixscale)
        AOstar_zen, AOstar_azi = sensor_position.px2polar(stars.pos_AO)
        star_zen, star_azi = sensor_position.px2polar(stars.pos_field)
        fig = sensor_position.plot_polar(star_zen, star_azi)
        fig = sensor_position.plot_polar(AOstar_zen, AOstar_azi, fig)
        plt.grid()
        fig.savefig(os.path.join(outdir_findStar, 'All_star_position.png'), dpi=400)

        # Plot data regions
        print_log('Plot and save data regions')
        data_regions = dataRegions(data, stars.pos_field, 100)
        graphic = Graphics(outdir_findStar)
        graphic.plot_regions(data_regions.data_regions, radii=25, saveflag=True)




    # Set the random generator
    rng_seed = random.randint(10, 100000)

    n_threads = 1
    gpu_list = [1]


    #============== ESTIMATOR ==================#

    # Directory and file names
    generic_ini_dir = 'Examples/Configuration'
    generic_ini_ftag = 'SOUL_generic'
    data_ini_ftag = 'data_temp'
    outdir_analysis_name = 'Analysis_1'

    # Path creation
    outdir_analysis = os.path.join(outdir, outdir_analysis_name)
    fpath_ao_pos = os.path.join(outdir_findStar, ao_star_pos_fname+'.txt')
    fpath_all_pos = os.path.join(outdir_findStar, all_stars_pos_fname+'.txt')
    generic_ini_fpath = os.path.join(generic_ini_dir, generic_ini_ftag + '.ini')
    data_ini_fpath = os.path.join(outdir_analysis, data_ini_ftag + '.ini')


    # Create the directory
    if not os.path.isdir(outdir_analysis):
        os.makedirs(outdir_analysis)



    # STARS POSITION 
    # Set stars inidecs
    chosen_indices = [8, 9, 10, 12, 14, 16, 17, 27, 28]
    radii = np.array([15, 13, 15, 12, 15, 12, 12, 12, 12])
    avoid_remain_indices = [116,185,189,0,1,2,3,4,5,6,10,13,15,18,19,22,24,25,31,33,35,36,37,38,40,42,43,44,48,49,50,51,54,55,56,58,59,60,62,63,65,66,67,68,69,71,72,74,76,78,79,80,81,82,85,86,87,88,89,90,91,93,94,95,96,97,98,100,101,102,103,104,105,106,108,110,111,112,114,114,115,118,120,121,122,123,125,126,127,129,130,131,134,137,138,139,140,141,144,147,148,149,150,151,152,153,155,157,158,156,160,161,162,163,164,166,167,169,170,171,172,174,175,180,182,183,184,186,187,188,190,191,192,193,196,197,199,201,203,206,207,208,209,210,211,212,213,214,215,217,218,221,222,223,224,225,226,227,228,229,235,238,239,240,241,242,243,244,245,246,247,248,249,252,253,254,255,256,257,259,260,261,262,265,266,268,269,270,271,272,274,275,276,277,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299]
            
    # Import data 
    pos_AO = np.loadtxt(fpath_ao_pos).reshape((1,2))
    all_pos_field = np.loadtxt(fpath_all_pos)

    # Create the array
    pos_field = all_pos_field[chosen_indices]
    pos_remaining_field = np.delete(all_pos_field.copy(), chosen_indices, axis=0)
    remaining_indices = np.delete(np.arange(all_pos_field.shape[0]), chosen_indices, axis=0)

    # Convert in polar coordiantes
    sensor_position = SensorPosition(tel_res, pixscale)
    AOstar_zen, AOstar_azi = sensor_position.px2polar(pos_AO)
    star_zen, star_azi = sensor_position.px2polar(pos_field)
    rem_star_zen, rem_star_azi = sensor_position.px2polar(pos_remaining_field)

    # Plot star position
    fig = sensor_position.plot_polar(rem_star_zen, rem_star_azi, Number=remaining_indices, c='#808080', label='Other stars')
    fig = sensor_position.plot_polar(star_zen, star_azi, fig, Number=chosen_indices, c='r', label='Used stars')
    fig = sensor_position.plot_polar(AOstar_zen, AOstar_azi, fig, Number=False,c='b', label='AO star')
    plt.grid()
    plt.legend()
    fig.savefig(os.path.join(outdir_analysis, 'All_star_position.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # Counts the stars
    nstars = pos_field.shape[0]



    # CREATE DATA REGIONS
    data_regions = dataRegions(data, pos_field, box_edge)

    # START THE GRAPHICS
    graphic = Graphics(outdir_analysis)
    graphic.plot_regions(data_regions.data_regions, radii=radii, saveflag=True)

    # CREATE THE INI FILE
    radii = fcu._format_radii(radii, nstars)
    star_param = [
        ("sensor_science", "FieldOfView", box_edge),
        ("sensor_science", "PixelScale", 1e3*pixscale),
        ("telescope", "Resolution", box_edge),             # pixels
        ("telescope", "ZenithAngle", 90-telescope_alt),   # degrees
        ("sources_HO", "Zenith", str([AOstar_zen.tolist()]) ),          # arcsec
        ("sources_HO", "Azimuth", str([AOstar_azi.tolist()]) ),         # degrees
        ("sources_science", "Wavelength", str([wavelenght])),
        ("sources_science", "Zenith", np.array2string(star_zen, separator=", ", precision=8).replace('\n', '') ),         # arcsec
        ("sources_science", "Azimuth", np.array2string(star_azi, separator=", ", precision=8).replace('\n', '') ),        # degrees
        ("RTC", "SensorFrameRate_HO", AO_freq),           # Hz
        ("sensor_HO", "NumberPhotons", n_photo)
    ]
    ff.write_ini(file_in=generic_ini_fpath, file_out=data_ini_fpath, param_dic=star_param)

    ## START THE ESTIMATOR
    estimator = Estimator(data_regions.data_regions, radii, request_cn2_height, data_ini_fpath, outdir_analysis, psf_fit_type=psf_fit_type, merit_shape=merit_shape)

    # Set the estimation
    estimator.set_estimation(est_type = est_type,                             
                            estimator_setup_dictionary = estimator_setup_dictionary,  
                            bounds = bounds , 
                            max_abberation_vec = max_abberation_vec, 
                            max_jitter_vec = max_jitter_vec, 
                            x0 = None, 
                            rng_seed = rng_seed,
                            radii = None, 
                            doAnimation = None, 
                            sim_param = None, 
                            resume = False, 
                            n_threads = n_threads, 
                            gpu_list = gpu_list)

    # Initialize the SAVER
    saver = Saver(outdir_analysis, generic_ini_fpath, data_filepath=image_fpath, headerReader=hdr, findStar=None, dataRegions=data_regions,  
                    estimator=estimator, graphics=graphic, simulation=None, filetag='save_status', verbose=True, selected_stars=chosen_indices,
                    avoid_stars=avoid_remain_indices, fpath_ao_pos=fpath_ao_pos, fpath_all_pos=fpath_all_pos)
    saver.save()

    # Start the estimator class
    est_parm_dic, merit, merit_stars = estimator.start_estimation()

    saver.save()

    # Run the final minimizer
    estimator.set_estimation(est_type = 'min',
                            bounds = bounds , 
                            max_abberation_vec = max_abberation_vec, 
                            max_jitter_vec = max_jitter_vec, 
                            x0 = estimator.xv_final, 
                            rng_seed = rng_seed,  
                            estimator_setup_dictionary = {  
                                        'method' : 'Nelder-Mead',
                                        'tol' : 5e-1, 
                                        'maxiter' : 5
                                        }  
                            )

    est_parm_dic, merit, merit_stars = estimator.start_estimation()

    saver.save()

    plot_results(outdir_analysis)
    