import numpy as np
import os # to operate with path strings

import scipy.optimize as opt
import pygad

import threading
from tqdm import tqdm
import time

from graphics import Graphics
import format_and_control as fcu
import matplotlib.pyplot as plt
from merit import Merit
plt.rcParams['text.usetex'] = True


class Estimator:
    def __init__(self, data_regions, radii, Cn2Heights, data_ini_filepath, outdir=None, doAnimation=False, sim_param=None, 
                save_cycle_status=True, save_merit_stars=True, resume=False, psf_fit_type='gauss', merit_shape='linear'):
        # Save inputs
        self.type = None
        self.data_regions = data_regions
        self.Cn2Heights = np.array(Cn2Heights)
        self.data_ini_filepath = data_ini_filepath
        self.outdir = outdir
        self.doAnimation = doAnimation
        self.sim_param = sim_param
        self.save_cycle_status = save_cycle_status
        self.save_merit_stars = save_merit_stars
        self.resume = resume
        self.psf_fit_type = psf_fit_type
        self.merit_shape = merit_shape
        # Use the Parallel PSF matching
        self.use_parallel_PSF_matching = True
        # Compute stars number
        self.n_star  = data_regions.shape[0]
        # Find the size of data regions
        self.region_width = data_regions.shape[1]
        # Format the radii for estimation
        self.radii = fcu._format_radii(radii, self.n_star)
        # Calc number of layers
        self.n_layer = self.Cn2Heights.shape[0]
        self.n_param = 3
        # Set default values for ncpa and jitter
        self.max_abberation_vec = None
        self.max_jitter_vec = None
        # Set default values
        self.gpu_list = None
        self.process_size_GB = 1.5
        self.n_threads = 1
        # Pygad 
        self.MIN_FITNESS = -1e40
        self.xv_pop = None
        self.generation_rate = 40
        # Defining all the paths and file names
        temp_dir = 'Temp'                                           # Directory that host all the temporary files
        self.temp_ini_filetag = 'temp_config'                       # Temporary configuration file with all parameters
        self.temp_sim_filetag = 'temp_TIPTOP_sim_output'            # Output file of TIPTOP, doesn't change anyting as no file is saved
        self.pygad_save_filetag = 'pyGAD_save'                      # Saving file for pyGAD save function
        self.parameters_mf_filetag = 'estimator_cycle_parameters'   # Output file with all best guess for the parameters at each generation
        self.stars_mf_filetag = 'merits_stars_cycle_parameters'     # Output file with all best solution merit functions for each generation
        self.population_filetag = 'pygad_last_population'           # Output file for last used population, used to resume the work
        # Create output directory
        if self.outdir is None:
            self.outdir = os.path.join(os.getcwd(), 'Output', 'Estimator_results')
            dir_idx = 0
            while os.path.isdir(self.outdir):
                self.outdir = os.path.join(os.getcwd(),'Output', 'Estimator_results_'+str(dir_idx))
                dir_idx += 1
            os.makedirs(self.outdir)
        else:
            if not os.path.isdir(self.outdir):
                os.makedirs(self.outdir)
        # Define file paths
        self.temp_dir = os.path.join(outdir, temp_dir)
        self.temp_ini_filepath = os.path.join(self.temp_dir, self.temp_ini_filetag+'.ini')
        self.ga_saver_filepath = os.path.join(self.outdir, self.pygad_save_filetag)
        self.pop_save_filepath = os.path.join(self.outdir, self.population_filetag + '.npy')
        # Check if Temp directory exist and create it if necessary
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)
        # Define the Merit function
        self.merit = Merit(self.data_regions, self.radii, self.outdir, self.temp_dir, self.temp_ini_filetag, self.temp_sim_filetag, 
                            self.data_ini_filepath, psf_fit_type = self.psf_fit_type, merit_shape=self.merit_shape)
        # Create file to save cycle status
        if self.save_cycle_status:
            self.cycle_status_filepath = os.path.join(self.outdir, self.parameters_mf_filetag + '.txt')
            if not resume:
                # Check if files arledy exits
                file_idx = 0
                while os.path.isfile(self.cycle_status_filepath):
                    self.cycle_status_filepath = os.path.join(self.outdir, self.parameters_mf_filetag + '_' + str(file_idx) + '.txt')
                    file_idx += 1
                # Create file
                f = open(self.cycle_status_filepath, "w")
                f.close()
        # Create a file to save merits function for each stars
        if self.save_merit_stars:
            self.merit_stars_cycle_filepath = os.path.join(self.outdir, self.stars_mf_filetag + '.txt')
            if not resume:
                # Check if files arledy exits
                file_idx = 0
                while os.path.isfile(self.merit_stars_cycle_filepath):
                    self.merit_stars_cycle_filepath = os.path.join(self.outdir, self.stars_mf_filetag + '_' + str(file_idx) + '.txt')
                    file_idx += 1
                # Create file
                f = open(self.merit_stars_cycle_filepath, "w")
                f.close()
        # Initialize the UI class
        if self.doAnimation == True:
            self.graphics = Graphics(outdir=self.outdir, sim_param=self.sim_param)
        # Define the time variables
        self.start_time = -1
        self.end_time = -2


    def set_estimation(self, est_type, estimator_setup_dictionary, bounds, max_abberation_vec=None, max_jitter_vec=None, x0=None, rng_seed=None, 
                       radii=None, doAnimation=None, sim_param=None, resume=True, n_threads=1, gpu_list=None):
        

        # === Common part
        self.resume = resume
        # Update bounds values
        if fcu._check_bounds(bounds):
            self.bounds = bounds
        # Store the information for the selected maximum NCPA
        self.max_abberation_vec = max_abberation_vec
        # Store the information for the jitter estimation
        self.max_jitter_vec = max_jitter_vec
        # Create the bounds with also the NCPA
        self.bounds, self.selected_NCPA_mode, self.do_jitter =fcu._make_bounds(self.bounds,  self.max_abberation_vec, self.max_jitter_vec)
        # Store do animation flag
        if not doAnimation is None:
            self.doAnimation = doAnimation
        # Format radii
        if not radii is None:
            self.radii = fcu._format_radii(radii, self.n_star)
            self.merit.update_radii(radii)
        # Update sim parameters
        if not sim_param is None:
            self.sim_param = sim_param
        
        # Specific estimator setup
        if est_type in ['dual', 'dual_annealing', 'annealing']:
            self.type = 'dual_annealing'
            self.set_estimation_dual(rng_seed=rng_seed, x0=x0, **estimator_setup_dictionary)

        elif est_type in ['minimize', 'min']:
            self.type = 'minimize'
            self.set_estimation_minimize(x0=x0, **estimator_setup_dictionary)

        elif est_type in ['diff','differntial']:
            self.type = 'differential_evolution'
            self.set_estimation_differential(x0=x0, rng_seed=rng_seed, n_threads=n_threads, gpu_list=gpu_list, **estimator_setup_dictionary)

        elif est_type in ['pygad', 'PyGad', 'PyGaD', 'pyGAD']:
            self.type = 'pyGAD'
            self.set_estimation_pygad(rng_seed, n_threads, gpu_list, **estimator_setup_dictionary)

        # Print estimator
        print('Estimator used: ', self.type)
        

    # TODO: se voglio usare un resume questo non deve essere necessario
        # Create results file
        if self.save_cycle_status:
            res_str = '# n_cycle  seeing\t'
            for ii in range(self.n_layer):
                res_str += 'Cn2Weight-lr%d  '%ii
                res_str += 'WindVeloc-lr%d  '%ii
                res_str += 'WindDirec-lr%d  '%ii
            res_str += ' Merit\n'
            with open(self.cycle_status_filepath, "a") as f:
                f.write(res_str)
        # Create merits stars file
        if self.save_merit_stars:
            mer_str = '# n_cycle  MF\t\t\t\t'
            for ii in range(self.n_star):
                mer_str += 'MF-star%d\t\t\t\t'%(ii)
            mer_str += '\n'
            with open(self.merit_stars_cycle_filepath, "a") as f:
                f.write(mer_str)
        # Create empty array to store the generation best solution
        self.parameters_vec = np.empty((0, self.bounds.shape[0]))
        self.post_vec = np.array([])
        self.post_star_vec = np.empty((0, self.n_star))

    def set_estimation_dual(self, rng_seed,  method = 'Nelder-Mead', initial_temp = 5230.0, temp_ratio=2e-05, visit= 2.62, accept=-5.0, 
                            maxfun = 1e5, no_local_search = False, tol=1e-2, maxiter=1e4, x0=None):
        # Store data
        self.maxiter = maxiter 
        self.initial_temp = initial_temp 
        self.temp_ratio = temp_ratio 
        self.visit = visit
        self.accept = accept 
        self.maxfun = maxfun 
        self.rng_seed = rng_seed 
        self.no_local_search = no_local_search
        self.x0 = x0
        self.tol = tol
        # Save the method
        METHOD_LIST = ['Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC']
        if method in METHOD_LIST:
            self.method = method
        else:
            raise ValueError('Minimize method %s is not bounds-contraints compatible.'%method)
        
    def set_estimation_minimize(self, x0, method='Nelder-Mead', tol=1e-2, maxiter=1e4):
        # Store the data
        self.x0 = x0
        self.tol = tol 
        self.maxiter = maxiter
        # Check and save method
        METHOD_LIST = ['Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC']
        if method in METHOD_LIST:
            self.method = method
        else:
            raise ValueError('Minimize method %s is not bounds-contraints compatible.'%method)
    
    def set_estimation_differential(self, x0, n_threads, gpu_list, rng_seed, maxiter=1e4, tol=0.5, popsize=100, mutation=(0.5, 1), cross_prob=0.7, 
                       strategy='best1bin', init_mode='latinhypercube', update_mode='immediate'):
        self.x0 = x0
        self.rng_seed = rng_seed
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        # Check that is in [0, 2)
        self.mutation = mutation
        # Check that is in [0, 1)
        self.cross_prob = cross_prob
        # Check the strategy exits
        self.strategy = strategy
        # Check the initial mode exists
        self.init_mode = init_mode
        # Check update mode exits
        self.update_mode = update_mode
        # Get n_threads from gpu_list
        if not gpu_list is None:
            len_gpu = len(gpu_list)
            if len_gpu != n_threads:
               print('BYPASS the n_threads values of %d and set to %d as gpu_list lenght' %(n_threads, len_gpu))
            print('Used the gpu list:', gpu_list)
            self.n_threads = len_gpu
            self.gpu_list = gpu_list
        else:
            self.n_threads = n_threads

    def set_estimation_pygad(self, rng_seed, n_threads, gpu_list, resume=False, num_generations=None, sol_per_pop=None, 
                       num_parents_mating=None, parent_selection_type="rws", K_tournament=None, crossover_type="two_points", crossover_probability=0.8, 
                       keep_elitism=None, keep_parents=0, initial_population=None, mutation_type="random", mutation_percent_genes=10, 
                       random_mutation_min_val=-0.5, random_mutation_max_val=+0.5, sample_size=1000):
        
        # Update random number generator seed
        if not rng_seed is None:
            self.rng_seed = rng_seed
        
        # Compute the numbers of genes by the size of bounds
        num_genes = self.bounds.shape[0]
        # Set the numbers of generation:
        if num_generations is None:
            num_generations = max(100, 5 * num_genes)
        # Set the number of solutions for each population
        if sol_per_pop is None:
            sol_per_pop = max(20, 4 * num_genes) 
        # Set the number of parents mating
        if num_parents_mating is None:
            num_parents_mating = max(10, 2*num_genes)
        # Set the number of bests individuals to keep
        if keep_elitism is None:
            keep_elitism = max(2, sol_per_pop // 10)
        # Get n_threads from gpu_list
        if not gpu_list is None:
            len_gpu = len(gpu_list)
            if len_gpu != n_threads:
               print('BYPASS the n_threads values of %d and set to %d as gpu_list lenght' %(n_threads, len_gpu))
            print('Used the gpu list:', gpu_list)
            self.n_threads = len_gpu
            self.gpu_list = gpu_list
        else:
            self.n_threads = n_threads
        # Check if parent_selection_type has an allowed value
        PARENT_SELECTION_TYPES = ["sss", "rws", "sus", "random", "tournament"]
        if not parent_selection_type in PARENT_SELECTION_TYPES:
            raise ValueError(
                f"Invalid value: for parent_selection_type has been passed '{parent_selection_type}'. "
                f"Allowed options are: {', '.join(sorted(PARENT_SELECTION_TYPES))}"
            )
        # In case parent selection type is tournament K_tournament must be set
        if parent_selection_type == "tournament":
            if not isinstance(K_tournament, int) or isinstance(K_tournament, bool):
                raise TypeError("K_tournament is expected as an integer (bool is not allowed) in case of parent selection type is tournament.")
        # Check if crossover_type has an allowed value
        CROSSOVER_TYPES = ["single_point", "two_points", "uniform", "scattered"]
        if not crossover_type in CROSSOVER_TYPES:
            raise ValueError(
                f"Invalid value: for crossover_type has been passed '{crossover_type}'. "
                f"Allowed options are: {', '.join(sorted(CROSSOVER_TYPES))}"
            )
        # Check if mutation_type has an allowed value
        MUTATION_TYPES = ["random", "swap", "inversion", "scramble", "adaptive"]
        if not mutation_type in MUTATION_TYPES:
            raise ValueError(
                f"Invalid value: for mutation_type has been passed '{mutation_type}'. "
                f"Allowed options are: {', '.join(sorted(MUTATION_TYPES))}"
            )
        # Check if there is a previous work to resume
        if resume and os.path.exists(self.ga_saver_filepath + '.pkl'):
            self.resume_pygad_estimation()
        else:
            self.new_pygad_estimation(num_genes, num_generations, sol_per_pop, num_parents_mating, parent_selection_type, K_tournament, 
                                crossover_type, crossover_probability, keep_elitism, keep_parents, initial_population, 
                                mutation_type, mutation_percent_genes, random_mutation_min_val, random_mutation_max_val, sample_size)
        
# TODO: rendere possibile definire quante generazioni in più si vogliono fare          
    def resume_pygad_estimation(self, num_add_gen=150):
        print('LOG:\tResume previous work from ', self.ga_saver_filepath)
        # Load data from pickel file
        self.ga = pygad.load(self.ga_saver_filepath)
        # Increment the generations number if it has been completed
        if self.ga.num_generations == self.ga.generations_completed:
            self.ga.num_generations += num_add_gen
            print('LOG:\tGeneration are set to %d' %self.ga.num_generations)

    def new_pygad_estimation(self, num_genes, num_generations, sol_per_pop, num_parents_mating, parent_selection_type, K_tournament, crossover_type, 
                       crossover_probability, keep_elitism, keep_parents, initial_population, mutation_type, mutation_percent_genes, 
                       random_mutation_min_val, random_mutation_max_val, sample_size):
        print('LOG:\tCreate a new pyGAD istance')
        # Default parameters for pyGAD class
        gene_space = [{'low': -1, 'high': 1} for _ in range(num_genes)]
        gene_constraint = None
        init_range_low = -1
        init_range_high = 1
        mutation_by_replacement = False
        stop_criteria = None
        # Initilize pyGAD class
        self.ga = pygad.GA(
            num_generations = num_generations,
            num_parents_mating = num_parents_mating,
            fitness_func = self.pygad_fitness_func,
            on_start = self.on_start,
            #on_fitness = self.on_fitness,
            #on_parents = self.on_parents,
            #on_crossover = self.on_crossover,
            on_mutation = self.on_mutation,
            on_generation = self.on_generation,
            #on_stop = self.on_stop,
            initial_population = initial_population,
            init_range_low = init_range_low,
            init_range_high = init_range_high,
            sol_per_pop = sol_per_pop,
            num_genes = num_genes,
            gene_space = gene_space,
            gene_constraint = gene_constraint,
            parent_selection_type = parent_selection_type,
            K_tournament = K_tournament,
            keep_parents = keep_parents,
            keep_elitism = keep_elitism,
            crossover_type = crossover_type,
            crossover_probability = crossover_probability,
            mutation_type = mutation_type,
            mutation_percent_genes = mutation_percent_genes,
            random_mutation_min_val = random_mutation_min_val,
            random_mutation_max_val = random_mutation_max_val,
            mutation_by_replacement = mutation_by_replacement,
            sample_size = sample_size,
            parallel_processing= self.n_threads,
            random_seed = self.rng_seed
        )
        # Create empty array to store all solution in a generation
        self.post_star_vec_generation = np.empty((sol_per_pop, self.n_star))
    # Questo può diventare molto pericoloso come dimensioni
        self.regions_generation = np.empty((sol_per_pop, self.n_star, self.region_width, self.region_width))


    def start_estimation(self, plot_rate=None):
        # Start the clock
        self.start_time = time.time()
        # Initialize the counter
        self.ncycle = 0
        # Save plot rate
        if not plot_rate is None:
            self.plot_rate = plot_rate
        # Create UI
        if self.doAnimation:
            self.graphics.init_animation(self.parameters_vec, self.post_vec, self.post_star_vec, self.data_regions, self.radii, sim_param=self.sim_param)
        # Set multiproces evaluation merit
        if self.n_threads > 1 or self.use_parallel_PSF_matching:
            self.merit.setup_pool()
        # Call the correct function
        if self.type == 'dual_annealing':
            self.start_estimation_dual()
        elif self.type == 'minimize':
            self.start_estimation_minimize()
        elif self.type == 'differential_evolution':
            # Set the gpu queue:
            self.merit.set_multi_threads(self.n_threads, self.process_size_GB, self.gpu_list)
            self.start_estimation_differential()
        elif self.type == 'pyGAD':
            print('PYGAD START')
            self.start_estimation_pygad()
        # Get the end time
        self.end_time = time.time()
        # ==== PLOT THE RESULTS ==== #
        # Unpack the array
        est_parm_dic = fcu._unpack_param(self.xv_final, self.Cn2Heights, self.selected_NCPA_mode, self.do_jitter)
        # Make the results plots
        output_res_image_dirname = 'Estimation_results'
        merit, merit_stars, _ = self.merit.make_estimation(est_parm_dic, show_estimators=False, save_estimators=False, show_regions=False, save_regions=False,outdir=os.path.join(self.outdir, output_res_image_dirname), fullResults=True)
        print("---- Estimation results with minimize ----")
        print("Initial parameters")
        print(self.x0)
        print("Results parameters after %d cycle" %self.ncycle)
        log_msg = 'seeing=%.4f \t' %est_parm_dic['seeing']
        log_msg += 'weight='+np.array2string(est_parm_dic['cn2wei'], separator=", ", precision=4).replace('\n', '')+"  "
        log_msg += 'velo='  +np.array2string(est_parm_dic['winvel'], separator=", ", precision=4).replace('\n', '')  +"  "
        log_msg += 'dir='   +np.array2string(est_parm_dic['windir'], separator=", ", precision=4).replace('\n', '')   +"\t"
        log_msg += 'merit=%.5f' %merit
        if 'zCoefStaticOn' in est_parm_dic:
            log_msg += '\nnpca='+np.array2string(est_parm_dic['zCoefStaticOn'], separator=", ", precision=2).replace('\n', '')   +"\t"
        if 'jitter' in est_parm_dic:
            log_msg += '\njitter = ' + np.array2string(est_parm_dic['jitter'], separator=", ", precision=2).replace('\n', '')   +"\t"
        log_msg +='\n'
        print(log_msg)
        print("Merits for each stars is")
        print(merit_stars)
        # Save plots
        if self.doAnimation:
            self.graphics.fig_param.savefig(os.path.join(self.outdir, 'Merit_vs_params.png'), dpi=400)
            self.graphics.fig_startrend.savefig(os.path.join(self.outdir, 'Merit_vs_stars.png'), dpi=400)
            self.graphics.fig_regions.savefig(os.path.join(self.outdir, 'Regions.png'), dpi=400)
        # Return values
        return est_parm_dic, merit, merit_stars
    
    def start_estimation_dual(self):
        # Define the minimizer arguments
        minimizer_kwargs ={
            'method': self.method,
            'bounds': self.bounds,
            'tol': self.tol,
        }
        # Start the estimator
        res = opt.dual_annealing(
            func = self.main_estimator,
            bounds = self.bounds, 
            maxiter = int(self.maxiter), 
            minimizer_kwargs = minimizer_kwargs, 
            initial_temp = self.initial_temp, 
            restart_temp_ratio = self.temp_ratio, 
            visit = self.visit, 
            accept = self.accept, 
            maxfun = self.maxfun, 
            seed = self.rng_seed, 
            no_local_search = self.no_local_search, 
            x0 = self.x0
            )
        # Save the results
        self.xv_final = res.x
        self.dual_flag = res.success
    
    def start_estimation_minimize(self):
        # Convert the bounderies from an array to a tuple list
        bounds = [tuple(row) for row in self.bounds]
        # Create the dictonary
        if self.method == 'Nelder-Mead':
            options = {
                'maxiter': self.maxiter,
                'maxfev' : self.maxiter,
                'fatol' : self.tol,
                'disp' : True
            }
        elif self.method == 'L-BFGS-B':
            options = {
                'maxfun' : self.maxiter,
                'maxiter': self.maxiter,
                'ftol' : self.tol,
                'disp' : True
            }
        elif self.method == 'Powell':
            options = {
                'maxiter': self.maxiter,
                'maxfev' : self.maxiter,
                'ftol' : self.tol, 
                'disp' : True
            }
        elif self.method == 'TNC':
            options = {
                'ftol' : self.tol,
                'maxfun': self.maxiter,
                'disp' : True
            }
        else:
            raise ValueError('Minimize method %s is not bounds-contraints compatible.'%self.method)
        # Define the minimizer
        res = opt.minimize(
            fun = self.main_estimator, 
            x0 = self.x0, 
            method = self.method, 
            tol = self.tol, 
            bounds = bounds, 
            options = options
            )
        # Save the results
        self.xv_final = res.x
        self.minimize_flag = res.success        
    
    def start_estimation_differential(self):
        res = opt.differential_evolution(self.main_estimator, 
                                         bounds = self.bounds, 
                                         strategy = self.strategy, 
                                         maxiter = self.maxiter, 
                                         popsize = self.popsize, 
                                         tol = self.tol, 
                                         mutation = self.mutation, 
                                         recombination = self.cross_prob, 
                                         rng = self.rng_seed, 
                                         callback = None, 
                                         disp = False, 
                                         polish = False, 
                                         init = self.init_mode, 
                                         atol = 0, 
                                         updating = self.update_mode, 
                                         workers = self.n_threads, 
                                         constraints = (), 
                                         x0 = self.x0,
                                         integrality = None, 
                                         vectorized = False)
        # Save the results
        self.xv_final = res.x
        self.minimize_flag = res.success

    def start_estimation_pygad(self):
        # Set the gpu queue:
        self.merit.set_multi_threads(self.n_threads, self.process_size_GB, self.gpu_list)
        # Run the simulation
        self.ga.run()
        # Save the plots
        if self.doAnimation:
            self.ga.plot_fitness()
            self.graphics.fig_param.savefig(os.path.join(self.outdir, 'Merit_vs_params.png'), dpi=400)
            self.graphics.fig_startrend.savefig(os.path.join(self.outdir, 'Merit_vs_stars.png'), dpi=400)
            self.graphics.fig_regions.savefig(os.path.join(self.outdir, 'Regions.png'), dpi=400)
        # Get the best solution
        fitness_pop = self.ga.last_generation_fitness
        solution_idx = fitness_pop.argmax()
        xv_wrap = self.ga.population[solution_idx]
        fitness = fitness_pop[solution_idx]
        # Unwrap
        xv_final = fcu._unwrap_bound(xv_wrap, self.bounds)
        # Unpack the array
        est_param_dic = fcu._unpack_param(xv_final, self.Cn2Heights, self.selected_NCPA_mode, self.do_jitter)
        # Make the results plots
        output_res_image_dirname = 'Estimation_results'
        merit = self.merit.make_estimation(est_param_dic, show_estimators=False, save_estimators=True, show_regions=False, save_regions=True, outdir=os.path.join(self.outdir, output_res_image_dirname))
        # Return values
        return est_param_dic, fitness, merit
    
    def main_estimator(self, xv):
        # Update cycle counts
        self.ncycle += 1
        # Get the thread id
        thread_id = threading.get_ident()
        # Unpack
        est_parm_dic = fcu._unpack_param(xv, self.Cn2Heights, self.selected_NCPA_mode, self.do_jitter)
        # Check if the parameters are good
        if np.sum(est_parm_dic['cn2wei']) == 1:
            merit_value, posteriors_regions, map_regions = self.merit.make_estimation(est_parm_dic, self.radii, thread_id, fullResults=True)
        else:
            print('Value outside the boundieries')
        # Append values to the arrays
        if hasattr(self, 'post_vec'):
            self.post_vec = np.append(self.post_vec, [merit_value], 0)
        if hasattr(self, 'parameters_vec'):
            self.parameters_vec = np.append(self.parameters_vec, [xv], 0)
        if hasattr(self, 'post_star_vec'):
            self.post_star_vec = np.append(self.post_star_vec, [posteriors_regions], 0)
        # Print Log
        log_msg = '===> CYCLE: %d \t' %self.ncycle
        log_msg += 'seeing=%.4f \t' %est_parm_dic['seeing']
        log_msg += 'weight='+np.array2string(est_parm_dic['cn2wei'], separator=", ", precision=4).replace('\n', '')+"  "
        log_msg += 'velo='  +np.array2string(est_parm_dic['winvel'], separator=", ", precision=4).replace('\n', '')  +"  "
        log_msg += 'dir='   +np.array2string(est_parm_dic['windir'], separator=", ", precision=4).replace('\n', '')   +"\t"
        log_msg += 'post=%.5f' %merit_value
        if 'zCoefStaticOn' in est_parm_dic:
            log_msg += '\nnpca='+np.array2string(est_parm_dic['zCoefStaticOn'], separator=", ", precision=2).replace('\n', '')   +"\t"
        if 'jitter' in est_parm_dic:
            log_msg += '\njitter = ' + np.array2string(est_parm_dic['jitter'], separator=", ", precision=2).replace('\n', '')   +"\t"
        log_msg +='\n'
        print(log_msg)
        # Add a line to the results file
        if self.save_cycle_status:
            res_str = '%d\t\t' %self.ncycle
            res_str += np.array2string(xv, separator="\t", floatmode='fixed', formatter={'float_kind':lambda x: "%3.8f" % x})[1:-1].replace('\n', '')
            res_str += '\t%.5e\n' %merit_value
            with open(self.cycle_status_filepath, "a") as f:
                f.write(res_str)
        # Add a line to the merits stars file
        if self.save_merit_stars:
            mer_str =  '%d\t\t' %self.ncycle
            mer_str += '%.8e\t' %merit_value
            mer_str += np.array2string(self.post_star_vec[-1], separator="\t", floatmode='fixed', formatter={'float_kind':lambda x: "%.8e" % x})[1:-1].replace('\n', '')
            mer_str += '\n'
            with open(self.merit_stars_cycle_filepath, "a") as f:
                f.write(mer_str)
        # Update live plots
        if self.doAnimation:
            if self.ncycle % self.plot_rate == 0:
                if self.graphics.plotParameter:
                    self.graphics.update_parameters_post_trend(self.parameters_vec, self.post_vec)
                if self.graphics.plotStarTrend:
                    self.graphics.update_post_trend_for_stars(self.post_star_vec)
                if self.graphics.plotRegions:
                    self.graphics.update_regions(map_regions, self.radii)
        # Return values
        return merit_value

    ### ======================================================
    ###            PY-GAD  FUNCTIONS
    ### ======================================================

    def on_start(self, ga_instanse):
        print('LOG:\tStarting pyGAD estimation with the following setup')
        print('Genes number: ', self.ga.num_genes)
        print('Generation number: ', self.ga.num_generations)
        print('Solutions per population: ', self.ga.sol_per_pop)
        print('Number of parents mating: ', self.ga.num_parents_mating)
        print('Parent selection type: ', self.ga.parent_selection_type)
        print('K_tournament: ', self.ga.K_tournament)
        print('Number of parents to keep: ', self.ga.keep_parents)
        print('Number of elitism to keep: ', self.ga.keep_elitism)
        print('Crossover type: ', self.ga.crossover_type)
        print('Crossover probablity: ', self.ga.crossover_probability)
        print('Mutation type: ', self.ga.mutation_type)
        print('Percent of mutation genes: ', self.ga.mutation_percent_genes)
        print('Random min and max val: [', self.ga.random_mutation_min_val, ', ', self.ga.random_mutation_max_val, ']')
        print('Number of threads used: ', self.ga.parallel_processing[1])
        print('Random seed uesd:', self.ga.random_seed)
        
        # Create UI
        if self.doAnimation:
            print('LOG:\tCreating animation')
            # Create the live plot figure
            self.graphics.init_animation(self.parameters_vec, self.post_vec, self.post_star_vec, self.data_regions, self.radii, sim_param=self.sim_param)
        else:
            print('LOG:\tNo animation created')
        
        # Set the thread locker
        self.lock = threading.Lock()
        # Start the progress bar for the generation
        self.progress_bar_generation = tqdm(total=self.ga.num_generations,position=0, unit="generations", desc="Computing the generation", colour='CYAN', ascii=True, dynamic_ncols=True)
        # Start the progress bar for the popolation
        self.progress_bar_population = tqdm(total=self.ga.sol_per_pop, position=1, unit="solutions", desc="Compute fintess for population 0", colour='MAGENTA', ascii=True, leave=True, dynamic_ncols=True)
        #self.progress_bar_population.set_lock(self.lock)
        
    def on_fitness(self, ga_instanse, last_gen_fitness):
        tqdm.write("on_fitness")

    def on_parents(self, ga_instanse, last_gen_parents):
        tqdm.write("on_parents")

    def on_crossover(self, ga_instanse, last_gen_offspring):
        tqdm.write("on_crossover")
        
    def on_mutation(self, ga_instanse, last_gen_offspring):
        #tqdm.write("on_mutation")
        self.progress_bar_generation.update(1)
        self.progress_bar_population.reset()
        self.progress_bar_population.set_description("Compute fintess for population %d"%(self.ga.generations_completed+1))
            
    def on_generation(self, ga_instanse):
        try:         
            # Remove the saved value to avoid error
            if self.doAnimation:
                if hasattr(self, 'graphics'):
                    graphics_temp = self.graphics
                    try:
                        del self.graphics
                    finally:
                        self.graphics = None
            gpu_queue = self.merit.gpu_queue
            self.merit.gpu_queue = None
            # Save GA_istance
            #ga_instanse.save(self.ga_saver_filepath)
            # Extract information
            xv_pop = fcu._unwrap_bound(self.ga.population, self.bounds)
            merits_pop = ga_instanse.last_generation_fitness
            self.xv_pop = np.column_stack((xv_pop, merits_pop))
            tqdm.write('xv_unwrap and fitness of generation %d' %self.ga.generations_completed)
            tqdm.write('     seeing  Cn2_ground   winvel1   windir1     winvel2   windir2     ast_amp   ast_phi    coma_amp   coma_phi    tri-amp    tri-phi   fitness')
            for row in self.xv_pop:
                tqdm.write(np.array2string(row, formatter={"float_kind": lambda x: f"{x:10.4f}"}, max_line_width=1000))
            # Save Population to file
            with open(self.pop_save_filepath, 'wb') as file:
                np.save(file, self.xv_pop, allow_pickle=False)
            # Find the best solution
            solution_idx = merits_pop.argmax()
            xv_wrap = ga_instanse.population[solution_idx]
            merit = merits_pop[solution_idx]
            #merit = - merit 
            xv = fcu._unwrap_bound(xv_wrap, self.bounds)
            # Add to the array the correct values
            if hasattr(self, 'parameters_vec'):
                self.parameters_vec = np.append(self.parameters_vec, [xv], 0)
            if hasattr(self, 'post_vec'):
                self.post_vec = np.append(self.post_vec, [merit], 0)
            if hasattr(self, 'post_star_vec'):
                self.post_star_vec = np.append(self.post_star_vec, [self.post_star_vec_generation[solution_idx]], 0)
            # Cast elements
            est_parm_dic = fcu._unpack_param(xv, self.Cn2Heights, self.selected_NCPA_mode, self.do_jitter)
            # Print Log
            if True:
                log_msg = '===> GENERATION: %d \t' %self.ga.generations_completed
                log_msg += 'seeing=%.4f \t' %est_parm_dic['seeing']
                log_msg += 'weight='+np.array2string(est_parm_dic['cn2wei'], separator=", ", precision=4).replace('\n', '')+"  "
                log_msg += 'velo='  +np.array2string(est_parm_dic['winvel'], separator=", ", precision=4).replace('\n', '')  +"  "
                log_msg += 'dir='   +np.array2string(est_parm_dic['windir'], separator=", ", precision=4).replace('\n', '')   +"\t"
                log_msg += 'post=%.5f' %merit
                if 'zCoefStaticOn' in est_parm_dic:
                    log_msg += '\nnpca='   +np.array2string(est_parm_dic['zCoefStaticOn'], separator=", ", precision=2).replace('\n', '')   +"\t"
                log_msg +='\n'
                tqdm.write(log_msg)
            # Add a line to the results file
            if self.save_cycle_status:
                res_str = '%d\t\t' %self.ga.generations_completed
                res_str += np.array2string(xv, separator="\t", floatmode='fixed', formatter={'float_kind':lambda x: "%3.8f" % x})[1:-1].replace('\n', '')
                res_str += '\t%.5e\n' %merit
                with open(self.cycle_status_filepath, "a") as f:
                    f.write(res_str)
            # Add a line to the merits stars file
            if self.save_merit_stars:
                mer_str =  '%d\t\t' %self.ga.generations_completed
                mer_str += '%.8e\t' %merit
                mer_str += np.array2string(self.post_star_vec[-1], separator="\t", floatmode='fixed', formatter={'float_kind':lambda x: "%.8e" % x})[1:-1].replace('\n', '')
                mer_str += '\n'
                with open(self.merit_stars_cycle_filepath, "a") as f:
                    f.write(mer_str)
            # Update live plots
            if self.doAnimation:
                if graphics_temp.plotParameter:
                    graphics_temp.update_parameters_post_trend(self.parameters_vec, self.post_vec)
                if graphics_temp.plotStarTrend:
                    graphics_temp.update_post_trend_for_stars(self.post_star_vec)
                if graphics_temp.plotRegions:
                    graphics_temp.update_regions(self.regions_generation[solution_idx], self.radii)
        # Reset the graphics handle after all the saving and plotting occuring
        finally:
            if self.doAnimation:
                self.graphics = graphics_temp
            self.merit.gpu_queue = gpu_queue
        # Reduce the random min and max
        if self.ga.generations_completed % self.generation_rate == 0:
            # Reduce the space for the random mutation after a fixed number of generation
            reducer = 1.25
            self.ga.random_mutation_min_val /= reducer
            self.ga.random_mutation_max_val /= reducer
            tqdm.write('Random mutation space is now set to: [%.1e, %.1e]'% (self.ga.random_mutation_min_val, self.ga.random_mutation_max_val) )

    def on_stop(self, ga_instanse, last_gen_fitness):
        tqdm.write("on_stop")

    ### ===========================================================
    ###                   MERTITS FUNCTIONS
    ### ===========================================================

    def pygad_fitness_func(self, ga_instanse, xv_wrap, solution_idx):
        # Get the thread id
        thread_id = threading.get_ident()
        # Cast elements
        xv = fcu._unwrap_bound(xv_wrap, self.bounds)
        # Look in the memory if there is a match
        res_look_xv_pop = self._find_fitness_solution(xv, self.xv_pop)
        if res_look_xv_pop:
            fitness = res_look_xv_pop
        else:
            est_parm_dic = fcu._unpack_param(xv, self.Cn2Heights, self.selected_NCPA_mode, self.do_jitter)
            # Check if the parameters are good
            if np.sum(est_parm_dic['cn2wei']) == 1:
                merit_value, posteriors_regions, map_regions = self.merit.make_estimation(est_parm_dic, self.radii, thread_id, fullResults=True)
                fitness = - merit_value**3
                # Save merit function for each stars
                if hasattr(self, 'post_star_vec_generation'):
                    self.post_star_vec_generation[solution_idx] = posteriors_regions
                # Save merit map for each star
                if hasattr(self, 'regions_generation'):
                    self.regions_generation[solution_idx] = map_regions
            else:
                fitness = self.MIN_FITNESS
        # Update the progress bar
        if self.progress_bar_population is not None:
            with self.lock:
                self.progress_bar_population.update(1)
        return fitness

    ### ================================================
    ###            AUXILIARY FUNCTIONS 
    ### ================================================

    def _on_function_print(self, ga_instanse):
        # Extract information
        xv_pop = fcu._unwrap_bound(self.ga.population, self.bounds)
        merits_pop = ga_instanse.last_generation_fitness
        xv_pop = np.column_stack((xv_pop, merits_pop))
        tqdm.write('Population of generation %d' %self.ga.generations_completed)
        for row in xv_pop:
            tqdm.write(np.array2string(row, formatter={"float_kind": lambda x: f"{x:10.4f}"}, max_line_width=100))
           
    def _find_fitness_solution(self, xv, matrix, rtol=1e-8, atol=1e-12):
        # Check if there is a record
        if matrix is None:
            return False
        # Set the correct data_type for xv
        xv = np.array(xv)
        # Check if xv has the correct lenght:
        if xv.shape[0] != self.ga.num_genes:
            raise ValueError('Invalid shape: xv has shape ', xv.shape ,' while must be (%d)' %self.ga.num_genes)
        # Boolean mask for approximate match
        mask = np.all(np.isclose(matrix[:, :self.ga.num_genes], xv, rtol=rtol, atol=atol), axis=1)
        # If there are no match in the matrix return false
        if not np.any(mask):
            return False
        # List all the candidate for the close-match
        candidate_idx = np.where(mask)[0]
        # If find just 1 candidate
        if len(candidate_idx) == 1:
            return matrix[candidate_idx[0], -1]
        # If there are multipole candidate find the best
        else:
            # Compute absolute differences
            diffs = np.abs(matrix[:, :self.ga.num_genes] - xv)
            sums = np.sum(diffs[candidate_idx, :], axis=1)
            best_idx = candidate_idx[np.argmin(sums)]
            return matrix[best_idx, -1]
