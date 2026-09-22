import os
import files as ff
import numpy as np

class Saver:
    def __init__(self, outdir, generic_ini_filepath, data_filepath=None, headerReader=None, findStar=None, 
                dataRegions=None,  estimator=None, graphics=None, simulation=None, filetag='save_status', 
                verbose=True, selected_stars=None, avoid_stars=None, fpath_ao_pos=None, fpath_all_pos=None):
        # Store filepaths and directories information
        self.outdir = outdir
        self.filetag = filetag
        self.out_filepath = os.path.join(outdir, filetag+'.ini')
        self.data_filepath = data_filepath
        self.generic_ini_filepath = generic_ini_filepath

        # Save class handler
        self.headerReader = headerReader
        self.findStar = findStar
        self.dataRegions = dataRegions
        self.estimator = estimator
        self.graphic = graphics
        self.simulation = simulation

        # Defing saving procedures
        self.saving_precision = 8

        # Save verbosing
        self.verbose = verbose

        # Save chosen star
        self.selected_stars = selected_stars
        self.avoid_stars    = avoid_stars
        self.fpath_ao_pos   = fpath_ao_pos
        self.fpath_all_pos  = fpath_all_pos

    def save(self):
        self.dictionary = [
            ("Project", "data-fpath", self.data_filepath),
            ("Project", "outdir", self.outdir),
            ("Project", "gen-ini-fpath", self.generic_ini_filepath),
            ("Project", "select-stars", self.selected_stars)
        ]
        if self.avoid_stars is not None:
            self.dictionary.extend([("Project", "avoid-stars", self.avoid_stars)])

        if self.fpath_ao_pos is not None:
            self.dictionary.extend([("Project", "fpath-ao-pos", self.fpath_ao_pos)])

        if self.fpath_all_pos is not None:
            self.dictionary.extend([("Project", "fpath-all-pos", self.fpath_all_pos)])

        self.dictionary.extend([
            ("Saver", "fname", self.filetag),
            ("Saver", "float-precision", self.saving_precision)
        ])

        if not self.headerReader is None:
            self.add_header()

        if not self.findStar is None:
            self.add_finder()

        if not self.estimator is None:
            self.add_estimator()

        if not self.dataRegions is None:
            self.add_regions()
        
        #if not self.merit is None:
        #    self.add_merit()

        ff.write_ini(self.out_filepath, self.dictionary)
        if self.verbose:
            print("LOG:\tFile saved")


    def add_header(self):
        self.dictionary.extend([
            ("Header", "teleResolution", self.headerReader.tel_res),
            ("Header", "pixelScale", self.headerReader.pixscale),
            ("Header", "A0-freq", self.headerReader.AO_freq),
            ("Header", "n-photo", self.headerReader.n_photo),
            ("Header", "telescopeAlt", self.headerReader.telescope_alt),
        ])

    def add_finder(self):
        self.dictionary.extend([
            ("Finder", "fwhm_AO", self.findStar.fwhm_AO),
            ("Finder", "thr_AO", self.findStar.thr_AO),
            ("Finder", "min_sep_AO", self.findStar.min_sep_AO),
            ("Finder", "fwhm_field", self.findStar.fwhm_field),
            ("Finder", "thr_field", self.findStar.thr_field),
            ("Finder", "min_sep_field", self.findStar.min_sep_field),
            ("Finder", "nField", self.findStar.nField),
            ("Finder", "findAO", self.findStar.findAO),
            ("Finder", "findFIELD", self.findStar.findFIELD),
            ("Finder", "doPlot", self.findStar.doPlot),
            ("Finder", "positionAO", self.findStar.pos_AO),
            ("Finder", "positionField", self.findStar.pos_field)
        ])

    def add_regions(self):
        self.dictionary.extend([
            ("Regions", "box_edge", self.dataRegions.box_edge),
            ("Regions", "star-pos", self.dataRegions.star_pos),
            ("Regions", "radii", self.dataRegions.radii),
            ("Regions", "regions-fpath", self.dataRegions.fpath)
        ])

    def add_estimator(self):
        self.dictionary.extend([
            ("Estimator", "estimator-type", self.estimator.type),
            ("Estimator", "psf-fit-type", self.estimator.psf_fit_type),
            ("Estimator", "merit-shape", self.estimator.merit_shape),
            ("Estimator", "cn2Heights", self._array2str(self.estimator.Cn2Heights) ),
            ("Estimator", "radii", self._array2str(self.estimator.radii) ),
            ("Estimator", "data-ini-fpath", self.estimator.data_ini_filepath ),("Estimator", "doAnimation", self.estimator.doAnimation),
            ("Estimator", "save-cycle-status", self.estimator.save_cycle_status),
            ("Estimator", "cycle-status-ftag", self.estimator.parameters_mf_filetag),
            ("Estimator", "save-stars-mf", self.estimator.save_merit_stars),
            ("Estimator", "stars-mf-ftag", self.estimator.stars_mf_filetag),
            ("Estimator", "resume", self.estimator.resume),
            ("Estimator", "max-ncpa-vec", self.estimator.max_abberation_vec),
            ("Estimator", "max-jitter-vec", self.estimator.max_jitter_vec),
            ("Estimator", "rng_seed", self.estimator.rng_seed),
            ("Estimator", "bounds", self._array2str(self.estimator.bounds)),
            ("Estimator", "run-time", self.estimator.end_time-self.estimator.start_time)
        ])

        if self.estimator.type == 'pyGAD':
            self.dictionary.extend([
                ("Estimator", "MIN-FITNESS", self.estimator.MIN_FITNESS),
                ("Estimator", "pop-ftag", self.estimator.population_filetag),
                ("Estimator", "num-generations", self.estimator.ga.num_generations),
                ("Estimator", "sol-per-pop", self.estimator.ga.sol_per_pop),
                ("Estimator", "num-pare-mate", self.estimator.ga.num_parents_mating),
                ("Estimator", "pare-sel-type", self.estimator.ga.parent_selection_type),
                ("Estimator", "k-tournament", self.estimator.ga.K_tournament),
                ("Estimator", "cross-type", self.estimator.ga.crossover_type),
                ("Estimator", "cross-prop", self.estimator.ga.crossover_probability),
                ("Estimator", "keep-elit", self.estimator.ga.keep_elitism),
                ("Estimator", "keep-pare", self.estimator.ga.keep_parents),
                ("Estimator", "muta-type", self.estimator.ga.mutation_type),
                ("Estimator", "muta-perc-gene", self.estimator.ga.mutation_percent_genes),
                ("Estimator", "muta-rand-min", self.estimator.ga.random_mutation_min_val),
                ("Estimator", "muta-rand-max", self.estimator.ga.random_mutation_max_val),
                ("Estimator", "n-threads", self.estimator.ga.parallel_processing[1]),
                ("Estimator", "sample-size", self.estimator.ga.sample_size),
            ])
        elif self.estimator.type == 'dual_annealing':
            self.dictionary.extend([
                ("Estimator", "n-threads", self.estimator.n_threads),
                ("Estimator", "initial-temp", self.estimator.initial_temp),
                ("Estimator", "temp-ratio", self.estimator.temp_ratio),
                ("Estimator", "visit", self.estimator.visit),
                ("Estimator", "accept", self.estimator.accept),
                ("Estimator", "maxiter", self.estimator.maxiter),
                ("Estimator", "maxfun", self.estimator.maxfun),
                ("Estimator", "no.local-search", self.estimator.no_local_search),
                ("Estimator", "x0", self.estimator.x0),
                ("Estimator", "resume", self.estimator.resume),
                ("Estimator", "tol", self.estimator.tol),
                ("Estimator", "method", self.estimator.method),
            ])
        elif self.estimator.type == 'differential_evolution':
            self.dictionary.extend([
                ("Estimator", "n-threads", self.estimator.n_threads),
                ("Estimator", "maxiter", self.estimator.maxiter),
                ("Estimator", "popsize", self.estimator.popsize),
                ("Estimator", "x0", self.estimator.x0),
                ("Estimator", "tol", self.estimator.tol),
                ("Estimator", "mutation", self.estimator.mutation),
                ("Estimator", "cross-prob", self.estimator.cross_prob),
                ("Estimator", "strategy", self.estimator.strategy),
                ("Estimator", "init-mode", self.estimator.init_mode),
                ("Estimator", "update-mode", self.estimator.update_mode),
            ])
        elif self.estimator.type == 'minimize':
            self.dictionary.extend([
                ("Estimator", "maxiter", self.estimator.maxiter),
                ("Estimator", "x0", self.estimator.x0),
                ("Estimator", "tol", self.estimator.tol),
                ("Estimator", "method", self.estimator.method),
            ])
                

    def _array2str(self, array):
        string = np.array2string(array, separator=", ", precision=self.saving_precision).replace('\n', '')
        return array #### ATTENZIONE QUESTO DI PROVA, NON FA NULLA
