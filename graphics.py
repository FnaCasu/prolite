import os # to operate with path strings
import numpy as np
# matplotlib importing
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # to set colorbar under imshow
# Imort tkinter to create live UI
import tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import pixelUtiles as pxu
import format_and_control as fcu

# Use TeX in the plots
#plt.rcParams['figure.dpi'] = 100
#plt.rcParams['savefig.dpi'] = 600
plt.rcParams['text.usetex'] = True

class Graphics:
    def __init__(self, outdir=None, sim_param=None, showflag=False):
        self.outdir = outdir
        self.sim_param = sim_param
        self.plotParameter = False
        self.plotStarTrend = False
        self.plotRegions   = False
        self.cmap = mpl.colormaps['viridis']
        self.cmap_res = mpl.colormaps['RdBu']
        # Default showflag used by the AO post-processing plots below
        # (plot_star_regions, plot_fwhm, ...); individual calls can still
        # override it via their own showflag= argument where offered.
        self.showflag = showflag

    def _calc_cbr_lowtop(self, regions, radii=None):
        # If there is a radius use the values inside it
        if not radii is None:
            # Just one region
            if regions.ndim == 2:
                # Find the maximum inside the circle as the upper scale of the colorbar
                region = regions.copy()
                radius = np.array(radii)
                if radius.ndim>0:
                    radius=radius[0]
                mask = pxu.inverse_circular_mask(region, [region.shape[0]/2,region.shape[0]/2], radius)
                region[mask] = np.nan
                top_cbr = np.nanmax(region)
                low_cbr = np.nanmin(region)
            # Multipole regions
            elif regions.ndim == 3:
                n_regions = regions.shape[0]
                top_cbr_vec = np.empty(n_regions)
                low_cbr_vec = np.empty(n_regions)
                radii = fcu._format_radii(radii, n_regions)
                # Loop over regions
                for ii, region in enumerate(regions):
                    # Find the maximum inside the circle as the upper scale of the colorbar
                    region_copy = region.copy()
                    mask = pxu.inverse_circular_mask(region_copy, [region_copy.shape[0]/2,region_copy.shape[0]/2], radii[ii])
                    region_copy[mask] = np.nan
                    top_cbr_vec[ii] = np.nanmax(region_copy)
                    low_cbr_vec[ii] = np.nanmin(region_copy)
                # Find overall max and min
                low_cbr = np.nanmin(low_cbr_vec)
                top_cbr = np.nanmax(top_cbr_vec)
        # If there is no radi use the all region
        else:
            top_cbr= np.nanmax(regions)
            low_cbr = np.nanmin(regions)
        # Return the values
        return low_cbr, top_cbr

    def plot_one_region(self, ax, region, radius=None, idx=None, n_col=0, cut_region=True, is_residuals=False, cbar_minmax=None):
        # Maximum for the colorbar
        if cbar_minmax is None:
           low_cbr, top_cbr = self._calc_cbr_lowtop(region, radius)
        else:
            low_cbr = cbar_minmax[0]
            top_cbr = cbar_minmax[1]
        # Residuals color map
        if is_residuals:
            top_cbr = max(abs(low_cbr), abs(top_cbr))
            low_cbr = -top_cbr
            cmap = self.cmap_res
        else:
            cmap = self.cmap
        # Plot data into the axis
        im = ax.imshow(region, cmap=cmap, vmin=low_cbr, vmax=top_cbr)
        # Add red circle
        if not radius is None:
            circle = plt.Circle((region.shape[1]/2, region.shape[0]/2), radius, color='r', fill=False, lw=2)
            ax.add_patch(circle)
            # Cut the regions near the circle
            if cut_region:
                ax.set_xlim([region.shape[0]/2-1.5*radius, region.shape[0]/2+1.5*radius])
                ax.set_ylim([region.shape[1]/2-1.5*radius, region.shape[1]/2+1.5*radius])
        # Find the axes divider
        divider = make_axes_locatable(ax)
        # Define the position and the orientation of the colorbar
        cax = divider.append_axes('bottom', size='5%', pad=0.7)
        # Create the colorbar
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        # Set the colorbar label and tick 
        if False:
            # In case it shows residuals we do not want the exponential notation (0, 100)
            cbar.formatter.set_useMathText(False)
            # The map is not in counts but in percents
            cbar.set_label(r'\% peak data')
        else:
            # In case it shows a data regions we like the exponential notation
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)
            # The map is in ADU counts
            cbar.set_label(r'counts [ADU]')
        # Set title
        if not idx is None:
            ax.set_title(r'\textbf{Star region nr. %d}'%idx)
        # Add x-label by default
        ax.set_xlabel(r'x relative position [px]')
        # If is on left add y-label
        if n_col==0:
            ax.set_ylabel(r'y relative position [px]')

    def plot_estimator(self, data, sim, res, star_idx=None, saveflag=False, radius=None, outdir=None, filestag=None, showfigure=True, starname=None, figdpi=400):
        # Create figure
        fig, ax = plt.subplots(1,3, figsize=(10.5,5.25))
        # Set geometry
        fig.tight_layout(rect=(0.03, 0, 1, 0.95), pad=0.5)
        # Set title
        if not starname is None:
            fig.suptitle(r'\textbf{Residuals for star %s}' %starname)
        elif not star_idx is None:
            fig.suptitle(r'\textbf{Residuals for star nr. %d}'%star_idx)
        
        # Find minimum and maximum to make the same color bar for data and sim
        cbar_min, cbar_max = self._calc_cbr_lowtop(np.array([data, sim]), radius)

        # Plot data
        self.plot_one_region(ax[0], data, radius, star_idx, n_col=0, cut_region=True, is_residuals=False, cbar_minmax=[cbar_min, cbar_max])
        ax[0].set_title(r'\textbf{Observed star}')
        # Plot simulation
        self.plot_one_region(ax[1], sim, radius, star_idx, n_col=1, cut_region=True, is_residuals=False, cbar_minmax=[cbar_min, cbar_max])
        ax[1].set_title(r'\textbf{Reconstructed star}')
        # Plot residuals
        self.plot_one_region(ax[2], res, radius, star_idx, n_col=2, cut_region=True, is_residuals=True)
        ax[2].set_title(r'\textbf{Residuals}')

        # Save figure
        if saveflag:
            # Define the output directory
            if outdir is None:
                if self.outdir is None:
                    outdir = os.path.join(os.getcwd(), 'Estimation_Plots')
                else:
                    outdir = os.path.join(self.outdir, 'Estimation_Plots')
            # If not exist creates the directory
            if not os.path.isdir(outdir):
                try:
                    os.makedirs(outdir)
                except FileExistsError:
                    print(f"Directory '{outdir}' already exists.")
                except PermissionError:
                    print(f"Permission denied: Unable to create '{outdir}'.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            # Create file name
            if filestag is None:
                filestag = 'Estimation_results_on_star'
            if not starname is None:
                filestag += '_'+starname
            else:
                filestag += '_'+str(star_idx)
            # Save figure
            fig.savefig(os.path.join(outdir, filestag+'.png'), dpi=figdpi)
        # Close the figure to not be shown after
        if showfigure:
            return fig
        else:
            plt.close(fig)
                

    def plot_posteriors_from_file(self, inputfile, sim_param=None, saveflag=False):
        # Check if file exist
        if not os.path.isfile(inputfile):
            raise FileNotFoundError('Data file {} not exist in data loading'.format(inputfile))
        # Read data
        pack_txt = np.loadtxt(inputfile)
        num_columns = pack_txt.shape[1]
        num_row     = pack_txt.shape[0]
        last_row = pack_txt[-1,:]
        pack_txt = pack_txt.T
        # Check data format and unpack
        if num_columns >= 6 and num_columns%3 == 0:
            n_cycle_vec    = pack_txt[0]
            parameters_vec = pack_txt[1:-1].T
            post_vec       = pack_txt[-1]
        else:
            raise ValueError('Unexpected number of columns in the input file')
        # Check if the laast row is not completed
        if post_vec[-1] == 0:
            n_cycle_vec    = n_cycle_vec[:-1]
            parameters_vec = parameters_vec[:-1]
            post_vec       = post_vec[:-1]
        # Make plot and return fig and axis
        return self.plot_parameters_post_trend(parameters_vec, post_vec, sim_param, saveflag)

    def plot_startrend_from_file(self, inputfile):
        # Check if file exist
        if not os.path.isfile(inputfile):
            raise FileNotFoundError('Data file {} not exist in data loading'.format(inputfile))
        # Read data
        pack_txt = np.loadtxt(inputfile).T
        pack_txt = pack_txt[2:].T
        return self.plot_post_trend_for_stars(pack_txt)


    def init_animation(self, parameters_vec, post_vec, post_star_vec, regions, radii=None, sim_param=None, plotParameter=True, plotStarTrend=True, plotRegions=True):
        if plotParameter:
            self.fig_param, self.axis_param = self.plot_parameters_post_trend(parameters_vec, post_vec, sim_param)
            # Create User Interface
            self.tk_param = Tk.Tk()
            self.tk_param.title("Realtime Posterior vs Parameters")
            self.canvas_param = FigureCanvasTkAgg(self.fig_param, master=self.tk_param)
            self.canvas_param.get_tk_widget().grid(column=0, row=0)
            self.tk_param.update_idletasks()
            self.plotParameter = True

        if plotStarTrend:
            self.fig_startrend, self.axis_startrend = self.plot_post_trend_for_stars(post_star_vec)
            # Create User Interface
            self.tk_startrend = Tk.Tk()
            self.tk_startrend.title("Realtime trend of posterior for stars")
            self.canvas_startrend = FigureCanvasTkAgg(self.fig_startrend, master=self.tk_startrend)
            self.canvas_startrend.get_tk_widget().grid(column=0, row=0)
            self.tk_startrend.update_idletasks()
            self.plotStarTrend = True

        if plotRegions:
            self.fig_regions, self.axis_regions = self.plot_regions(regions, radii, title='Merit function in real-time')
            self.tk_regions = Tk.Tk()
            self.tk_regions.title("Realtime Merit Function")
            self.canvas_regions = FigureCanvasTkAgg(self.fig_regions, master=self.tk_regions)
            self.canvas_regions.get_tk_widget().grid(column=0, row=0)
            self.tk_regions.update_idletasks()
            self.plotRegions = True


    def plot_parameters_post_trend(self, parameters_vec, post_vec, sim_param=None, saveflag=False):
        fig_param, axis_param = plt.subplots(3,2, figsize=(9, 10.5))
        fig_param.tight_layout(rect=(0.03, 0.02, 1, 0.97), w_pad=-1.5, h_pad= 2)
        fig_param.suptitle(r'\textbf{Minimize results as parameter estimation}')
        # Seeing
        axis_param[0,0].set_yscale('log')
        axis_param[0,0].set_xlabel(r'Seeing')
        axis_param[0,0].set_ylabel(r'Posterior')
        axis_param[0,0].grid()
        # Weight
        axis_param[0,1].set_yscale('log')
        axis_param[0,1].set_xlabel(r'Cn$^2$ weight of the lower layer')
        axis_param[0,1].grid()
        axis_param[0,1].set_yticklabels([])
        # Velocity 1
        axis_param[1,0].set_yscale('log')
        axis_param[1,0].set_xlabel(r'Lower layer wind velocity [m/s]')
        axis_param[1,0].set_ylabel(r'Posterior')
        axis_param[1,0].grid()
        # Velocity 2
        axis_param[1,1].set_yscale('log')
        axis_param[1,1].set_xlabel(r'Higher layer wind velocity [m/s]')
        axis_param[1,1].grid()
        axis_param[1,1].set_yticklabels([])
        # Direction 1
        axis_param[2,0].set_yscale('log')
        axis_param[2,0].set_xlabel(r'Lower layer wind direction [deg]')
        axis_param[2,0].set_ylabel(r'Posterior')
        axis_param[2,0].grid()
        # Direction 2
        axis_param[2,1].set_yscale('log')
        axis_param[2,1].set_xlabel(r'Higher layer wind direction [deg]')
        axis_param[2,1].grid()
        axis_param[2,1].set_yticklabels([])
        # Plot simulation values
        if not sim_param is None:
            axis_param[0,0].axvline(sim_param[0], color='r', lw=1.5)
            axis_param[0,1].axvline(sim_param[1], color='r', lw=1.5)
            axis_param[1,0].axvline(sim_param[2], color='r', lw=1.5)
            axis_param[1,1].axvline(sim_param[3], color='r', lw=1.5)
            axis_param[2,0].axvline(sim_param[4], color='r', lw=1.5)
            axis_param[2,1].axvline(sim_param[5], color='r', lw=1.5)
        elif not self.sim_param is None:
            axis_param[0,0].axvline(self.sim_param[0], color='r', lw=1.5)
            axis_param[0,1].axvline(self.sim_param[1], color='r', lw=1.5)
            axis_param[1,0].axvline(self.sim_param[2], color='r', lw=1.5)
            axis_param[2,0].axvline(self.sim_param[3], color='r', lw=1.5)
            axis_param[1,1].axvline(self.sim_param[4], color='r', lw=1.5)
            axis_param[2,1].axvline(self.sim_param[5], color='r', lw=1.5)
        # Make plot
        seeing_vec = parameters_vec[:,0]
        weight_vec = parameters_vec[:,1]
        vel1_vec   = parameters_vec[:,2]
        dir1_vec   = parameters_vec[:,3]
        vel2_vec   = parameters_vec[:,4]
        dir2_vec   = parameters_vec[:,5]
        axis_param[0,0].plot(seeing_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        axis_param[0,1].plot(weight_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        axis_param[1,0].plot(vel1_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        axis_param[1,1].plot(vel2_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        axis_param[2,0].plot(dir1_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        axis_param[2,1].plot(dir2_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        # Save figure
        if saveflag:
            plt.savefig(os.path.join(self.outdir, "Parameters_vs_posterior.png"), dpi=600)
        return fig_param, axis_param

    def update_parameters_post_trend(self, parameters_vec, post_vec):
        # Remove prevoius lines
        [line for line in self.axis_param[0,0].get_lines()][-1].remove()
        [line for line in self.axis_param[0,1].get_lines()][-1].remove()
        [line for line in self.axis_param[1,0].get_lines()][-1].remove()
        [line for line in self.axis_param[1,1].get_lines()][-1].remove()
        [line for line in self.axis_param[2,0].get_lines()][-1].remove()
        [line for line in self.axis_param[2,1].get_lines()][-1].remove()
        # Unpack array
        seeing_vec = parameters_vec[:,0]
        weight_vec = parameters_vec[:,1]
        vel1_vec   = parameters_vec[:,2]
        dir1_vec   = parameters_vec[:,3]
        vel2_vec   = parameters_vec[:,4]
        dir2_vec   = parameters_vec[:,5]
        # Add updated lines
        self.axis_param[0,0].plot(seeing_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        self.axis_param[0,1].plot(weight_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        self.axis_param[1,0].plot(vel1_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        self.axis_param[1,1].plot(vel2_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        self.axis_param[2,0].plot(dir1_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        self.axis_param[2,1].plot(dir2_vec, post_vec, mfc='k', ms=8, marker='.', mew=0, ls=':', c='grey')
        # Update tk
        self.canvas_param.draw()
        self.tk_param.update_idletasks()

    def plot_post_trend_for_stars(self, post_star_vec, saveflag=False):
        fig_startrend, axis_startrend = plt.subplots(1,1, figsize=(10,8))
        fig_startrend.tight_layout(rect=(0.03, 0.02, 1, 0.97), w_pad=-1.5, h_pad= 2)
        fig_startrend.suptitle(r'\textbf{Posterior trend for each star}')
        axis_startrend.set_xlabel(r'Iteration number')
        axis_startrend.set_ylabel(r'Posterior')
        axis_startrend.grid()
        axis_startrend.set_yscale('log')
        axis_startrend.plot(post_star_vec)
        axis_startrend.legend()
        # Save figure
        if saveflag:
            plt.savefig(os.path.join(self.outdir, "Posterior_trend_for_each_star.png"), dpi=600)
        return fig_startrend, axis_startrend

    def update_post_trend_for_stars(self, post_star_vec):
        for line in self.axis_startrend.get_lines():
            line.remove()

        star_number = post_star_vec.shape[1]
        for ii in range(star_number):
            self.axis_startrend.plot(post_star_vec[:,ii], label=str(ii), color=self.cmap(ii/star_number))

        self.axis_startrend.legend()
        self.canvas_startrend.draw()
        self.tk_startrend.update_idletasks()


    def plot_regions(self, regions, radii=None, fig=None, axis=None, title=' ', saveflag=False, is_residuals=False, outdir=None, ftag='Regions', stars_numbres_vec=None, showflag=True, share_cbr=False):
        # Calc number of regions and grid size
        if len(regions.shape) == 3:
            num_regions = regions.shape[0]
        elif len(regions.shape) == 2:
            num_regions = 1
        num_col  =  int(np.sqrt(num_regions))
        num_col  += (num_col**2 < num_regions)
        num_row  =  int(num_regions/num_col)
        num_row  += (num_regions%num_col > 0)
        # Reshape the radii in case it is not an array or it is just a value
        if not radii is None:
            radii = np.array(radii)
            if len(radii.shape) == 0:
                radii = np.full(num_regions, radii)
        # Check the axis and figure
        if axis is None:
            if fig is None:
                # Fixed values for graphic
                height_suptitle = 0.39
                width_ylegend = 0.3
                pad        = 0.5
                # Compute graphic dimensions
                fig_width  = width_ylegend + 3.2333*num_col
                fig_height = height_suptitle + 4.2033*num_row
                rect_left  = width_ylegend/fig_width
                rect_top   = (fig_height-height_suptitle)/fig_height
                sup_pos    = (fig_height-height_suptitle/2)/fig_height
                # Create figure, subplots and title
                fig, axis = plt.subplots(num_row, num_col, figsize=(fig_width, fig_height))
                fig.tight_layout(rect=(rect_left, 0, 1, rect_top), h_pad=2.2, w_pad=0.5)
                fig.suptitle(r'\textbf{'+title+'}', y=sup_pos)
            else:
                axis = fig.axes
                print(axis)
        else:
            if np.prod(axis.shape) < num_regions:
                raise Warning('Numbers of axis smaller than regions')
            if fig is None:
                fig = np.array(axis).flatten()[0].get_figure()
        # If use same scale
        if share_cbr:
            cbar_minmax = self._calc_cbr_lowtop(regions, radii)
        else:
            cbar_minmax = None
        # Find the axis-regions map
        if num_regions == 1:
            self.plot_one_region(fig, axis, regions, radii, idx=0, is_residuals=is_residuals, cbar_minmax=cbar_minmax)
        else:
            if len(axis.shape) == 1: # One line plots
                num_col = axis.shape[0]
                num_row = 1
                axis_map = np.empty(num_regions, dtype=np.int8)
                for ii in range(num_regions):
                    if ii < num_col:
                        axis_map[ii] = ii
                    else:
                        axis_map[ii] = np.nan
            elif len(axis.shape) == 2: # Multipole lines plot
                num_row = axis.shape[0]
                num_col = axis.shape[1]
                axis_map = np.empty((num_regions,2), dtype=np.int8)
                for ii in range(num_regions):
                    col_idx = int(ii%num_col)
                    row_idx = int(ii/num_col)
                    if row_idx < num_row:
                        axis_map[ii] = [row_idx, col_idx]
                    else:
                        axis_map[ii] = np.nan
            else:
                raise ValueError('Unexpected number of axis: given {}, but has to be 1 or 2'.format(len(axis.shape)))
            # Populayte the sub-plots
            for ii in range(num_regions):
                if not np.isnan(np.prod(axis_map[ii])):
                    if num_row == 1:
                        ax = axis[axis_map[ii]]
                        n_col = axis_map[ii]
                    elif num_row >= 1:
                        ax = axis[axis_map[ii,0],axis_map[ii,1]]
                        n_col = axis_map[ii,1]
                    else:
                        break
                    # Stars number
                    if stars_numbres_vec is None:
                        idx=ii
                    else:
                        idx = stars_numbres_vec[ii]
                    # Make the plot
                    if not radii is None:
                        radii = np.array(radii)
                        if radii.shape[0] == 1:
                            self.plot_one_region(ax, regions[ii], radii[0], idx=idx, n_col=n_col, is_residuals=is_residuals, cbar_minmax=cbar_minmax)
                        elif ii < radii.shape[0]:
                            self.plot_one_region(ax, regions[ii], radii[ii], idx=idx, n_col=n_col, is_residuals=is_residuals, cbar_minmax=cbar_minmax)
                        else:
                            self.plot_one_region(ax, regions[ii], idx=idx, n_col=n_col, is_residuals=is_residuals, cbar_minmax=cbar_minmax)
                    else:
                        self.plot_one_region(ax, regions[ii], idx=idx, n_col=n_col, is_residuals=is_residuals, cbar_minmax=cbar_minmax)
            # remove empty axis
            num_white_axis = num_col*num_row - num_regions
            if num_white_axis > 0 :
                for ii in range(num_white_axis):
                    fig.delaxes(axis[-1,-1-ii])
        # Save figure
        if saveflag:
            if outdir is not None:
                if not os.path.isdir(outdir):
                    try:
                        os.makedirs(outdir)
                    except FileExistsError:
                        print(f"Directory '{outdir}' already exists.")
                    except PermissionError:
                        print(f"Permission denied: Unable to create '{outdir}'.")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                plt.savefig(os.path.join(outdir, ftag+'.png'))
            elif self.outdir is None:
                plt.savefig(ftag+'.png')
            else:
                plt.savefig(os.path.join(self.outdir, ftag+'.png'))
        # Close the figure
        if not showflag:
            plt.close(fig)
            fig = None
            axis = None
        # Return
        return fig, axis

    def update_regions(self, regions, radii=None):
        num_col = self.axis_regions.shape[1]
        num_row = self.axis_regions.shape[0]
        num_regions = regions.shape[0]
        # Draw all the regions
        for ii in range(regions.shape[0]):
            ax = self.axis_regions[int(ii/num_col), int(ii%num_col)]
            ax.images[-1].colorbar.remove()
            ax.clear()
            self.plot_one_region(ax, regions[ii], radius=radii[ii], idx=ii, n_col=int(ii%num_col), is_residuals=True)
        # Update
        self.canvas_regions.draw()
        self.tk_regions.update_idletasks()


    def plot_2D_param(self, mode, xx, yy, merit_vecs, sim_param=None, lvs=50, fig_path=None, showplot=True, saveplot=False):
        # Create plot
        Y, X = np.meshgrid(yy, xx)
        fig = plt.figure()
        plt.contourf(X, Y, np.log(merit_vecs), levels=lvs)
        plt.grid()
        # Set axis label
        if mode == 0: # Seeing & Cn2
            plt.xlabel(r'Seeing')
            plt.ylabel(r'Cn2 Weight')
        elif mode == 1: # Ground Wind
            plt.xlabel(r'Wind Velocity Ground')
            plt.ylabel(r'Wind Direction Ground')
        elif mode == 2: # High Wind
            plt.xlabel(r'Wind Velocity High')
            plt.ylabel(r'Wind Direction High')
        # Plot refence lines
        if not sim_param is None:
            if mode == 0: # Seeing & Cn2
                plt.axvline(sim_param[0], c='r', lw=2)
                plt.axhline(sim_param[1], c='r', lw=2)
            elif mode == 1: # Ground Wind
                plt.axvline(sim_param[2], c='r', lw=2)
                plt.axhline(sim_param[3], c='r', lw=2)
            elif mode == 2: # High Wind
                plt.axvline(sim_param[4], c='r', lw=2)
                plt.axhline(sim_param[5], c='r', lw=2)
        # Save figure
        if saveplot:
            fig.savefig(fig_path, dpi=400)
        # Delete plot if requests
        if not showplot:
            plt.close(fig)
        else:
            return fig

    # =======================================================================
    #  AO estimation post-processing plots
    # =======================================================================

    @staticmethod
    def switch_fwhm_orientation(fig, ax_fwhm, orientation='horizontal'):
        ax0, ax1 = ax_fwhm
        if orientation == 'vertical':
            fig.set_size_inches(6, 6)
            ax0.set_position([0.15, 0.53, 0.80, 0.40])
            ax1.set_position([0.15, 0.10, 0.80, 0.40])
            ax0.tick_params(labelbottom=False)
            ax1.tick_params(labelbottom=True, labelleft=True)
            ax0.set_xlabel('')
            ax0.set_ylabel(r'FWHM major [mas]')
            ax1.set_xlabel(r'Distance from AO [arcsec]')
            ax1.set_ylabel(r'FWHM minor [mas]')
        elif orientation == 'horizontal':
            fig.set_size_inches(11, 4.5)
            ax0.set_position([0.08, 0.15, 0.40, 0.75])
            ax1.set_position([0.58, 0.15, 0.40, 0.75])
            ax0.tick_params(labelbottom=True, labelleft=True)
            ax1.tick_params(labelbottom=True, labelleft=True)
            ax0.set_xlabel(r'Distance from AO [arcsec]')
            ax0.set_ylabel(r'FWHM major [mas]')
            ax1.set_xlabel(r'Distance from AO [arcsec]')
            ax1.set_ylabel(r'FWHM minor [mas]')
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal', got %r" % orientation)

    def plot_merit_trend(self, merit_gens, label, fpath):
        cum_min = np.minimum.accumulate(merit_gens)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(merit_gens, c='#777777', lw=1.5, label='Merit')
        plt.plot(cum_min, c='k', lw=1.5, label='Minimum')
        plt.legend()
        plt.yscale('log')
        plt.xlabel(r'Iteration number')
        plt.ylabel(r'Merit function [\% data peak]')
        plt.title(r'\textbf{Merit trend -- ' + label + '}')
        plt.grid()
        fig.savefig(fpath, dpi=400)
        plt.close(fig)

    def plot_star_regions(self, outdir, result):
        """
        Every region/residual grid (data, reconstructed, residuals,
        residuals-shared-scale) for one analysis run's reference (+
        control) stars, via plot_regions() above. Named distinctly from
        plot_regions (which plots a single grid) to avoid confusion --
        this calls it several times.
        """
        self.plot_regions(result.data_regions.data_regions, radii=result.radii, title='Data regions',
                           ftag='Data_regions', stars_numbres_vec=result.select_stars,
                           showflag=self.showflag, saveflag=True, outdir=outdir)
        self.plot_regions(result.rec_data_regions, radii=result.radii, title='Reconstructed data regions',
                           ftag='Data_reconstruct_regions', stars_numbres_vec=result.select_stars,
                           showflag=self.showflag, saveflag=True, outdir=outdir)
        self.plot_regions(result.map_regions, radii=result.radii, title='Residuals', ftag='Data_residuals',
                           is_residuals=True, stars_numbres_vec=result.select_stars,
                           showflag=self.showflag, saveflag=True, outdir=outdir)
        self.plot_regions(result.map_regions, radii=result.radii, title='Residuals (shared scale)',
                           ftag='Data_residuals_share_cbr', is_residuals=True, stars_numbres_vec=result.select_stars,
                           showflag=self.showflag, saveflag=True, share_cbr=True, outdir=outdir)
        if result.has_control_stars:
            self.plot_regions(result.rem_data_regions.data_regions, radii=result.rem_radii, title='Remaining data regions',
                               ftag='Remaining_regions', stars_numbres_vec=result.remaining_indices,
                               showflag=self.showflag, saveflag=True, outdir=outdir)
            self.plot_regions(result.rec_remain_regions, radii=result.rem_radii, title='Remaining reconstructed regions',
                               ftag='Remaining_reconstruct_regions', stars_numbres_vec=result.remaining_indices,
                               showflag=self.showflag, saveflag=True, outdir=outdir)
            self.plot_regions(result.rem_map_regions, radii=result.rem_radii, title='Remaining residuals',
                               ftag='Remaining_residuals', is_residuals=True, stars_numbres_vec=result.remaining_indices,
                               showflag=self.showflag, saveflag=True, outdir=outdir)
            self.plot_regions(result.rem_map_regions, radii=result.rem_radii, title='Remaining residuals (shared scale)',
                               ftag='Remaining_residuals_share_cbr', is_residuals=True, stars_numbres_vec=result.remaining_indices,
                               showflag=self.showflag, saveflag=True, share_cbr=True, outdir=outdir)

    @staticmethod
    def _plot_one_fwhm_axis(ax, dist_ref, dist_ctrl, data_ref, rec_ref, data_ctrl, rec_ctrl,
                             rec_ref_err=None, rec_ctrl_err=None):
        ax.plot(dist_ref, data_ref, label='data ref', ls='', marker='o', ms=8)
        if rec_ref_err is None:
            ax.plot(dist_ref, rec_ref, label='rec ref', ls='', marker='o', ms=8)
        else:
            ax.errorbar(dist_ref, rec_ref, rec_ref_err, label='rec ref', ls='', marker='o', ms=8, capsize=5)
        if not np.isnan(dist_ctrl[0]):
            ax.plot(dist_ctrl, data_ctrl, label='data ctrl', ls='', marker='D', ms=4)
            if rec_ctrl_err is None:
                ax.plot(dist_ctrl, rec_ctrl, label='rec ctrl', ls='', marker='D', ms=4)
            else:
                ax.errorbar(dist_ctrl, rec_ctrl, rec_ctrl_err, label='rec ctrl', ls='', marker='D', ms=4, capsize=5)
        ax.plot(dist_ref, data_ref - rec_ref, label='res ref', ls='', marker='P', ms=8)
        if not np.isnan(dist_ctrl[0]):
            ax.plot(dist_ctrl, data_ctrl - rec_ctrl, label='res ctrl', ls='', marker='P', ms=5)
        ax.legend(fontsize='x-small', ncols=2, loc='upper left', handletextpad=0.4, handlelength=1, columnspacing=1)
        ax.grid()

    def plot_fwhm(self, dist_ref, dist_ctrl, data_std_vec, reconstruct_std_vec,
                  data_remain_std_vec, reconstruct_remain_std_vec, outdir,
                  reconstruct_std_err=None, reconstruct_remain_std_err=None):
        """
        reconstruct_std_err / reconstruct_remain_std_err (optional): std of
        the reconstructed FWHM across several analyses, same shape as
        reconstruct_std_vec / reconstruct_remain_std_vec -- when given, the
        reconstructed points are drawn with error bars instead of plain
        markers (used for the project-level mean-across-analyses plot).
        """
        order_ref = np.argsort(dist_ref)
        order_ctrl = np.argsort(dist_ctrl)
        dist_ref = dist_ref[order_ref]
        data_std_vec = data_std_vec[order_ref]
        reconstruct_std_vec = reconstruct_std_vec[order_ref]
        if reconstruct_std_err is not None:
            reconstruct_std_err = reconstruct_std_err[order_ref]
        if order_ctrl.shape[0] > 0 and not np.isnan(dist_ctrl[0]):
            dist_ctrl = dist_ctrl[order_ctrl]
            data_remain_std_vec = data_remain_std_vec[order_ctrl]
            reconstruct_remain_std_vec = reconstruct_remain_std_vec[order_ctrl]
            if reconstruct_remain_std_err is not None:
                reconstruct_remain_std_err = reconstruct_remain_std_err[order_ctrl]
        else:
            dist_ctrl = np.full(1, np.nan)
            data_remain_std_vec = np.full((1, 2), np.nan)
            reconstruct_remain_std_vec = np.full((1, 2), np.nan)
            reconstruct_remain_std_err = None

        gridspec_kw = {'right': 0.97, 'top': 0.93, 'hspace': 0.02}
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), sharex=True, gridspec_kw=gridspec_kw)
        self._plot_one_fwhm_axis(
            ax[0], dist_ref, dist_ctrl, data_std_vec[:, 0], reconstruct_std_vec[:, 0],
            data_remain_std_vec[:, 0], reconstruct_remain_std_vec[:, 0],
            rec_ref_err=(reconstruct_std_err[:, 0] if reconstruct_std_err is not None else None),
            rec_ctrl_err=(reconstruct_remain_std_err[:, 0] if reconstruct_remain_std_err is not None else None))
        ax[0].set_ylabel(r'FWHM major [mas]')
        self._plot_one_fwhm_axis(
            ax[1], dist_ref, dist_ctrl, data_std_vec[:, 1], reconstruct_std_vec[:, 1],
            data_remain_std_vec[:, 1], reconstruct_remain_std_vec[:, 1],
            rec_ref_err=(reconstruct_std_err[:, 1] if reconstruct_std_err is not None else None),
            rec_ctrl_err=(reconstruct_remain_std_err[:, 1] if reconstruct_remain_std_err is not None else None))
        ax[1].set_xlabel(r'Distance from AO [arcsec]')
        ax[1].set_ylabel(r'FWHM minor [mas]')
        fig.suptitle(r'\textbf{Trend of PSF FWHM across the field}')

        y_limit = 1e3
        for axis, col in ((ax[0], 0), (ax[1], 1)):
            peak = max(np.nanmax(data_std_vec[:, col]), np.nanmax(reconstruct_std_vec[:, col]),
                       np.nanmax(data_remain_std_vec[:, col]), np.nanmax(reconstruct_remain_std_vec[:, col]))
            if peak > y_limit:
                y_min = min(-50, np.nanmin(data_std_vec[:, col] - reconstruct_std_vec[:, col]),
                            np.nanmin(data_remain_std_vec[:, col] - reconstruct_remain_std_vec[:, col]))
                axis.set_ylim([y_min, y_limit])

        fig.savefig(os.path.join(outdir, 'FWHM.png'), dpi=400)
        self.switch_fwhm_orientation(fig, ax)
        fig.savefig(os.path.join(outdir, 'FWHM_landscape.png'), dpi=400)
        if not self.showflag:
            plt.close(fig)

    def plot_flux(self, result, dist_ref, dist_ctrl, outdir):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5), sharey=True)
        ax0.set_position([0.09, 0.15, 0.43, 0.75])
        ax1.set_position([0.54, 0.15, 0.43, 0.75])

        ax0.plot(result.tot_flux_ref, result.err_flux_ref * 100, label='reference', ls='', marker='o', ms=8)
        ax1.plot(dist_ref, result.err_flux_ref * 100, label='reference', ls='', marker='o', ms=8)
        if result.has_control_stars:
            ax0.plot(result.tot_flux_ctrl, result.err_flux_ctrl * 100, label='control', ls='', marker='D', ms=8)
            ax1.plot(dist_ctrl, result.err_flux_ctrl * 100, label='control', ls='', marker='D', ms=8)

        ax0.set_xscale('log')
        ax0.set_xlabel(r'Observed Flux [log ADU]')
        ax0.set_ylabel(r'Relative Flux Error [\%]')
        ax0.grid()
        ax0.legend(fontsize='x-small', loc='lower right', handletextpad=0.4, handlelength=1, columnspacing=1)
        ax1.set_xlabel(r'Distance from AO [arcsec]')
        ax1.grid()

        fig.suptitle(r'\textbf{Flux error vs.\ flux and distance}')
        fig.savefig(os.path.join(outdir, 'Fluxes.png'), dpi=400)
        plt.close(fig)

    def plot_fwhm_error_vs_flux(self, result, outdir):
        """
        Relative FWHM residual -- (data - reconstructed) / data * 100, same
        quantity as the FWHM/results tables -- against observed flux, major
        and minor axis together on one plot, distinguished by marker
        ('o' major, 's' minor); reference stars filled, control stars open,
        so both distinctions are visible without a third encoding.
        """
        def rel_err(data_vec, rec_vec, col):
            d, r = data_vec[:, col], rec_vec[:, col]
            return (d - r) / d * 100

        fig, ax = plt.subplots(figsize=(7, 5))
        color_major, color_minor = 'tab:blue', 'tab:orange'

        err_major_ref = rel_err(result.data_std_vec, result.reconstruct_std_vec, 0)
        err_minor_ref = rel_err(result.data_std_vec, result.reconstruct_std_vec, 1)
        ax.plot(result.tot_flux_ref, err_major_ref, ls='', marker='o', ms=7,
                color=color_major, label='major (ref)')
        ax.plot(result.tot_flux_ref, err_minor_ref, ls='', marker='s', ms=7,
                color=color_minor, label='minor (ref)')

        if result.has_control_stars:
            err_major_ctrl = rel_err(result.data_remain_std_vec, result.reconstruct_remain_std_vec, 0)
            err_minor_ctrl = rel_err(result.data_remain_std_vec, result.reconstruct_remain_std_vec, 1)
            ax.plot(result.tot_flux_ctrl, err_major_ctrl, ls='', marker='o', ms=6, mfc='none',
                    color=color_major, label='major (ctrl)')
            ax.plot(result.tot_flux_ctrl, err_minor_ctrl, ls='', marker='s', ms=6, mfc='none',
                    color=color_minor, label='minor (ctrl)')

        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xscale('log')
        ax.set_xlabel(r'Observed Flux [log ADU]')
        ax.set_ylabel(r'Relative FWHM error [\%]')
        ax.grid()
        ax.legend(fontsize='x-small', ncols=2, loc='best', handletextpad=0.4, handlelength=1, columnspacing=1)
        fig.suptitle(r'\textbf{Relative FWHM error vs.\ flux}')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'FWHM_error_vs_flux.png'), dpi=400)
        plt.close(fig)

    @staticmethod
    def _px_to_arcsec(pos_px, center_px, pixscale):
        """
        Convert pixel position(s) to arcsec, referenced to the image centre:
        pixel offset from centre -> mas (via the sensor's pixel scale,
        assumed arcsec/pixel) -> arcsec.
        """
        offset_px = np.atleast_2d(pos_px) - center_px
        offset_mas = offset_px * (pixscale * 1000.0)
        return offset_mas / 1000.0

    def plot_field_positions(self, all_pos_field, select_stars, remaining_indices, ao_pos, outdir,
                              pixscale=0.015, tel_res=2048):
        """
        The field layout, in arcsec relative to the image centre (pixel
        position converted to mas then arcsec via the sensor's pixel
        scale): "Used stars" (reference) red circles, "Other stars"
        (control) grey diamonds, the "AO star" (if known) a blue dot. Each
        point is labeled with its star id (the index used everywhere else
        in this pipeline). Avoided/unused stars are not shown.
        """
        center_px = tel_res / 2.0
        pos_arcsec = self._px_to_arcsec(all_pos_field, center_px, pixscale)

        fig, ax = plt.subplots(figsize=(8, 8))

        if len(remaining_indices) > 0:
            ax.plot(pos_arcsec[remaining_indices, 0], pos_arcsec[remaining_indices, 1],
                    ls='', marker='D', ms=8, color='grey', label='Other stars')
            for idx in remaining_indices:
                ax.annotate(str(idx), pos_arcsec[idx], textcoords='offset points',
                             xytext=(0, 8), ha='center', fontsize='small')

        if len(select_stars) > 0:
            ax.plot(pos_arcsec[select_stars, 0], pos_arcsec[select_stars, 1],
                    ls='', marker='o', ms=9, color='red', label='Used stars')
            for idx in select_stars:
                ax.annotate(str(idx), pos_arcsec[idx], textcoords='offset points',
                             xytext=(0, 8), ha='center', fontsize='small')

        if ao_pos is not None:
            ao_arcsec = self._px_to_arcsec(ao_pos, center_px, pixscale)
            ax.plot(ao_arcsec[:, 0], ao_arcsec[:, 1], ls='', marker='o', ms=9, color='blue', label='AO star')

        ax.set_xlabel(r'x-position [arcsec]')
        ax.set_ylabel(r'y-position [arcsec]')
        ax.set_aspect('equal', adjustable='box')
        ax.grid()
        ax.legend(fontsize='small', loc='lower left')
        ax.set_title(r'\textbf{Star position}')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'Field_positions.png'), dpi=400)
        plt.close(fig)

    def plot_branch_comparison(self, branch_results, project_output_dir, colors):
        branch_labels = list(branch_results.keys())
        if not branch_labels:
            print('No branches to compare.')
            return

        fig_mf, ax_mf = plt.subplots(figsize=(7, 5))
        for ii, label in enumerate(branch_labels):
            res = branch_results[label]
            order = np.argsort(res['dist'])
            ax_mf.plot(res['dist'][order], res['mf_mean'][order], marker='o', ls=':', ms=7,
                       color=colors[ii % len(colors)], label=label)
        ax_mf.set_xlabel(r'Distance from AO [arcsec]')
        ax_mf.set_ylabel(r'Merit function')
        ax_mf.set_title(r'\textbf{Merit function vs.\ distance by branch}')
        ax_mf.grid()
        ax_mf.legend(fontsize='x-small', loc='upper right')
        fig_mf.tight_layout()
        fig_mf.savefig(os.path.join(project_output_dir, 'MF_vs_distance_branches.png'), dpi=400)
        plt.close(fig_mf)

        fig_fwhm, ax_fwhm = plt.subplots(nrows=2, ncols=1, figsize=(7, 8), sharex=True,
                                          gridspec_kw={'right': 0.97, 'top': 0.93, 'hspace': 0.05})
        ref_res = branch_results[branch_labels[0]]
        ref_order = np.argsort(ref_res['dist'])
        fwhm_data_major = ref_res['data_fwhm_major'][ref_order]
        fwhm_data_minor = ref_res['data_fwhm_minor'][ref_order]
        for ii, label in enumerate(branch_labels):
            res = branch_results[label]
            order = np.argsort(res['dist'])
            color = colors[ii % len(colors)]
            ax_fwhm[0].errorbar(res['dist'][order], res['fwhm_major_mean'][order], res['fwhm_major_std'][order],
                                 marker='o', ls=' ', ms=5, capsize=4, color=color, label=label)
            ax_fwhm[1].errorbar(res['dist'][order], res['fwhm_minor_mean'][order], res['fwhm_minor_std'][order],
                                 marker='o', ls=' ', ms=5, capsize=4, color=color, label=label)
        ax_fwhm[0].plot(ref_res['dist'][ref_order], fwhm_data_major, marker='x', ls=' ', ms=7, color='k', label='data')
        ax_fwhm[1].plot(ref_res['dist'][ref_order], fwhm_data_minor, marker='x', ls=' ', ms=7, color='k', label='data')
        ax_fwhm[0].set_ylabel(r'FWHM major [mas]')
        ax_fwhm[1].set_ylabel(r'FWHM minor [mas]')
        ax_fwhm[1].set_xlabel(r'Distance from AO [arcsec]')
        ax_fwhm[0].grid(); ax_fwhm[1].grid()
        ax_fwhm[0].legend(fontsize='x-small', ncols=2, loc='lower right')
        fig_fwhm.suptitle(r'\textbf{FWHM vs.\ distance by branch}')
        fig_fwhm.savefig(os.path.join(project_output_dir, 'FWHM_vs_distance_branches.png'), dpi=400)
        self.switch_fwhm_orientation(fig_fwhm, ax_fwhm)
        fig_fwhm.savefig(os.path.join(project_output_dir, 'FWHM_vs_distance_branches_landscape.png'), dpi=400)
        plt.close(fig_fwhm)
        print('Saved branch-comparison plots to %s' % project_output_dir)