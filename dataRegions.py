import numpy as np
import matplotlib.pyplot as plt
import os

from graphics import Graphics
import pixelUtiles as pxu
import warnings

class dataRegions:

    def __init__ (self, data, star_pos, box_edge, outdir=None, doPlot=False):
        self.box_edge = int(box_edge)
        self.n_regions = star_pos.shape[0]
        self.data_regions = None
        self.radii = None
        self.star_pos = star_pos

        self.fig_ftag = 'DataRegions'
        self.outdir = outdir
        if self.outdir is None:
            self.fpath =  None
        else:
            self.fpath = os.path.join(outdir,'data_regions.npy')

        self.makeDataRegions(data, star_pos, box_edge, saveplot=True)

        self.radii = np.full(self.n_regions , np.nan)
        for ii in range(self.n_regions):
            self.radii[ii] = self.calcArea(self.data_regions[ii])

        self.save()

        if doPlot:
            self.plot()
        #self.removeBG()

    def makeDataRegions(self, data, star_pos_px, box_edge, saveplot=False):
        data_len = data.shape[0]
        data_regions = np.full((star_pos_px.shape[0], box_edge, box_edge), np.nan)
        ii = 0
        for (xx, yy) in star_pos_px.astype(np.int32):
            #  Find regions limits
            MA_lim, mi_lim = pxu.find_bound_regions(xx, yy, data_len, box_edge)
            # Cast data into regions
            data_regions[ii, mi_lim[1,0]:mi_lim[1,1], mi_lim[0,0]:mi_lim[0,1] ] = data[MA_lim[1,0]:MA_lim[1,1], MA_lim[0,0]:MA_lim[0,1]]
            #Shift index
            ii += 1
        # Store data
        self.data_regions = data_regions
        if self.outdir is not None:
            if saveplot:
                fig = self.plot()
                fig.savefig(os.path.join(self.outdir, self.fig_ftag+'.png'), dpi=400)
                plt.close(fig)
        return data_regions

    def calcArea(self, data, perc=0.7):
        data_copy = data.copy()
        data_copy[data_copy<0] = np.nan
        data_no_nan = data_copy[~np.isnan(data_copy)]
        total = np.nansum(data_no_nan)
        data_sort = np.sort(data_no_nan)
        data_sum = np.cumsum(data_sort)
        min_area = data_sum[data_sum > (1-perc)*total]
        idx = np.where(data_sum == min_area[0])[0]
        X, Y = np.where(data_copy == data_sort[idx])
        distance = np.sqrt((X-data_copy.shape[0]/2)**2 + (Y-data_copy.shape[1]/2)**2)
        min_dist = np.min(distance)
        return min_dist

    def removeBG(self):
        plt.figure()
        temp = self.data_regions[0].copy()
        temp[pxu.create_inverse_annular_mask(temp, 1,10)] = np.nan
        ttemp = temp.flatten()
        ttemp = ttemp[~np.isnan(ttemp)]
        count, bin = np.histogram(ttemp)
        plt.plot(count)
        print(np.nanmean(temp), np.nanstd(temp))
        plt.figure()
        plt.imshow(temp)

    def select_regions(self, radii, doPlot=False):
        # Selecting data regions
        self.unselected_data_regions = self.data_regions.copy()
        self.unselected_radii = self.radii.copy()
        self.unselected_star_pos = self.star_pos.copy()

        n_selc = np.count_nonzero(~np.isnan(radii))
        if n_selc >= self.n_regions:
            warnings.warn("Set radii without selecting stars")
            self.radii = radii
        else:
            self.data_regions = np.empty((n_selc, self.box_edge, self.box_edge))
            self.radii = np.empty(n_selc)
            self.star_pos = np.empty((n_selc, 2))

            # cast elements
            jj = 0
            for ii in range(self.n_regions):
                if ~np.isnan(radii[ii]):
                    self.data_regions[jj] = self.unselected_data_regions[ii]
                    self.radii[jj] = radii[ii]
                    self.star_pos[jj] = self.unselected_star_pos[ii]
                    jj += 1

            self.n_regions  = n_selc

        fig = self.plot()
        if self.outdir is not None:
            fig.savefig(os.path.join(self.outdir, self.fig_ftag+'_selected.png'), dpi=400)
        if not doPlot:
            plt.close(fig)

        self.save()

    def plot(self, radii=None):
        gr = Graphics()
        if radii is None:
            fig, ax = gr.plot_regions(self.data_regions, self.radii)
        else:
            fig, ax = gr.plot_regions(self.data_regions, radii)
        return fig

    def save(self):
        if self.fpath is not None:
            np.save(self.fpath, self.data_regions)
