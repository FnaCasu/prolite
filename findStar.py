import numpy as np # for the mathematical operation
import matplotlib.pyplot as plt   # to make plots
from matplotlib.patches import Circle
from photutils.detection import DAOStarFinder
from matplotlib.colors import LogNorm
import os

import pixelUtiles as pxu

class findStar:

    def findAOstar(self, data, thr_AO, fwhm_AO, min_sep_AO):
        daofind = DAOStarFinder(threshold=thr_AO, fwhm=fwhm_AO, min_separation=min_sep_AO)
        sources = daofind.find_stars(data)
        if sources is None:
            print('No AO-star found with threshold=%d, fwhm=%.1f, min_separation=%d' %(thr_AO, fwhm_AO, min_sep_AO))
        else:
            for col in sources.colnames:
                if col not in ('id', 'npix'):
                    sources[col].info.format = '%.2f'  # for consistent table output
            sources.sort(keys='flux', reverse=True)
            self.pos_AO = np.array([[sources[0]['xcentroid'], sources[0]['ycentroid']]])
            if not(self.ax is None):
                for pos in self.pos_AO:
                    cir = Circle(pos, fwhm_AO, color='green', fill=False, lw=3)
                    self.ax.add_patch(cir)

    def findFIELDstar(self, data, fwhm_field, thr_field, min_sep_field, nField, mask_AO_star=None):
        daofind = DAOStarFinder(threshold=thr_field, fwhm=fwhm_field, min_separation=min_sep_field)
        if mask_AO_star is None:
            sources = daofind.find_stars(data)
        else:
            sources = daofind.find_stars(data, mask=mask_AO_star)
        if sources is None:
            print('No fiel stars found with threshold=%d, fwhm=%.1f, min_separation=%d' %(thr_field, fwhm_field, min_sep_field))
        else:
            for col in sources.colnames:
                if col not in ('id', 'npix'):
                    sources[col].info.format = '%.2f'  # for consistent table output
            sources.sort(keys='flux', reverse=True)
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            if nField >= positions.shape[0]:
                nField = -1
            self.pos_field = positions[0:nField]
            self.flux_field = np.array(sources['flux'][0:nField])
            if not(self.ax is None):
                for (x,y) in self.pos_field:
                    cir = Circle((x, y), 15, color='blue', fill=False, lw=4)
                    self.ax.add_patch(cir)

    def __init__(self, data, AO_param, field_param, nField = -1, findAO = True, findFIELD = True, doPlot=True, outdir=None):
        self.fwhm_AO = AO_param['fwhm']
        self.thr_AO = AO_param['thr']
        self.min_sep_AO = AO_param['min_sep']
        self.fwhm_field = field_param['fwhm']
        self.thr_field = field_param['thr']
        self.min_sep_field = field_param['min_sep']
        self.nField = nField
        self.findAO = findAO
        self.findFIELD = findFIELD
        self.doPlot=doPlot
        self.outdir=outdir
        # To save values
        self.pos_AO = None
        self.pos_field = None

        if doPlot or not(outdir is None):
            vmin = np.log(80)
            vmax = np.log(np.max(data))
            norm = LogNorm(vmin=vmin, vmax=vmax)
            self.fig, self.ax = plt.subplots(figsize=(20,20))
            self.ax.imshow(np.log(1+abs(data)), norm=norm, cmap='gist_heat', origin='lower')
            self.ax.set_xlabel('x label')
            self.ax.set_ylabel('y label')
            self.ax.set_title('Star position')
        else:
            self.ax = None

        if findAO:
            self.findAOstar(data, self.thr_AO, self.fwhm_AO, self.min_sep_AO)
            # Create a circulare Mask around the AO star
            mask_AO_star = pxu.circular_mask(data, self.pos_AO[0], 2*self.fwhm_AO)
            self.findFIELDstar(data, self.fwhm_field, self.thr_field, self.min_sep_field, nField, mask_AO_star)
        else:
            self.findFIELDstar(data, self.fwhm_field, self.thr_field, self.min_sep_field, nField)

        if not(outdir is None):
            fname = os.path.join(outdir, 'find_stars.png')
            self.fig.savefig(fname)
            if not doPlot:
                plt.close(self.fig)
