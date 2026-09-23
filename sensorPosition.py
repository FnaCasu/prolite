#from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np

class SensorPosition:
    def __init__(self, sensor_dimension, pixel_scale):
        self.sensor_dimension = sensor_dimension
        self.sensor_center    = sensor_dimension/2
        self.pixel_scale      = pixel_scale

    def px2arcsec(self, pos_array):
        return (pos_array - self.sensor_center) * self.pixel_scale

    def arcsec2px(self, arcsec_array):
        return (arcsec_array / self.pixel_scale) + self.sensor_center

    def arcsec2polar(self, arcsec_array):
        if len(arcsec_array.shape) == 1:
            zenith = np.sqrt(arcsec_array[0]**2 + arcsec_array[1]**2)
            azimuth = np.rad2deg(np.arctan2(arcsec_array[0], arcsec_array[1]))
        else:
            zenith = np.sqrt(arcsec_array[:,0]**2 + arcsec_array[:,1]**2)
            azimuth = np.rad2deg(np.arctan2(arcsec_array[:,0], arcsec_array[:,1]))
        return zenith, azimuth

    def px2polar(self, pos_array):
        return self.arcsec2polar(self.px2arcsec(pos_array))

    def polar2arcsec(self, *arg):
        if len(arg) == 1:
            polar_coord = arg[0]
            if isinstance(polar_coord, np.ndarray):
                if len(polar_coord.shape) == 1:
                    zenith  = polar_coord[0]
                    azimuth = polar_coord[1]
                    arcsec_array = np.empty((1,2))
                    arcsec_array[0,0] = zenith * np.sin(np.deg2rad(azimuth))
                    arcsec_array[0,1] = zenith * np.cos(np.deg2rad(azimuth))

                elif len(polar_coord.shape) == 2:
                    zenith  = polar_coord[0, :]
                    azimuth = polar_coord[1, :]
                    arcsec_array = np.empty((zenith.shape[0],2))
                    arcsec_array[:,0] = zenith * np.sin(np.deg2rad(azimuth))
                    arcsec_array[:,1] = zenith * np.cos(np.deg2rad(azimuth))
            else:
                raise ValueError('Unkown format')
        elif len(arg) == 2:
            zenith  = arg[0]
            azimuth = arg[1]
            arcsec_array = np.empty((zenith.shape[0],2))
            arcsec_array[:,0] = zenith * np.sin(np.deg2rad(azimuth))
            arcsec_array[:,1] = zenith * np.cos(np.deg2rad(azimuth))
        return arcsec_array

    def polar2px(self, *args):
        return self.arcsec2px(self.polar2arcsec(*args))

    def change_axes_scale(self, fig):
        ax = fig.gca()
        tick_pos = np.linspace(0, self.sensor_dimension, 7)
        tick_label = np.round(self.px2arcsec(tick_pos), 2)
        ax.set_xticks(tick_pos, tick_label)
        ax.set_yticks(tick_pos, tick_label)
        ax.set_xlabel(r'x-position [arcsec]')
        ax.set_ylabel(r'y-position [arcsec]')

    def create_figure(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        #ax.set_aspect('equal')
        ax.set_title(r'\textbf{Star position}')
        ax.set_xlim([0,self.sensor_dimension])
        ax.set_ylim([0,self.sensor_dimension])
        self.change_axes_scale(fig)
        return fig

    def plot_px(self, pos_array, fig = None, Number=True, **kwargs):
        # Check if there is a previous figure
        if fig is None:
            fig = self.create_figure()
        plt.scatter(pos_array[:,0], pos_array[:,1], **kwargs)
        # Add the number over the stars
        dist = self.sensor_dimension/50
        # If an array number is given
        if isinstance(Number, list):
            Number = np.array(Number)
        if isinstance(Number, np.ndarray):
            if Number.shape[0] == pos_array.shape[0]:
                for ii in range(pos_array.shape[0]):
                    plt.text(pos_array[ii,0]+dist, pos_array[ii,1]+dist, str(Number[ii]))
        # If is just a number
        elif Number:
            for ii in range(pos_array.shape[0]):
                plt.text(pos_array[ii,0]+dist, pos_array[ii,1]+dist, str(ii))
        # Return figure
        return fig

    def plot_arcsec(self, arcsec_array, fig = None):
        pos_array = self.arcsec2px(arcsec_array)
        return self.plot_px(pos_array, fig)

    def plot_polar(self, zenith, azimuth, fig = None, **kwargs):
        pos_array = self.polar2px(zenith, azimuth)
        return self.plot_px(pos_array, fig, **kwargs)
