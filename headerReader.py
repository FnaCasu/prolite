from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle # to handle astronomical coordiantes
from astropy.time import Time
from astropy import units as u

class headerReader:
    def __init__(self, hdr):
        self.hdr = hdr

        self.reader()

    def reader(self):
        # Telescope resolution
        self.tel_res = self.hdr['NAXIS1']
        # PixelScale
        self.pixscale = self.hdr['PIXSCALE'] #arcsec/pixe
        self.pixscale = 1e-3*14.9
        # Frequenza loop AO
        self.AO_freq = self.hdr['AO_FREQ']
        # Nuber pf photons in AO wfs
        self.n_photo = self.hdr['AOCOUNTS']
        # Telescope AltAz
        self.telescope_alt = self.hdr['TELALT']
        # self.telescope_az  = self.hdr['TELAZ']
        self.instrument_number = self.hdr['INS_ID']
        self.filters = self.hdr['FILTERS']
        self.wavelenght = self.get_central_lamda(self.filters, self.instrument_number)
        

    @staticmethod
    def get_central_lamda(filters, luci_inst=1):
        # Filter central lambda
        if 'zJspec' in filters:
            lambda_vec = [1.175	,	1.175]
        elif 'HKspec' in filters:
            lambda_vec = [1.95	,	1.953]
        elif 'Y1' in filters:
            lambda_vec = [1.007	,	1.007]
        elif 'Y2' in filters:
            lambda_vec = [1.074	,	1.074]
        elif 'OH_1060' in filters:
            lambda_vec = [1.065	,	1.065]
        elif 'OH_1190' in filters:
            lambda_vec = [1.194	,	1.194]
        elif 'HeI' in filters:
            lambda_vec = [1.088	,	1.088]
        elif 'P_gamma' in filters:
            lambda_vec = [1.097	,	1.096]
        elif 'P_beta' in filters:
            lambda_vec = [1.283	,	1.284]
        elif 'J_low' in filters:
            lambda_vec = [1.199	,	1.199]
        elif 'J_high' in filters:
            lambda_vec = [1.303	,	1.303]
        elif 'FeII' in filters:
            lambda_vec = [1.646	,	1.645]
        elif 'H2' in filters:
            lambda_vec = [2.124	,	2.127]
        elif 'Br_gamma' in filters:
            lambda_vec = [2.17	,	2.171]
        elif 'Ks' in filters:
            lambda_vec = [2.163	,	2.161]
        elif 'z' in filters:
            lambda_vec = [0.957	,	0.965]
        elif 'J' in filters:
            lambda_vec = [1.247	,	1.25]
        elif 'H' in filters:
            lambda_vec = [1.653	,	1.651]
        elif 'K' in filters:
            lambda_vec = [2.194	,	2.199]
        # retun the lambda
        return lambda_vec[int(luci_inst-1)]*1e-6


