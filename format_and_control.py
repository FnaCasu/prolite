import numbers
import numpy as np


def _format_radii(radii, n_star):
    # Check stars number
    if not np.isscalar(n_star):
        raise TypeError('n_star must be a scalar')
    # Numpy array
    if isinstance(radii, np.ndarray):
        if len(radii.shape) == 0:
             radii_final = np.full(n_star, radii)
        elif len(radii.shape) == 1:
            if radii.shape[0] == 1:
                radii_final = np.full(n_star, radii[0])
            elif radii.shape[0] ==n_star:
                radii_final = radii.copy()
            elif radii.shape[0] > n_star:
                raise Warning('Given more radii than star numbers, the array has been cut')
                radii_final = radii[:n_star].copy()
            else:
                raise ValueError('Unexpected elements in radii: expected {} numbers, given {}'.format(n_star, radii.shape[0]))
        else:
            raise ValueError('Uxepected dimensions for radii: expected 0 or 1 given {}'.format(len(radii.shape)))
    # Scalar number
    elif np.isscalar(radii):
        radii_final = np.full(n_star, radii)
    # List of values
    elif isinstance(radii, list):
        if len(radii) == 1:
            radii_final = np.full(n_star, radii[0])
        elif len(radii) == n_star:
            radii_final = np.array(radii)
        elif len(radii) > n_star:
            raise Warning('Given more radii than star numbers, the list has been cut')
            radii_final = np.array(radii[:n_star])
        else:
            raise ValueError('Unexpected lenght for radii list: expected 1 or {}, given {}'.format(n_star, len(radii)))
    # Other format
    else:
        raise TypeError('Uxepected type for radii, expected list, np.array or scalar')
    # Returns np.array with dimensions equal to star dimensions
    return radii_final

"""
Zerike base:
    TIPTOP Notation for static aberattion
        zCoefStaticOn = [Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11]
                         0   1   2   3   4   5   6   7   8    9

    ZERIKE notation
        Z2  Tilt seno
        Z3  Tilt coseno
        Z4  Defocus
        Z5  Astigmatismo seno
        Z6  Astigmatismo coseno
        Z7  Coma seno
        Z8  Coma coseno
        Z9  Trifoglio seno
        Z10 Trifoglio coseno
        Z11 Sferica
    
    Angular period = 2\\pi / m
    Selected Mode:
        M0:  Tilt -->  Z2, Z3 -->  m0, deg0 in (0, 360)
        M1:  Defocus -->  Z4   -->  m1
        M2:  Astigmatismo -->  Z5, Z6 --> m2, deg2 in (0, 180)
        M3:  Coma --> Z7, Z8 -->  m3, deg3 in (0, 360)
        M4:  Trifoglio --> Z9, Z10 --> m4, deg4 in (0, 120)
        M5:  Sferica -->  Z11 -->  m5

"""
# Define the mode name list
MODE_NAMES = ['Tilt', 'Defocus', 'Astigmatismo', 'Coma', 'Trifolgio', 'Sferica']
# Define the maximum aplitude for each mode
#                    M0     M1      M2      M3      M4      M5
MODE_DEG = np.array([360,   0,      180,    360,    120,    0   ])

def _make_bounds(bounds, max_abberation_vec=None, max_jitter_vec=None):
    # If there are no NCPA
    if max_abberation_vec is None:
        print('LOG:\tNon Common Paths Abberations are not used.')
        selected_mode = None
    # If there are the NCPA
    else:
        # Make correct data type and copy it
        max_abberation_vec = np.array(max_abberation_vec).copy()
        # Check if the array is made of zeros:
        if all(v == 0 for v in max_abberation_vec):
            print('LOG:\tAll the elements in max_abberation_vec are equal to zero, so Non Common Paths Abberations are not used.')
            selected_mode = None
        # Check if there is some negative value
        if any(v < 0 for v in max_abberation_vec):
            raise ValueError('Max aberration must be positive.')
        # Create an empty array with boolean objcet
        selected_mode = np.full_like(max_abberation_vec.copy(), False, dtype=bool)
        # Prepare the log print:
        msg = 'LOG\tUsed as the NCPA modes the following: '
        # Aggiungo solo i modi che voglio stimare
        for ii, mode in enumerate(max_abberation_vec):
            # Check if the mode is supported
            if ii >= len(MODE_NAMES):
                raise ValueError("Sorry mode not jet supported :\'(")
            # Check if the mode is active
            if mode > 0:
                # Flage the mode as active
                selected_mode[ii] = True
                # Add the mode to the 
                msg += MODE_NAMES[ii]+' '
                # Add the angle if is necessary
                if MODE_DEG[ii] > 0:
                    # Add the amplitude to the bounderies
                    bounds = np.append(bounds, [[0, mode]], axis=0)
                    bounds = np.append(bounds, [[0, MODE_DEG[ii]]], axis=0)
                else:
                    bounds = np.append(bounds, [[-mode, mode]], axis=0)
        # Print the log message
        print(msg)
    # JITTER
    # In case the Jitter is not set
    if max_jitter_vec is None:
        print('LOG:\tJitter are not used.')
        do_jitter = False
    # If is set
    else:
        # Define a friendly format
        max_jitter_vec = np.array(max_jitter_vec).ravel()
        # check the dimension_
        if max_jitter_vec.shape[0] == 2:
            # Check for negative values
            if any(max_jitter_vec <= 0):
                raise ValueError('Elements of max_jitter_vec must be all positive')
            # Check that minor axis is minor than the grater
            if max_jitter_vec[0] < max_jitter_vec[1]:
                raise ValueError('First element of max_jitter_vec must be grater or equal to the second')
            # Major axis jitter 
            bounds = np.append(bounds, [[0, max_jitter_vec[0]]], axis=0)
            # Minor axis jitter
            bounds = np.append(bounds, [[0, max_jitter_vec[1]]], axis=0)
            # Angle of jitter
            bounds = np.append(bounds, [[0, 3.14159]], axis=0)
        else:
            raise ValueError('Unvalid number of elements in max_jitter_vec: must be an array with 2 elements, given %d elements' %max_jitter_vec.shape[0] )
        # Set the flag
        do_jitter = True
    # Full return
    return bounds, selected_mode, do_jitter

def _rhodeg_2_zernike(rho_mode, deg_mode, max_deg_mode):
    # Convert the angle in radians
    alpha = np.deg2rad(deg_mode)
    # find the m of zernike notation  
    m = 360 // max_deg_mode
    # Convert into the TIPTOP friendly notation
    Za = rho_mode * np.cos(m * alpha)
    Zb = rho_mode * np.sin(m * alpha)
    # return this values
    return Za, Zb

def _zernike_2_rhodeg(Za, Zb, max_deg_mode):
    # find the kypotenusa, ie rho = sqrt(Za^2 + Zb^2)
    rho_mode = np.hypot(Za, Zb)
    # Check if it null
    if rho_mode == 0:
        return 0.0, 0.0
    # Define the m of the zernike mode
    m = 360 // max_deg_mode
    # Other wise compute the arctan with the correct sign
    alpha = np.arctan2(Zb, Za) / m
    alpha = alpha % (2 * np.pi / m)
    # Convert angle into deg
    deg_mode = np.rad2deg(alpha)
    # Return the values
    return rho_mode, deg_mode


def _unpack_param(xv, cn2hei=None, selected_mode=None, do_jitter=False, zCoefStaticOn=None, jitter=None):
    param_per_layer = 3
    # Compute the number of layers
    if cn2hei is None:
        n_layer= int(xv.shape[0]/param_per_layer)
    else:
        cn2hei = np.array(cn2hei)
        n_layer = cn2hei.shape[0]
    # Check if xv has the correct dimension
    xv = np.array(xv).copy()
    if xv.shape[0] < n_layer*param_per_layer:
        raise ValueError('xv must has at least %d elements, has given %d' %(n_layer*param_per_layer, xv.shape[0]))
    # Create empty array to collect parameters
    cn2wei = np.empty(n_layer)
    winvel = np.empty(n_layer)
    windir = np.empty(n_layer)
    # Cast first element as seeing
    seeing = xv[0]
    # Cast other parameters but last layer
    for ii in range(n_layer-1):
        # Cn2 weight for multipole level without constraints
        if ii == 0:
            cn2wei[ii] = xv[1]
        else:
            cn2wei[ii] = (1-np.sum(cn2wei[:ii])) * xv[1+param_per_layer*ii]
        # Wind velocity and direction
        winvel[ii] = xv[2+param_per_layer*ii]
        windir[ii] = xv[3+param_per_layer*ii]
    # Find last layer Cn2 weight as diffrence from the previous
    cn2wei[-1] = 1-np.sum(cn2wei[:-1])
    # Cast last elements
    winvel[-1] = xv[n_layer*param_per_layer-2]
    windir[-1] = xv[n_layer*param_per_layer-1]
    # Create dictionary
    parameters_dic = {
        'seeing' : seeing,
        'cn2wei' : cn2wei,
        'winvel' : winvel,
        'windir' : windir
    }
    # Add the Height of layers
    if not cn2hei is None:
        parameters_dic['cn2hei'] = cn2hei
    # Add the Non Common Path Abberation
    if xv.shape[0] > n_layer*param_per_layer:
        if do_jitter:
            # extract the information on jitter
            parameters_dic['jitter'] = xv[-3:]
            # Cut the xv for just the NCPA modes values
            mode_param = xv[n_layer*param_per_layer:-3]
        else:
            # Cut the xv for just the NCPA modes values
            mode_param = xv[n_layer*param_per_layer:]
        # If is not used the selection of mode, but is just passed an array of amplitudes
        if selected_mode is None:
            parameters_dic['zCoefStaticOn'] = mode_param
        # The case with the selection mode behaviour, that is used in the estimators
        else:
            # Define the vector that contains the rms distance for each Null coefficient
            ncpa_vec = []
            # The index in the mode_param array: the mode with staus False, will be not used
            idx = 0
            # See if the mode is active 
            for ii, mode_status in enumerate(selected_mode):
                if mode_status:
                    # Get the amplitude of the mode
                    rho_mode = mode_param[idx]
                    # Set the counter to the next value in mode_param
                    idx += 1
                    # Check if there is an angle that describe the mode
                    if MODE_DEG[ii] > 0:
                        # Get the angle
                        deg_mode = mode_param[idx]
                        # Set the counter to the next element in mode_param
                        idx += 1
                        # Transform the values into the TIPTOP notation
                        Za, Zb = _rhodeg_2_zernike(rho_mode, deg_mode, MODE_DEG[ii])
                        # Add this coefficients to the arrays
                        ncpa_vec.append(Za)
                        ncpa_vec.append(Zb)
                    # If the mode is just described with ONE amplitude, just add it
                    else:
                        ncpa_vec.append(rho_mode)
                # If it is not active just add a zero
                else:
                    ncpa_vec.append(0)
                    # If it requires an angle, ie has to Zs for the same mode, add another zero
                    if MODE_DEG[ii] > 0:
                        ncpa_vec.append(0)
            # Save values
            parameters_dic['zCoefStaticOn'] = np.array(ncpa_vec)
    # Bypass all the NCPA setup
    if not zCoefStaticOn is None:
        parameters_dic['zCoefStaticOn'] = np.array(zCoefStaticOn).copy()
    # Bypass the jitter setup
    if not jitter is None:
        jitter = np.array(jitter).copy().ravel()
        # Check the array and cast
        if jitter.shape[0] == 3:
            if jitter[0] >= jitter[1]:
                parameters_dic['jitter'] = jitter
            else:
                raise ValueError('In jitter vector, first element must be grater or equan than the second')
        else:
            raise ValueError('The vector jitter must be a 1D array with 3 elemnts')
    # Retrun the dictionary
    return parameters_dic


def _pack_param(param_dic):
    # Compute the number of layers
    n_layer = param_dic['cn2wei'].shape[0]
    n_eleme = n_layer * 3
    # Extract the ncpa section
    if 'zCoefStaticOn' in param_dic:
        has_ncpa = True
        ncpa_coef = param_dic['zCoefStaticOn']
        ncpa_xv = []
        sele_mode = []
        mode_idx = 0 
        ii = 0
        while ii < ncpa_coef.shape[0]:
            if MODE_DEG[mode_idx] > 0:
                if not (ncpa_coef[ii] == 0 and ncpa_coef[ii+1] == 0):
                    rho_mode, deg_mode = _zernike_2_rhodeg(ncpa_coef[ii], ncpa_coef[ii+1], MODE_DEG[mode_idx])
                    ncpa_xv.append(rho_mode)
                    ncpa_xv.append(deg_mode)
                    sele_mode.append(True)
                else:
                    sele_mode.append(False)
                ii = ii + 1  # this skip now actually works
            else:
                if not ncpa_coef[ii] == 0:
                    ncpa_xv.append(ncpa_coef[ii])
                    sele_mode.append(True)
                else:
                    sele_mode.append(False)
            mode_idx = mode_idx + 1
            ii = ii + 1
        ncpa_xv = np.asarray(ncpa_xv)
        n_eleme = n_eleme + ncpa_xv.shape[0]
    else:
        sele_mode = None
        has_ncpa = False
    # Check if has the jitter
    if 'jitter' in param_dic:
        n_eleme = n_eleme + 3
        has_jitter = True
    else:
        has_jitter = False
    # Create the emptry array
    xv = np.full(n_eleme, np.nan)
    # Seeing
    xv[0] = param_dic['seeing']
    # Ground layers
    for ii in range(n_layer-1):
        xv[1+3*ii] = param_dic['cn2wei'][ii]
        xv[2+3*ii] = param_dic['winvel'][ii]
        xv[3+3*ii] = param_dic['windir'][ii]
    # Higher layers
    xv[1+3*(n_layer-1)] = param_dic['winvel'][-1]
    xv[2+3*(n_layer-1)] = param_dic['windir'][-1]
    # NCPA
    if has_ncpa:
        xv[3*n_layer:-3] = ncpa_xv 
    # jitter
    if has_jitter:
        xv[-3:] = param_dic['jitter']
    # return
    return xv, sele_mode, has_jitter

def _check_bounds(bounds):
    # Unpack the boundries
    bound_min = bounds[:, 0]
    bound_max = bounds[:, 1]
        # Check that bounds is made by couple
    if not bounds.shape[1] == 2:
        raise ValueError("Invalid bounds shape, must be (N,2) has been given (N,%d)" %bounds.shape[1])
    # Check that any bound-couple has null width
    if np.any(bound_min == bound_max):
        raise ValueError("Invalid bounds: min == max")
    # Check that any buond-couple is correctly oriented
    if np.any(bound_max <= bound_min):
        raise ValueError("Invalid bounds: max must be > min")
    # If the boundries are correct return True, other wise an error has been raised
    return True

def _unwrap_bound(xv_wrap_original, bounds_original):
    # Copy and shape the array
    xv_wrap = np.asarray(xv_wrap_original, copy=True)
    bounds = np.asarray(bounds_original, copy=True)
    # Unpack bounderies
    bound_min = bounds[:, 0]
    bound_max = bounds[:, 1]
    # Check that xv and bounds has the same lenght
    if xv_wrap.ndim == 1:
        if not xv_wrap.shape[0] == bounds.shape[0]:
            raise ValueError("Invalid shape: xv_wrap has length %d, while bounds has shape %d" %(xv_wrap.shape[0], bounds.shape[0]))
    elif xv_wrap.ndim == 2:
        if not xv_wrap.shape[1] == bounds.shape[0]:
            raise ValueError("Invalid shape: xv_wrap has %d variables, while bounds has shape %d" %(xv_wrap.shape[1], bounds.shape[0]))
    else:
        raise ValueError("Invalid shape: xv_wrap must be 1D or 2D array")
    # Check that all the elements lay in the intervall
    if np.any((xv_wrap < -1) | (xv_wrap > +1)):
        raise ValueError("Some values are outside their bounds")
    # Make the propper trasformation
    xv = bound_min + (xv_wrap + 1) * (bound_max - bound_min) / 2
    # Return the unwraped parameters
    return xv

def _wrap_bound(xv_original, bounds_original):
    # Copy and shape the array
    xv = np.asarray(xv_original, copy=True)
    bounds = np.asarray(bounds_original, copy=True)
    # Unpack bounderies
    bound_min = bounds[:, 0]
    bound_max = bounds[:, 1]
     # Check that xv and bounds has the same lenght
    if xv.ndim == 1:
        if not xv.shape[0] == bounds.shape[0]:
            raise ValueError("Invalid shape: xv has length %d, while bounds has shape %d" %(xv.shape[0], bounds.shape[0]))
    elif xv.ndim == 2:
        if not xv.shape[1] == bounds.shape[0]:
            raise ValueError("Invalid shape: xv has %d variables, while bounds has shape %d" %(xv.shape[1], bounds.shape[0]))
    else:
        raise ValueError("Invalid shape: xv must be 1D or 2D array")
    # Check that all the elements lay in the intervall
    if np.any((xv < bound_min) | (xv > bound_max)):
        raise ValueError("Some values are outside their bounds")
    # Make the propper trasformation
    xv_wrap = 2 * (xv - bound_min) / (bound_max - bound_min) - 1
    # Return the wraped parametres
    return xv_wrap

