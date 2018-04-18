#!python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to undertake full waveform inversion for a source. Can invert for DC, full unconstrained MT, DC-crack or single force sources.

# Input variables:


# Output variables:
# Plots saved to file.
# Data in dictionary (of MTFIT format)

# Created by Tom Hudson, 28th March 2018

# Notes:
# Currently performs variance reduction on normalised data (as there is an issue with amplitude scaling when using DC events)
# Units of greens functions produced by fk are:
#   10^-20 cm/(dyne cm) - for DC and explosive sources
#   10^-15 cm/dyne - for single force
#   Therefore DC/explosive sources have displacements that are 10^3 times smaller - this correction factor is applied when importing greens functions
#   Also, convert to SI units, so system isn't in cm and dyne but in metres and newtons. Conversion factor is 10^5 (for 1/dyne) x 100 (for cm) = 10^7
#   (See fk readme)


# Notes to do:
# Get output to output in suitable format for using my plotting scripts (like MTINV output)
# Check that sampling is unbiased - It is currently non biased with regards to points on a sphere, but is baised with regards to sampling of Lune coords

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh # For calculating eigenvalues and eigenvectors of symetric (Hermitian) matrices
import scipy.signal as signal # For cross-correlation calculations
import os,sys
from obspy import UTCDateTime
import pickle
import random # For entirely random number generation

# Specify parameters:
datadir = '/Users/tomhudson/Python/obspy_scripts/fk/MATLAB_inversion_scripts/test_data/output_data_for_inversion_MT_and_single_force'
outdir = "./python_FW_outputs"
real_data_fnames = ['real_data_RA51_z.txt', 'real_data_RA52_z.txt', 'real_data_RA53_z.txt'] # List of real waveform data files within datadir corresponding to each station (i.e. length is number of stations to invert for)
MT_green_func_fnames = ['green_func_array_MT_RA51_z.txt', 'green_func_array_MT_RA52_z.txt', 'green_func_array_MT_RA53_z.txt'] # List of Green's functions data files (generated using fk code) within datadir corresponding to each station (i.e. length is number of stations to invert for)
single_force_green_func_fnames = ['green_func_array_single_force_RA51_z.txt', 'green_func_array_single_force_RA52_z.txt', 'green_func_array_single_force_RA53_z.txt'] # List of Green's functions data files (generated using fk code) within datadir corresponding to each station (i.e. length is number of stations to invert for)
data_labels = ["RA51, Z", "RA52, Z", "RA53, Z"] # Format of these labels must be of the form "station_name, comp" with the comma
inversion_type = "DC_single_force_no_coupling" # Inversion type can be: full_mt, DC, single_force, DC_single_force_couple, or DC_single_force_no_coupling. (if single force, greens functions must be 3 components rather than 6)
perform_normallised_waveform_inversion = True # Boolean - If True, performs normallised waveform inversion, whereby each synthetic and real waveform is normallised before comparision. Effectively removes overall amplitude from inversion if True. Should use True if using VR comparison method.
num_samples = 10000 #1000000 # Number of samples to perform Monte Carlo over
comparison_metric = "CC" # Options are VR (variation reduction), CC (cross-correlation of static signal), or PCC (Pearson correlation coeficient) (Note: CC is the most stable, as range is naturally from 0-1, rather than -1 to 1)
synth_data_fnames = []
manual_indices_time_shift = [2,1,0]
nlloc_hyp_filename = "NLLoc_data/loc.run1.20171222.022435.grid0.loc.hyp" # Nonlinloc filename for saving event data to file in MTFIT format (for plotting, further analysis etc)
plot_switch = True # If True, will plot outputs to screen



# ------------------- Define various functions used in script -------------------
def load_input_data(datadir, real_data_fnames, green_func_fnames, manual_indices_time_shift=[]):
    """Function to load input data and output as arrays of real data and greens functions.
    Inputs: arrays containing filenames of files with real data (columns for P (L component) only at the moment) and greens functions data (For M_xx, M_yy, M_zz, M_xy, M_xz, M_yz), respectively. Optional input is manual_indices_time_shift (an array/list of manual integer index time shifts for each station).
    Outputs: Real data array of shape (t, n) where t is number of time data points and n is number of stations; greens functions array of shape (t, g_n) where g_n is the number of greens functions components."""
    # Set up data storage arrays:
    tmp_real_data = np.loadtxt(datadir+"/"+real_data_fnames[0],dtype=float)
    tmp_green_func_data = np.loadtxt(datadir+"/"+green_func_fnames[0],dtype=float)
    num_time_pts = len(tmp_real_data) # Number of time points
    num_green_func_comp = len(tmp_green_func_data[0,:]) # Number of greens functions components
    real_data_array = np.zeros((len(real_data_fnames), num_time_pts), dtype=float)
    green_func_array_raw = np.zeros((len(real_data_fnames), num_green_func_comp, num_time_pts), dtype=float)
    
    # Loop over files, saving real and greens functions data to arrays:
    for i in range(len(real_data_fnames)):
        real_data_array[i, :] = np.loadtxt(datadir+"/"+real_data_fnames[i],dtype=float)
        green_func_array_raw[i, :, :] = np.transpose(np.loadtxt(datadir+"/"+green_func_fnames[i],dtype=float))
    
    # Shift greens functions by manually specified amount in time (if specified):
    # (for allignment so that real data and greens functions are alligned)
    if not len(manual_indices_time_shift) == 0:
        green_func_array = np.zeros(np.shape(green_func_array_raw), dtype=float)
        for i in range(len(manual_indices_time_shift)):
            green_func_array[i,:,:] = np.roll(green_func_array_raw[i,:,:], manual_indices_time_shift[i], axis=1)
    
    return real_data_array, green_func_array


def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT
    
    
def get_six_MT_from_full_MT_array(full_MT):
    six_MT = np.array([full_MT[0,0], full_MT[1,1], full_MT[2,2], np.sqrt(2.)*full_MT[0,1], np.sqrt(2.)*full_MT[0,2], np.sqrt(2.)*full_MT[1,2]])
    return six_MT
    
def find_eigenvalues_from_sixMT(sixMT):
    """Function to find ordered eigenvalues given 6 moment tensor."""
    # Get full MT:
    MT_current = sixMT
    # And get full MT matrix:
    full_MT_current = get_full_MT_array(MT_current)
    # Find the eigenvalues for the MT solution and sort into descending order:
    w,v = eigh(full_MT_current) # Find eigenvalues and associated eigenvectors for the symetric (Hermitian) MT matrix (for eigenvalue w[i], eigenvector is v[:,i])
    full_MT_eigvals_sorted = np.sort(w)[::-1] # Sort eigenvalues into descending order
    # Calculate gamma and delta (lat and lon) from the eigenvalues:
    lambda1 = full_MT_eigvals_sorted[0]
    lambda2 = full_MT_eigvals_sorted[1]
    lambda3 = full_MT_eigvals_sorted[2]
    return lambda1, lambda2, lambda3
    

def rot_mt_by_theta_phi(full_MT, theta=np.pi, phi=np.pi):
    """Function to rotate moment tensor by angle theta and phi (rotation about Y and then Z axes). Theta and phi must be in radians."""
    rot_theta_matrix = np.vstack(([np.cos(theta), 0., np.sin(theta)],[0., 1., 0.],[-1.*np.sin(theta), 0., np.cos(theta)]))
    rot_phi_matrix = np.vstack(([np.cos(phi), -1.*np.sin(phi), 0.], [np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]))
    full_MT_first_rot = np.dot(rot_theta_matrix, np.dot(full_MT, np.transpose(rot_theta_matrix)))
    full_MT_second_rot = np.dot(rot_phi_matrix, np.dot(full_MT_first_rot, np.transpose(rot_phi_matrix)))
    return full_MT_second_rot
    
def rot_single_force_by_theta_phi(single_force_vector, theta=np.pi, phi=np.pi):
    """Function to rotate NED single force by angle theta and phi, consistent with same theta, phi for mt rotation (rotate about Y and then Z axes). Theta and phi must be in radians."""
    rot_theta_matrix = np.vstack(([np.cos(theta), 0., np.sin(theta)],[0., 1., 0.],[-1.*np.sin(theta), 0., np.cos(theta)]))
    rot_phi_matrix = np.vstack(([np.cos(phi), -1.*np.sin(phi), 0.], [np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]))
    single_force_vector_first_rot = np.dot(rot_theta_matrix, single_force_vector)
    single_force_vector_second_rot = np.dot(rot_phi_matrix, single_force_vector_first_rot)
    return single_force_vector_second_rot

def perform_inversion(real_data_array, green_func_array):
    """Function to perform inversion using real data and greens functions to obtain the moment tensor. (See Walter 2009 thesis, Appendix C for theoretical details).
    Inputs are: real_data_array - array of size (k,t_samples), containing real data to invert for; green_func_array - array of size (k, n_mt_comp, t_samples), containing the greens function data for the various mt components and each station (where k is the number of stations*station components to invert for, t_samples is the number of time samples, and n_mt_comp is the number of moment tensor components specficied in the greens functions array).
    Outputs are: M - tensor of length n_mt_comp, containing the moment tensor (or single force) inverted for."""
    # Perform the inversion:
    D = np.transpose(np.array([np.hstack(list(real_data_array[:]))])) # equivilent to matlab [real_data_array[0]; real_data_array[1]; real_data_array[2]]
    G =  np.transpose(np.vstack(np.hstack(list(green_func_array[:])))) # equivilent to matlab [green_func_array[0]; green_func_array[1]; green_func_array[2]]
    M, res, rank, sing_values_G = np.linalg.lstsq(G,D) # Equivilent to M = G\D; for G not square. If G is square, use linalg.solve(G,D)
    return M


def forward_model(green_func_array, M):
    """Function to forward model for a given set of greens functions and a specified moment tensor (or single force tensor).
    Inputs are: green_func_array - array of size (k, n_mt_comp, t_samples), containing the greens function data for the various mt components and each station; and M - tensor of length n_mt_comp, containing the moment tensor (or single force) to forward model for (where k is the number of stations*station components to invert for, t_samples is the number of time samples, and n_mt_comp is the number of moment tensor components specficied in the greens functions array)
    Outputs are: synth_forward_model_result_array - array of size (k, t_samples), containing synthetic waveforms for a given moment tensor."""
    # And get forward model synthetic waveform result:
    synth_forward_model_result_array = np.zeros(np.shape(green_func_array[:,0,:]), dtype=float)
    # Loop over signals:
    for i in range(len(green_func_array[:,0,0])):
        # Loop over components of greens function and MT solution (summing):
        for j in range(len(M)):
            synth_forward_model_result_array[i,:] += green_func_array[i,j,:]*M[j] # greens function for specific component over all time * moment tensor component
    return synth_forward_model_result_array


def plot_specific_forward_model_result(real_data_array, synth_forward_model_result_array, data_labels, plot_title="", perform_normallised_waveform_inversion=True):
    """Function to plot real data put in with specific forward model waveform result."""
    fig, axarr = plt.subplots(len(real_data_array[:,0]), sharex=True)
    for i in range(len(axarr)):
        if perform_normallised_waveform_inversion:
            axarr[i].plot(real_data_array[i,:]/np.max(np.absolute(real_data_array[i,:])),c='k', alpha=0.6) # Plot real data
            axarr[i].plot(synth_forward_model_result_array[i,:]/np.max(np.absolute(synth_forward_model_result_array[i,:])),c='r',linestyle="--", alpha=0.6) # Plot synth data
        else:
            axarr[i].plot(real_data_array[i,:],c='k', alpha=0.6) # Plot real data
            axarr[i].plot(synth_forward_model_result_array[i,:],c='r',linestyle="--", alpha=0.6) # Plot synth data
        axarr[i].set_title(data_labels[i])
    plt.suptitle(plot_title)
    plt.show()


def generate_random_MT():
    """Function to generate random moment tensor using normal distribution projected onto a 6-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised 6 MT."""
    # Generate 6 indepdendent normal deviates:
    six_MT_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float)
    # Normallise sample onto unit 6-sphere:
    six_MT_normalised = six_MT_unnormalised/(np.sum(six_MT_unnormalised**2)**-0.5) # As in Muller (1959)
    # And normallise so that moment tensor magnitude = 1:
    six_MT_normalised = six_MT_normalised/((np.sum(six_MT_normalised**2))**0.5)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    six_MT_normalised = np.reshape(six_MT_normalised, (6, 1))
    return six_MT_normalised
    
def generate_random_DC_MT():
    """Function to generate random DC constrained moment tensor using normal distribution projected onto a 3-sphere method (detailed in algorithm B2,B3, Pugh 2015). (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised 6 MT. DC component is derived from eigenvalues using CDC decomposition as in Tape and Tape 2013. DC moment tensor is specifed then rotated by random theta and phi to give random DC mt."""
    # Specify DC moment tensor to rotate:
    DC_MT_to_rot = np.vstack(([0.,0.,1.],[0.,0.,0.], [1.,0.,0.])) # DC moment tensor
    # Get a random sample 3-vector on a 3-unit sphere to use to calculate random theta and phi rotation angles:
    a_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float) # Generate 3 indepdendent normal deviates
    a_normalised = a_unnormalised/(np.sum(a_unnormalised**2)**-0.5) # Normallise sample onto unit 3-sphere - As in Muller (1959)
    # And normallise so that vector magnitude = 1:
    a_normalised = a_normalised/((np.sum(a_normalised**2))**0.5)
    x = a_normalised[0]
    y = a_normalised[1]
    z = a_normalised[2]
    theta = np.arccos(z)
    phi = np.arccos(x/np.sin(theta))
    # And rotate DC moment tensor by random 3D angle:
    random_DC_MT = rot_mt_by_theta_phi(DC_MT_to_rot, theta, phi)
    random_DC_six_MT = get_six_MT_from_full_MT_array(random_DC_MT)
    # And normallise so that moment tensor magnitude = 1:
    random_DC_six_MT_normalised = random_DC_six_MT/((np.sum(random_DC_six_MT**2))**0.5)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    random_DC_six_MT_normalised = np.reshape(random_DC_six_MT_normalised, (6, 1))
    return random_DC_six_MT_normalised
    
    
def generate_random_single_force_vector():
    """Function to generate random single force vector (F_x,F_y,F_z) using normal distribution projected onto a 3-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised single force 3-vector."""
    # Generate 3 indepdendent normal deviates:
    single_force_vector_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float)
    # Normallise sample onto unit 3-sphere:
    single_force_vector_normalised = single_force_vector_unnormalised/(np.sum(single_force_vector_unnormalised**2)**-0.5) # As in Muller (1959)
    # And normallise so that moment tensor magnitude = 1:
    single_force_vector_normalised = single_force_vector_normalised/((np.sum(single_force_vector_normalised**2))**0.5)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    single_force_vector_normalised = np.reshape(single_force_vector_normalised, (3, 1))
    return single_force_vector_normalised
    
def generate_random_DC_single_force_coupled_tensor():
    """Function to generate random DC-single-force coupled tensor (M_11,M_22,M_33,M_12,M_13_M_23,F_x,F_y,F_z) using normal distribution projected onto a 3-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised DC-single-force tensor of length 9 and the fraction amplitude of DC (0-1.).
    Note on coupling: Coupled so that single force in direction of slip vector of DC solution."""
    # 1. --- Specify DC moment tensor to rotate ---:
    DC_MT_to_rot = np.vstack(([0.,0.,1.],[0.,0.,0.], [1.,0.,0.])) # DC moment tensor
    # 2. --- Specify single force 3-vector in same direction as slip of DC solution ---:
    NED_single_force_vector_to_rotate = np.array([1.,0.,0.], dtype=float)
    # 3. --- Rotate DC moment tensor and single force by same random rotation on sphere:
    # 3.a. Get a random sample 3-vector on a 3-unit sphere to use to calculate random theta and phi rotation angles:
    a_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float) # Generate 3 indepdendent normal deviates
    a_normalised = a_unnormalised/(np.sum(a_unnormalised**2)**-0.5) # Normallise sample onto unit 3-sphere - As in Muller (1959)
    # And normallise so that vector magnitude = 1:
    a_normalised = a_normalised/((np.sum(a_normalised**2))**0.5)
    x = a_normalised[0]
    y = a_normalised[1]
    z = a_normalised[2]
    theta = np.arccos(z)
    phi = np.arccos(x/np.sin(theta))
    # 3.b. Rotate DC moment tensor by random 3D angle:
    random_DC_MT = rot_mt_by_theta_phi(DC_MT_to_rot, theta, phi)
    random_DC_six_MT = get_six_MT_from_full_MT_array(random_DC_MT)
    random_DC_six_MT_normalised = random_DC_six_MT/((np.sum(random_DC_six_MT**2))**0.5) # And normallise so that moment tensor magnitude = 1
    random_DC_six_MT_normalised = np.reshape(random_DC_six_MT_normalised, (6, 1)) # And set to correct dimensions (so matrix multiplication in forward model works correctly)
    # 3.c. Rotate Single force 3-vector by the same random 3D angle:
    random_coupled_NED_single_force = rot_single_force_by_theta_phi(NED_single_force_vector_to_rotate, theta, phi)
    random_coupled_single_force = np.array([random_coupled_NED_single_force[1], random_coupled_NED_single_force[0], random_coupled_NED_single_force[2]]) # Convert single force from NED coords to END coords (as that is what greens functions are in)
    random_coupled_single_force = np.reshape(random_coupled_single_force, (3, 1))
    # 4. --- Get random fraction amplitude DC and single force components (sum of amplitudes is to 1) ---:
    random_amp_frac = random.random() # random number between 0. and 1.
    random_DC_six_MT_normalised = random_DC_six_MT_normalised*random_amp_frac
    random_coupled_single_force = random_coupled_single_force*(1.-random_amp_frac)
    # 5. --- Finally combine to tensor of length 9 ---:
    random_DC_single_force_coupled_tensor = np.vstack((random_DC_six_MT_normalised, random_coupled_single_force))
    return random_DC_single_force_coupled_tensor, random_amp_frac

def generate_random_DC_single_force_uncoupled_tensor():
    """Function to generate random DC-single-force uncoupled tensor (M_11,M_22,M_33,M_12,M_13_M_23,F_x,F_y,F_z) using normal distribution projected onto a 3-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised DC-single-force tensor of length 9 and the fraction amplitude of DC (0-1.).
    Note on coupling: Uncoupled, i.e. single force and DC slip vector can be oriented in any direction with respect to one another."""
    # Generate random DC MT and single force:
    random_DC_MT_normallised = generate_random_DC_MT() # Generate a random DC sample
    random_single_force_normallised = generate_random_single_force_vector()
    # And split the amplitude of DC to single force randomly:
    random_amp_frac = random.random() # random number between 0. and 1.
    random_DC_MT_normallised = random_DC_MT_normallised*random_amp_frac
    random_single_force_normallised = random_single_force_normallised*(1.-random_amp_frac)
    # 5. --- Finally combine to tensor of length 9 ---:
    random_DC_single_force_uncoupled_tensor = np.vstack((random_DC_MT_normallised, random_single_force_normallised))
    return random_DC_single_force_uncoupled_tensor, random_amp_frac

def variance_reduction(data, synth):
    """Function to perform variance reduction of data and synthetic. Based on Eq. 2.1 in Walter 2009 thesis. Originally from Templeton and Dreger 2006.
    (When using all data in flattened array into this function, this is approximately equivilent to L2 norm (e.g. Song2011))."""
    ###VR = 1. - (np.sum(np.square(data-synth))/(len(data)*np.max(np.absolute(np.square(data-synth)))))
    VR = 1. - (np.sum(np.square(data-synth))/np.sum(np.square(data))) # Calculate variance reduction
    # And account for amplitude difference error (between synth and data) giving negative results:
    # Note: This is artificial! Can avoid by setting normalisation of data before input.
    if VR < 0.:
        VR = 0.
    return VR
    
def cross_corr_comparison(data, synth):
    """Function performing cross-correlation between long waveform data (data) and template.
    Performs normalized cross-correlation in fourier domain (since it is faster).
    Returns normallised correlation coefficients. See notes on cross-correlation search for more details on normalisation theory"""
    synth_normalised = (synth - np.mean(synth))/np.std(synth)/len(synth) # normalise and detrend
    std_data = np.std(data)    
    #correlation of unnormalized data and normalized template:
    f_corr = signal.correlate(data, synth_normalised, mode='valid', method='fft') # Note: Needs scipy version 19 or greater (mode = full or valid, use full if want to allow shift)
    ncc = np.true_divide(f_corr, std_data) #normalization of the cross correlation
    ncc_max = np.max(ncc) # get max cross-correlation coefficient (only if use full mode)
    if ncc_max<0.:
        ncc_max = 0.
    return ncc_max
    
# def cross_corr_comparison_shift_allowed(data, synth, max_samples_shift_limit=5):
#     """Function performing cross-correlation between long waveform data (data) and template.
#     Performs normalized cross-correlation in fourier domain (since it is faster).
#     Returns normallised correlation coefficients. See notes on cross-correlation search for more details on normalisation theory"""
#     synth_normalised = (synth - np.mean(synth))/np.std(synth)/len(synth) # normalise and detrend
#     std_data = np.std(data)
#     #correlation of unnormalized data and normalized template:
#     f_corr = signal.correlate(data, synth_normalised, mode='valid', method='fft') # Note: Needs scipy version 19 or greater (mode = full or valid, use full if want to allow shift)
#     ncc = np.true_divide(f_corr, std_data) #normalization of the cross correlation
#     ncc_max = np.max(ncc) # get max cross-correlation coefficient (only if use full mode)
#     if ncc_max<0.:
#         ncc_max = 0.
#     return ncc_max
    
def pearson_correlation_comparison(data, synth):
    """Function to compare similarity of data and synth using Pearson correlation coefficient. Returns number between 0. and 1. (as negatives are anti-correlation, set PCC<0 to 0)."""
    mean_data = np.average(data)
    mean_synth = np.average(synth)
    cov_data_synth = np.sum((data-mean_data)*(synth-mean_synth))/len(data)
    PCC = cov_data_synth/(np.std(data)*np.std(synth)) # Pearson correlation coefficient (-1 to 1, where 0 is no correlation, -1 is anti-correlation and 1 is correlation.)
    if PCC<0.:
        PCC = 0.
    return PCC
    
def perform_monte_carlo_sampled_waveform_inversion(real_data_array, green_func_array, num_samples=1000, M_amplitude=1.,inversion_type="full_mt",comparison_metric="CC",perform_normallised_waveform_inversion=True):
    """Function to use random Monte Carlo sampling of the moment tensor to derive a best fit for the moment tensor to the data.
    Notes: Currently does this using M_amplitude (as makes comparison of data realistic) (alternative could be to normalise real and synthetic data).
    Inversion type can be: full_mt, DC or single_force. If it is full_mt or DC, must give 6 greens functions in greeen_func_array. If it is a single force, must use single force greens functions (3).
    Comparison metrics can be: VR (variation reduction), CC (cross-correlation of static signal), or PCC (Pearson correlation coeficient)."""
    
    # 1. Set up data stores to write inversion results to:
    MTs = np.zeros((len(green_func_array[0,:,0]), num_samples), dtype=float)
    similarity_values_all_samples = np.zeros(num_samples, dtype=float)
    MTp = np.zeros(num_samples, dtype=float)
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
        MT_single_force_rel_amps = np.zeros(num_samples, dtype=float)
    
    # 2. Assign prior probabilities:
    # Note: Don't need to assign p_data as will find via marginalisation
    p_model = 1./num_samples # P(model) - Assume constant P(model)
    
    # 3. Loop over samples, checking how well a given MT sample synthetic wavefrom from the forward model compares to the real data:
    for i in range(num_samples):
        # 4. Generate synthetic waveform for current sample:
        if inversion_type=="full_mt":
            MT_curr_sample = generate_random_MT()*M_amplitude # Generate a random MT sample
        elif inversion_type=="DC":
            MT_curr_sample = generate_random_DC_MT()*M_amplitude # Generate a random DC sample
        elif inversion_type=="single_force":
            MT_curr_sample = generate_random_single_force_vector()*M_amplitude # Generate a random single force sample
        elif inversion_type == "DC_single_force_couple":
            MT_curr_sample, random_DC_to_single_force_amp_frac = generate_random_DC_single_force_coupled_tensor() # Generate a random DC-single-force coupled sample, with associated relative amplitude of DC to single force
            MT_curr_sample = MT_curr_sample*M_amplitude
        elif inversion_type == "DC_single_force_no_coupling":
            MT_curr_sample, random_DC_to_single_force_amp_frac = generate_random_DC_single_force_uncoupled_tensor()
            MT_curr_sample = MT_curr_sample*M_amplitude
        synth_waveform_curr_sample = forward_model(green_func_array, MT_curr_sample) # Note: Greens functions must be of similar amplitude units going into here...
        
        # 5. Compare real data to synthetic waveform (using variance reduction or other comparison metric), to assign probability that data matches current model:
        # Note: Do for all stations combined!
        # Note: Undertaken currently on normalised real and synthetic data!
        if perform_normallised_waveform_inversion:
            # Normalise:
            real_data_array_normalised = real_data_array.copy()
            synth_waveform_curr_sample_normalised = synth_waveform_curr_sample.copy()
            for j in range(len(real_data_array[:,0])):
                real_data_array_normalised[j,:] = real_data_array[j,:]/np.max(np.absolute(real_data_array[j,:]))
                synth_waveform_curr_sample_normalised[j,:] = synth_waveform_curr_sample[j,:]/np.max(np.absolute(synth_waveform_curr_sample[j,:]))
            # And do waveform comparison, via specified method:
            similarity_ind_stat_comps = np.zeros(len(real_data_array_normalised[:,0]), dtype=float)
            for k in range(len(similarity_ind_stat_comps)):
                if comparison_metric == "VR":
                    # get variance reduction:
                    similarity_ind_stat_comps[k] = variance_reduction(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "CC":
                    # And get cross-correlation value:
                    similarity_ind_stat_comps[k] = cross_corr_comparison(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "PCC":
                    # And get Pearson correlation coeficient:
                    similarity_ind_stat_comps[k] = pearson_correlation_comparison(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
        else:
            # Do waveform comparison, via specified method:
            similarity_ind_stat_comps = np.zeros(len(real_data_array[:,0]), dtype=float)
            for k in range(len(similarity_ind_stat_comps)):
                if comparison_metric == "VR":
                    # get variance reduction:
                    similarity_ind_stat_comps[k] = variance_reduction(real_data_array[k,:], synth_waveform_curr_sample[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "CC":
                    # And get cross-correlation value:
                    similarity_ind_stat_comps[k] = cross_corr_comparison(real_data_array[k,:], synth_waveform_curr_sample[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "PCC":
                    # And get Pearson correlation coeficient:
                    similarity_ind_stat_comps[k] = pearson_correlation_comparison(real_data_array[k,:], synth_waveform_curr_sample[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
        # And get equal weighted similarity value:
        similarity_curr_sample = np.average(similarity_ind_stat_comps)
        
        # 6. Append results to data store:
        MTs[:,i] = MT_curr_sample[:,0]
        similarity_values_all_samples[i] = similarity_curr_sample
        if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
            MT_single_force_rel_amps[i] = random_DC_to_single_force_amp_frac
        
        if i % 100000 == 0:
            print "Processed for",i,"samples out of",num_samples,"samples"
    
    # 7. Get P(model|data):
    p_data = np.sum(p_model*similarity_values_all_samples) # From marginalisation, P(data) = sum(P(model_i).P(data|model_i))
    MTp = similarity_values_all_samples*p_model/p_data

    # 8. Any final inversion specific data processing:    
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
        MTs = np.vstack((MTs, MT_single_force_rel_amps)) # For passing relative amplitude DC as well as MT and single force components
    
    return MTs, MTp
    
def get_event_uid_and_station_data_MTFIT_FORMAT_from_nonlinloc_hyp_file(nlloc_hyp_filename):
    """Function to get event uid and station data (station name, azimuth, takeoff angle, polarity) from nonlinloc hyp file. This data is required for writing to file for plotting like MTFIT data."""
    # Array shape is (num_stations, 4) in the order: station name, azimuth, takeoff angle, polarity
    
    # Get event UID:
    # Get event origin times:
    # Get event time from NLLoc file for basal icequake:
    os.system("grep 'GEOGRAPHIC' "+nlloc_hyp_filename+" > ./tmp_event_GEO_line.txt")
    GEO_line = np.loadtxt("./tmp_event_GEO_line.txt", dtype=str)
    event_origin_time = UTCDateTime(GEO_line[2]+GEO_line[3]+GEO_line[4]+GEO_line[5]+GEO_line[6]+GEO_line[7])
    # And remove temp files:
    os.system("rm ./tmp_*GEO_line.txt")
    uid = event_origin_time.strftime("%Y%m%d%H%M%S%f")
    
    # And get station arrival times and azimuth+takeoff angles for each phase, for event:
    os.system("awk '/PHASE ID/{f=1;next} /END_PHASE/{f=0} f' "+nlloc_hyp_filename+" > ./tmp_event_PHASE_lines.txt") # Get phase info and write to tmp file
    PHASE_lines = np.loadtxt("./tmp_event_PHASE_lines.txt", dtype=str) # And import phase lines as np str array
    arrival_times_dict = {} # Create empty dictionary to store data (with keys: event_origin_time, station_arrivals {station {station_P_arrival, station_S_arrival}}})
    arrival_times_dict['event_origin_time'] = event_origin_time
    arrival_times_dict['station_arrival_times'] = {}
    arrival_times_dict['azi_takeoff_angles'] = {}
    # Loop over stations:
    for i in range(len(PHASE_lines[:,0])):
        station = PHASE_lines[i, 0]
        station_current_phase_arrival = UTCDateTime(PHASE_lines[i,6]+PHASE_lines[i,7]+PHASE_lines[i,8])
        station_current_azimuth_event_to_sta = float(PHASE_lines[i,22])
        station_current_toa_event_to_sta = float(PHASE_lines[i,24])
        station_current_toa_sta_inclination = 180. - station_current_toa_event_to_sta
        # See if station entry exists, and if does, write arrival to array, otherwise, create entry and write data to file:
        # For station arrival times:
        try:
            arrival_times_dict['station_arrival_times'][station]
        except KeyError:
            # If entry didnt exist, create it and fill:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['station_arrival_times'][station] = {}
                arrival_times_dict['station_arrival_times'][station]["P"] = station_current_phase_arrival
            elif PHASE_lines[i, 4] == "S":
                arrival_times_dict['station_arrival_times'][station] = {}
                arrival_times_dict['station_arrival_times'][station]["S"] = station_current_phase_arrival
        # And if entry did exist:
        else:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['station_arrival_times'][station]["P"] = station_current_phase_arrival
            elif PHASE_lines[i, 4] == "S":
                arrival_times_dict['station_arrival_times'][station]["S"] = station_current_phase_arrival
        # And for azimuth and takeoff angle:
        try:
            arrival_times_dict['azi_takeoff_angles'][station]
        except KeyError:
            # If entry didnt exist, create it and fill:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['azi_takeoff_angles'][station] = {}
                arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"] = station_current_azimuth_event_to_sta
                arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"] = station_current_toa_sta_inclination
        # And if entry did exist:
        else:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"] = station_current_azimuth_event_to_sta
                arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"] = station_current_toa_sta_inclination  
    
    # And clean up:
    os.system("rm ./tmp*PHASE_lines.txt")
    
    # And create stations array:
    stations = []
    for i in range(len(arrival_times_dict['azi_takeoff_angles'])):
        station = arrival_times_dict['azi_takeoff_angles'].keys()[i]
        azi = arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"]
        toa = arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"]
        pol = 0 # Assign zero polarity, as not needed for full waveform
        stations.append([np.array([station], dtype=str), np.array([[azi]], dtype=float), np.array([[toa]], dtype=float), np.array([[pol]], dtype=int)])
    stations = np.array(stations) # HERE!!! (need to find out what type of object stations is!)
    
    return uid, stations
    
def save_to_MTFIT_style_file(MTs, MTp, nlloc_hyp_filename, inversion_type, outdir):
    """Function to save data to MTFIT style file, containing arrays of uid, MTs (array of 6xn for possible MT solutions), MTp (array of length n storing probabilities of each solution) and stations (station name, azimuth, takeoff angle, polarity (set to zero here)).
    Output is a pickled file containing a dictionary of uid, stations, MTs and MTp."""
    # Get uid and stations data:
    uid, stations = get_event_uid_and_station_data_MTFIT_FORMAT_from_nonlinloc_hyp_file(nlloc_hyp_filename)
    # Write all data to output dict:
    out_dict = {}
    out_dict["MTs"] = MTs
    out_dict["MTp"] = MTp
    out_dict["uid"] = uid
    out_dict["stations"] = stations
    # And save to file:
    out_fname = outdir+"/"+uid+"_FW_"+inversion_type+".pkl"
    print "Saving FW inversion to file:", out_fname
    pickle.dump(out_dict, open(out_fname, "wb"))
    
def save_specific_waveforms_to_file(real_data_array, synth_data_array, data_labels, nlloc_hyp_filename, inversion_type, outdir):
    """Function to save specific waveforms to dictionary format file."""
    # Put waveform data in dict format:
    out_wf_dict = {}
    for i in range(len(data_labels)):
        out_wf_dict[data_labels[i]] = {}
        out_wf_dict[data_labels[i]]["real_wf"] = real_data_array[i,:]
        out_wf_dict[data_labels[i]]["synth_wf"] = synth_data_array[i,:]
    # Get uid for filename:
    uid, stations = get_event_uid_and_station_data_MTFIT_FORMAT_from_nonlinloc_hyp_file(nlloc_hyp_filename)
    # And write to file:
    out_fname = outdir+"/"+uid+"_FW_"+inversion_type+".wfs"
    print "Saving FW inversion to file:", out_fname
    pickle.dump(out_wf_dict, open(out_fname, "wb"))

# ------------------- End of defining various functions used in script -------------------


# ------------------- Main script for running -------------------
if __name__ == "__main__":
    # Load input data into arrays:
    if inversion_type=="full_mt" or inversion_type=="DC":
        real_data_array, green_func_array = load_input_data(datadir, real_data_fnames, MT_green_func_fnames, manual_indices_time_shift)
        # correct for different units of single force to DC (see note in script header):
        green_func_array = green_func_array*(10**3)
    elif inversion_type=="single_force":
        real_data_array, green_func_array = load_input_data(datadir, real_data_fnames, single_force_green_func_fnames, manual_indices_time_shift)
    elif inversion_type=="DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
        real_data_array, MT_green_func_array = load_input_data(datadir, real_data_fnames, MT_green_func_fnames, manual_indices_time_shift)
        real_data_array, SF_green_func_array = load_input_data(datadir, real_data_fnames, single_force_green_func_fnames, manual_indices_time_shift)
        # correct for different units of single force to DC (see note in script header):
        MT_green_func_array = MT_green_func_array*(10**3)
        # And combine all greens functions:
        green_func_array = np.hstack((MT_green_func_array, SF_green_func_array))
    # And convert to SI units (see note in script header):
    green_func_array = green_func_array*(10**7)

    # Perform the inversion:
    M = perform_inversion(real_data_array, green_func_array)
    M_amplitude = ((np.sum(M**2))**0.5)

    # And get forward model synthetic waveform result:
    synth_forward_model_result_array = forward_model(green_func_array, M)

    # And plot the results:
    if plot_switch:
        plot_specific_forward_model_result(real_data_array, synth_forward_model_result_array, data_labels, plot_title="Initial theoretical inversion solution", perform_normallised_waveform_inversion=perform_normallised_waveform_inversion)
    
    # And do Monte Carlo random sampling to obtain PDF of moment tensor:
    MTs, MTp = perform_monte_carlo_sampled_waveform_inversion(real_data_array, green_func_array, num_samples, M_amplitude=M_amplitude,inversion_type=inversion_type, comparison_metric=comparison_metric, perform_normallised_waveform_inversion=perform_normallised_waveform_inversion)
        
    # And plot most likely solution:
    if plot_switch:
        if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
            synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
        else:
            synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:, np.where(MTp==np.max(MTp))[0][0]])
        plot_specific_forward_model_result(real_data_array, synth_forward_model_most_likely_result_array, data_labels, plot_title="Most likely Monte Carlo sampled solution", perform_normallised_waveform_inversion=perform_normallised_waveform_inversion)
        print "Most likely solution:", MTs[:,np.where(MTp==np.max(MTp))[0][0]]
    
    # And save data to MTFIT style file:
    save_to_MTFIT_style_file(MTs, MTp, nlloc_hyp_filename, inversion_type, outdir) # Saves pickled dictionary containing data from inversion
    # And save most likely solution and real data waveforms to file:
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
        synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
    else:
        synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:, np.where(MTp==np.max(MTp))[0][0]])
    save_specific_waveforms_to_file(real_data_array, synth_forward_model_most_likely_result_array, data_labels, nlloc_hyp_filename, inversion_type, outdir)

    print "Finished"










