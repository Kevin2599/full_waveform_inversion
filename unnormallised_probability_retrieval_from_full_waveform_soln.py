#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to get unnormallised probability from a full waveform solution.

# Input variables:
# MT_inversion_result_mat_file_path - Path to full waveform inversion result


# Output variables:
# Plots saved to file.

# Created by Tom Hudson, 25th April 2018

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import obspy
import scipy.io as sio # For importing .mat MT solution data
import os,sys
import glob
import pickle
from full_waveform_inversion import * #forward_model, load_input_data, variance_reduction_OLD, variance_reduction, cross_corr_comparison, cross_corr_comparison_shift_allowed, pearson_correlation_comparison


# ------------------- Define functions -------------------
def load_MT_dict_from_file(matlab_data_filename):
    # If output from MTFIT:
    if matlab_data_filename[-3:] == "mat":
        data=sio.loadmat(matlab_data_filename)
        i=0
        while True:
            try:
                # Load data UID from matlab file:
                if data['Events'][0].dtype.descr[i][0] == 'UID':
                    uid=data['Events'][0][0][i][0]
                if data['Events'][0].dtype.descr[i][0] == 'Probability':
                    MTp=data['Events'][0][0][i][0] # stored as a n length vector, the probability
                if data['Events'][0].dtype.descr[i][0] == 'MTSpace':
                    MTs=data['Events'][0][0][i] # stored as a 6 by n array (6 as 6 moment tensor components)
                i+=1
            except IndexError:
                break
    
        try:
            stations = data['Stations']
        except KeyError:
            stations = []
    # Else if output from full waveform inversion:
    elif matlab_data_filename[-3:] == "pkl":
        FW_dict = pickle.load(open(matlab_data_filename,"rb"))
        uid = FW_dict["uid"]
        MTp = FW_dict["MTp"]
        MTs = FW_dict["MTs"]
        stations = FW_dict["stations"]
        
    else:
        print "Cannot recognise input filename."
        
    return uid, MTp, MTs, stations
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def load_MT_waveforms_dict_from_file(waveforms_data_filename):
    """Function to read waveforms dict output from full_waveform_inversion."""
    wfs_dict = pickle.load(open(waveforms_data_filename, "rb"))
    return wfs_dict

def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT
      
def plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=[]):
    """Function to get the probability distribution based on all samples for % DC vs. single force (which is the final value in MTs). Input is 10xn array of moment tensor samples and a length n array of their associated probability. Output is results plotted and shown to display or saved to file."""
    
    # Setup arrays to store data:
    percentage_DC_all_solns_bins = np.arange(0.,101.,1.)
    probability_percentage_DC_all_solns_bins = np.zeros(len(percentage_DC_all_solns_bins), dtype=float)
    probability_percentage_SF_all_solns_bins = np.zeros(len(percentage_DC_all_solns_bins), dtype=float)
    
    # Loop over MTs:
    for i in range(len(MTs[0,:])):
        MT_prob_current = MTp[i]
        if not MT_prob_current == 0:
            # Get frac DC and frac crack from CDC decomposition:
            frac_DC = MTs[9,i] # Last value from the inversion
            frac_SF = 1. - MTs[9,i]
            # And append probability to bin:
            # For DC:
            val, val_idx = find_nearest(percentage_DC_all_solns_bins,frac_DC*100.)
            probability_percentage_DC_all_solns_bins[val_idx] += MTp[i] # Append probability of % DC to bin
            # And for single force:
            val, val_idx = find_nearest(percentage_DC_all_solns_bins,frac_SF*100.)
            probability_percentage_SF_all_solns_bins[val_idx] += MTp[i] # Append probability of % single force to bin
        
    # And plot results:
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    plt.plot(percentage_DC_all_solns_bins[:], probability_percentage_DC_all_solns_bins[:], c='#D94411')
    ax1.set_xlabel("Percentage DC")
    ax1.set_xlim((0,100))
    ax1.set_ylim((0.,np.max(probability_percentage_DC_all_solns_bins[:])*1.05))
    ###plt.plot(percentage_DC_all_solns_bins, probability_percentage_DC_all_solns_bins, c='k')
    ax2 = ax1.twiny()
    ax2.set_xlim((0,100))
    ax2.set_xlabel("Percentage single force")
    plt.gca().invert_xaxis()
    #plt.plot(percentage_DC_all_solns_bins[:], probability_percentage_crack_all_solns_bins[:], c='#309BD8')
    ax1.set_ylabel("Probability")
    # And do some presentation formatting:
    ax1.tick_params(labelright=True)
    ax1.tick_params(right = 'on')
    ax1.axvline(x=50,ls="--",c="#CDE64E")
    ax1.axvline(x=25,ls="--",c="#A5B16B")
    ax1.axvline(x=75,ls="--",c="#A5B16B")
    #plt.legend()
    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
        print "Saving plot to file:", figure_filename
    else:
        plt.show()

def compare_synth_solution_to_real_data_summary_func(synth_waveform_curr_sample, real_data_array, comparison_metric, perform_normallised_waveform_inversion=True):
    """Function to compare solution synth data to real data."""
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
            elif comparison_metric == "CC-shift":
                # And get cross-correlation value, with shift allowed:
                similarity_ind_stat_comps[k] = cross_corr_comparison_shift_allowed(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:], max_samples_shift_limit=5)
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
            elif comparison_metric == "CC-shift":
                # And get cross-correlation value, with shift allowed:
                similarity_ind_stat_comps[k] = cross_corr_comparison_shift_allowed(real_data_array[k,:], synth_waveform_curr_sample[k,:], max_samples_shift_limit=5)
    # And get equal weighted similarity value:
    similarity_curr_sample = np.average(similarity_ind_stat_comps)
    
    return similarity_curr_sample
        
def get_unnormallised_prob_for_specific_soln(real_data_array, green_func_array, MT_specific_soln, comparison_metric, perform_normallised_waveform_inversion=True):
    """Function to get unnormallised probability for a specific solution, based on a particular comparison type."""
    # Forward model to get specific synth waveforms:
    synth_forward_model_result_array = forward_model(green_func_array, MT_specific_soln)
    
    # And compare to specific data:
    # Get similarity:
    similarity_curr_sample = compare_synth_solution_to_real_data_summary_func(synth_forward_model_result_array, real_data_array, comparison_metric, perform_normallised_waveform_inversion)
    prob_specific_soln = similarity_curr_sample
    
    return prob_specific_soln
    
# ------------------- Main script for running -------------------
if __name__ == "__main__":
    
    # Plot for inversion:
    print "Getting unnormallised probability for a particular sample"
    
    # Specify information needed:
    # Specify event and inversion type:
    inversion_type = "DC_single_force_no_coupling" # can be: full_mt, DC, single_force, DC_single_force_couple or DC_single_force_no_coupling
    event_uid = "20171222022435216400" # Event uid (numbers in FW inversion filename)
    datadir_FW_outputs = "./python_FW_outputs" # Data directory for FW inversion outputs
    datadir_greens_functions = '/Users/tomhudson/Python/obspy_scripts/fk/MATLAB_inversion_scripts/test_data/output_data_for_inversion_MT_and_single_force'
    real_data_fnames = ['real_data_RA51_z.txt', 'real_data_RA52_z.txt', 'real_data_RA53_z.txt'] # List of real waveform data files within datadir corresponding to each station (i.e. length is number of stations to invert for)
    MT_green_func_fnames = ['green_func_array_MT_RA51_z.txt', 'green_func_array_MT_RA52_z.txt', 'green_func_array_MT_RA53_z.txt'] # List of Green's functions data files (generated using fk code) within datadir corresponding to each station (i.e. length is number of stations to invert for)
    single_force_green_func_fnames = ['green_func_array_single_force_RA51_z.txt', 'green_func_array_single_force_RA52_z.txt', 'green_func_array_single_force_RA53_z.txt'] # List of Green's functions data files (generated using fk code) within datadir corresponding to each station (i.e. length is number of stations to invert for)
    manual_indices_time_shift = [2,1,0]
    comparison_metric = "CC" # Options are VR (variation reduction), CC (cross-correlation of static signal), CC-shift (cross-correlation of signal with shift allowed), or PCC (Pearson correlation coeficient) (Note: CC is the most stable, as range is naturally from 0-1, rather than -1 to 1)
    perform_normallised_waveform_inversion = True # Boolean - If True, performs normallised waveform inversion, whereby each synthetic and real waveform is normallised before comparision. Effectively removes overall amplitude from inversion if True. Should use True if using VR comparison method.
    
    
    # Get greens function data:
    real_data_array, green_func_array = get_overall_real_and_green_func_data(datadir, real_data_fnames, MT_green_func_fnames, single_force_green_func_fnames, inversion_type, manual_indices_time_shift)
    
    # Get inversion result data:
    # Get inversion filenames:
    MT_data_filename = datadir_FW_outputs+"/"+event_uid+"_FW_"+inversion_type+".pkl" #"./python_FW_outputs/20171222022435216400_FW_DC.pkl"
    MT_waveforms_data_filename = datadir_FW_outputs+"/"+event_uid+"_FW_"+inversion_type+".wfs"  #"./python_FW_outputs/20171222022435216400_FW_DC.wfs"
    print "Processing data for:", MT_data_filename
    # Import MT data and associated waveforms:
    uid, MTp, MTs, stations = load_MT_dict_from_file(MT_data_filename)
    wfs_dict = load_MT_waveforms_dict_from_file(MT_waveforms_data_filename)
    
    # Get most likely solution:
    index_MT_max_prob = np.argmax(MTp) # Index of most likely MT solution
    MT_max_prob = MTs[:,index_MT_max_prob]
    # And deal with if MT contains probability:
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
        MT_max_prob = MT_max_prob[:-1]
    
    # And get prob that model MT matches real data:
    prob_specific_soln = get_unnormallised_prob_for_specific_soln(real_data_array, green_func_array, MT_max_prob, comparison_metric, perform_normallised_waveform_inversion)
    print "Similarity of most likely sample:", prob_specific_soln
    
    print "Finished processing unconstrained inversion data for:", MT_data_filename
    
    print "Finished"
    
    
    
    





