#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to take unconstrained moment tensor or DC or single force inversion result (from full waveform inversion) and plot results.

# Input variables:
# MT_inversion_result_mat_file_path - Path to full waveform inversion result


# Output variables:
# Plots saved to file.

# Created by Tom Hudson, 9th April 2018

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
from numpy.linalg import eigh # For calculating eigenvalues and eigenvectors of symetric (Hermitian) matrices
import matplotlib
import matplotlib.pyplot as plt
import obspy
import scipy.io as sio # For importing .mat MT solution data
import scipy.optimize as opt # For curve fitting
import math # For plotting contours as line
import os,sys
import random
from matplotlib import path # For getting circle bounding path for MT plotting
from obspy.imaging.scripts.mopad import MomentTensor, BeachBall # For getting nodal planes for unconstrained moment tensors
from obspy.core.event.source import farfield # For calculating MT radiation patterns
from matplotlib.patches import Polygon, Circle # For plotting MT radiation patterns
import matplotlib.patches as mpatches # For adding patches for creating legends etc
from matplotlib.collections import PatchCollection # For plotting MT radiation patterns
import glob
import pickle
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import MTconvert # For doing various MT conversions/rotations etc (functions are originally part of MTINV package)


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
    
    

def load_MT_waveforms_dict_from_file(waveforms_data_filename):
    """Function to read waveforms dict output from full_waveform_inversion."""
    wfs_dict = pickle.load(open(waveforms_data_filename, "rb"))
    return wfs_dict

def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT

def convert_spherical_coords_to_cartesian_coords(r,theta,phi):
    """Function to take spherical coords and convert to cartesian coords. (theta between 0 and pi, phi between 0 and 2pi)"""
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D equal area projection."""
    X = x * np.sqrt(2/(1+z))
    Y = y * np.sqrt(2/(1+z))
    return X,Y

def create_and_plot_bounding_circle_and_path(ax):
    """Function to create and plot bounding circle for plotting MT solution. 
    Inputs are ax to plot on. Outputs are ax and bounding_circle_path (defining area contained by bounding circle)."""
    # Setup bounding circle:
    theta = np.ones(200)*np.pi/2
    phi = np.linspace(0.,2*np.pi,len(theta))
    r = np.ones(len(theta))
    x,y,z = convert_spherical_coords_to_cartesian_coords(r,theta,phi)
    X_bounding_circle,Y_bounding_circle = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
    ax.plot(Y_bounding_circle,X_bounding_circle, c="k")

    # And create bounding path from circle:
    path_coords = [] # list to store path coords
    for i in range(len(X_bounding_circle)):
        x_tmp = X_bounding_circle[i]
        y_tmp = Y_bounding_circle[i]
        path_coords.append((x_tmp,y_tmp))
    bounding_circle_path = path.Path(path_coords) # bounding path that can be used to find points
    #bounding_circle_path.contains_points([(.5, .5)])
    
    return ax, bounding_circle_path

def get_nodal_plane_xyz_coords(mt_in):
    """Function to get nodal plane coords given 6 MT in, in NED coords. Returns 2 arrays, describing the two nodal planes in terms of x,y,z coords on a unit sphere."""
    ned_mt = mt_in # 6 MT
    mopad_mt = MomentTensor(ned_mt,system='NED') # In north, east, down notation
    bb = BeachBall(mopad_mt, npoints=200)
    bb._setup_BB(unit_circle=True)
    neg_nodalline = bb._nodalline_negative # extract negative nodal plane coords (in 3D x,y,z)
    pos_nodalline = bb._nodalline_positive # extract positive nodal plane coords (in 3D x,y,z)
    return neg_nodalline, pos_nodalline

def plot_radiation_pattern_for_given_NED_DC_sixMT(ax, radiation_pattern_MT, bounding_circle_path, lower_upper_hemi_switch="lower", radiation_MT_phase="P", unconstrained_vs_DC_switch="unconstrained"):
    """Function to plot radiation pattern on axis ax, given 6 MT describing MT to plot radiation pattern for and other args.
    Outputs axis ax with radiation pattern plotted."""
    # Get MT to plot radiation pattern for:
    ned_mt = radiation_pattern_MT

    # Get spherical points to sample for radiation pattern:
    theta = np.linspace(0,np.pi,250)
    phi = np.linspace(0.,2*np.pi,len(theta))
    r = np.ones(len(theta))
    THETA,PHI = np.meshgrid(theta, phi)
    theta_flattened = THETA.flatten()
    phi_flattened = PHI.flatten()
    r_flattened = np.ones(len(theta_flattened))
    x,y,z = convert_spherical_coords_to_cartesian_coords(r_flattened,theta_flattened,phi_flattened)
    radiation_field_sample_pts = np.vstack((x,y,z))
    # get radiation pattern using farfield fcn:
    if radiation_MT_phase=="P":
        disp = farfield(ned_mt, radiation_field_sample_pts, type="P") # Gets radiation displacement vector
        disp_magn = np.sum(disp * radiation_field_sample_pts, axis=0) # Magnitude of displacement (alligned with radius)  ???np.sqrt???
    elif radiation_MT_phase=="S":
        disp = farfield(ned_mt, radiation_field_sample_pts, type="S") # Gets radiation displacement vector
        disp_magn = np.sqrt(np.sum(disp * disp, axis=0)) # Magnitude of displacement (perpendicular to radius)
    disp_magn /= np.max(np.abs(disp_magn)) # Normalised magnitude of displacemnet
    
    # If solution is DC, convert radiation pattern to 1/-1:
    if unconstrained_vs_DC_switch == "DC":
        disp_magn[disp_magn>=0.] = 1.
        disp_magn[disp_magn<0.] = -1.

    # And convert radiation pattern to 2D coords:
    ###X_radiaton_coords,Y_radiaton_coords = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(radiation_field_sample_pts[0],radiation_field_sample_pts[1],radiation_field_sample_pts[2])

    # Create 2D XY radial mesh coords (for plotting) and plot radiation pattern:
    theta_spacing = theta[1]-theta[0]
    phi_spacing = phi[1]-phi[0]
    patches = []
    # Plot majority of radiation points as polygons:
    for b in range(len(disp_magn)):
        # Convert coords if upper hemisphere plot rather than lower hemisphere:
        if lower_upper_hemi_switch=="upper":
            theta_flattened[b] = np.pi-theta_flattened[b]
            #phi_flattened[b] = phi_flattened[b]-np.pi
        # Get coords at half spacing around point:
        theta_tmp = np.array([theta_flattened[b]-(theta_spacing/2.), theta_flattened[b]+(theta_spacing/2.)])
        phi_tmp = np.array([phi_flattened[b]-(phi_spacing/2.), phi_flattened[b]+(phi_spacing/2.)])
        # And check that doesn't go outside boundaries:
        if theta_flattened[b] == 0. or theta_flattened[b] == np.pi:
            continue # ignore as outside boundaries
        if phi_flattened[b] == 0.:# or phi_flattened[b] == 2*np.pi:
            continue # ignore as outside boundaries
        THETA_tmp, PHI_tmp = np.meshgrid(theta_tmp, phi_tmp)
        R_tmp = np.ones(4,dtype=float)
        x,y,z = convert_spherical_coords_to_cartesian_coords(R_tmp,THETA_tmp.flatten(),PHI_tmp.flatten())
        X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
        # And plot (but ONLY if within bounding circle):
        if bounding_circle_path.contains_point((X[0],Y[0]), radius=0):
            poly_corner_coords = [(Y[0],X[0]), (Y[2],X[2]), (Y[3],X[3]), (Y[1],X[1])]
            if unconstrained_vs_DC_switch == "DC":
                polygon_curr = Polygon(poly_corner_coords, closed=True, facecolor=matplotlib.cm.binary(int(disp_magn[b]*256)), alpha=0.6)
            elif unconstrained_vs_DC_switch == "unconstrained":
                polygon_curr = Polygon(poly_corner_coords, closed=True, facecolor=matplotlib.cm.jet(int(disp_magn[b]*256)), alpha=0.6)
            ax.add_patch(polygon_curr)
    # Plot final point (theta,phi=0,0) (beginning point):
    if unconstrained_vs_DC_switch == "DC":
        centre_area = Circle([0.,0.], radius=theta_spacing/2., facecolor=matplotlib.cm.binary(int(disp_magn[0]*256)), alpha=0.6)
    elif unconstrained_vs_DC_switch == "unconstrained":
        centre_area = Circle([0.,0.], radius=theta_spacing/2., facecolor=matplotlib.cm.jet(int(disp_magn[0]*256)), alpha=0.6)
    ax.add_patch(centre_area)
    
    return ax

def plot_nodal_planes_for_given_NED_sixMT(ax, MT_for_nodal_planes, bounding_circle_path, lower_upper_hemi_switch="lower", alpha_nodal_planes=0.3):
    """Function for plotting nodal planes on axis ax, for given 6MT in NED format for nodal planes."""

    ned_mt = MT_for_nodal_planes

    # Get 3D nodal planes:
    plane_1_3D, plane_2_3D = get_nodal_plane_xyz_coords(ned_mt)
    # And switch vertical if neccessary:
    if lower_upper_hemi_switch=="upper":
        plane_1_3D[2,:] = -1*plane_1_3D[2,:] # as positive z is down, therefore down gives spherical projection
        plane_2_3D[2,:] = -1*plane_2_3D[2,:] # as positive z is down, therefore down gives spherical projection
    # And convert to 2D:
    X1,Y1 = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(plane_1_3D[0],plane_1_3D[1],plane_1_3D[2])
    X2,Y2 = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(plane_2_3D[0],plane_2_3D[1],plane_2_3D[2])

    # Get only data points within bounding circle:
    path_coords_plane_1 = [] # list to store path coords
    path_coords_plane_2 = [] # list to store path coords
    for j in range(len(X1)):
        path_coords_plane_1.append((X1[j],Y1[j]))
    for j in range(len(X2)):
        path_coords_plane_2.append((X2[j],Y2[j]))
    stop_plotting_switch = False # If true, would stop plotting on current axis (as can't)
    try:
        path_coords_plane_1_within_bounding_circle = np.vstack([p for p in path_coords_plane_1 if bounding_circle_path.contains_point(p, radius=0)])
        path_coords_plane_2_within_bounding_circle = np.vstack([p for p in path_coords_plane_2 if bounding_circle_path.contains_point(p, radius=0)])
        path_coords_plane_1_within_bounding_circle = np.vstack((path_coords_plane_1_within_bounding_circle, path_coords_plane_1_within_bounding_circle[0,:])) # To make no gaps
        path_coords_plane_2_within_bounding_circle = np.vstack((path_coords_plane_2_within_bounding_circle, path_coords_plane_2_within_bounding_circle[0,:])) # To make no gaps
        X1_within_bounding_circle = path_coords_plane_1_within_bounding_circle[:,0]
        Y1_within_bounding_circle = path_coords_plane_1_within_bounding_circle[:,1]
        X2_within_bounding_circle = path_coords_plane_2_within_bounding_circle[:,0]
        Y2_within_bounding_circle = path_coords_plane_2_within_bounding_circle[:,1]
    except ValueError:
        print "(Skipping current nodal plane solution as can't plot.)"
        stop_plotting_switch = True # Stops rest of script plotting on current axis

    # And plot 2D nodal planes:
    if not stop_plotting_switch:
        # Plot plane 1:
        for a in range(len(X1_within_bounding_circle)-1):
            if np.abs(Y1_within_bounding_circle[a]-Y1_within_bounding_circle[a+1])<0.25 and np.abs(X1_within_bounding_circle[a]-X1_within_bounding_circle[a+1])<0.25:
                ax.plot([Y1_within_bounding_circle[a], Y1_within_bounding_circle[a+1]],[X1_within_bounding_circle[a], X1_within_bounding_circle[a+1]], color="k", alpha=alpha_nodal_planes, marker="None")
            else:
                continue # And don't plot line between bounding circle intersections
        # And plot plane 2:
        for a in range(len(X2_within_bounding_circle)-1):
            if np.abs(Y2_within_bounding_circle[a]-Y2_within_bounding_circle[a+1])<0.25 and np.abs(X2_within_bounding_circle[a]-X2_within_bounding_circle[a+1])<0.25:
                ax.plot([Y2_within_bounding_circle[a], Y2_within_bounding_circle[a+1]],[X2_within_bounding_circle[a], X2_within_bounding_circle[a+1]], color="k", alpha=alpha_nodal_planes, marker="None")
            else:
                continue # And don't plot line between bounding circle intersections
    
    return ax

def plot_full_waveform_result_beachball_DC(MTs_to_plot, wfs_dict, radiation_pattern_MT=[], stations=[], lower_upper_hemi_switch="lower", figure_filename=[], num_MT_solutions_to_plot=20, unconstrained_vs_DC_switch="unconstrained"):
    """Function to plot full waveform DC constrained inversion result on sphere, then project into 2D using an equal area projection.
    Input MTs are np array of NED MTs in shape [6,n] where n is number of solutions. Also takes optional radiation_pattern_MT, which it will plot a radiation pattern for.
        Note: x and y coordinates switched for plotting to take from NE to EN
        Note: stations is a dictionary containing station info."""
    
    # Setup figure:
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111) #projection="3d")
    
    # Add some settings for main figure:
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    ax.set_xlim(-3.5,3.5)
    ax.set_ylim(-3.5,3.5)
    #plt.plot([-2.,2.],[0.,0.],c="k", alpha=0.5)
    #plt.plot([0.,0.],[-2.,2.],c="k", alpha=0.5)
    plt.axis('off')
    
    # Setup bounding circle and create bounding path from circle:
    ax, bounding_circle_path = create_and_plot_bounding_circle_and_path(ax)
    
    # Plot radiation pattern if provided with radiation pattern MT to plot:
    if not len(radiation_pattern_MT)==0:
        plot_radiation_pattern_for_given_NED_DC_sixMT(ax, radiation_pattern_MT, bounding_circle_path, lower_upper_hemi_switch=lower_upper_hemi_switch, radiation_MT_phase="P", unconstrained_vs_DC_switch="DC") # Plot radiation pattern
    
    # Plot MT nodal plane solutions:
    # Get samples to plot:
    # IF single sample, plot most likely (assocaited with radiation pattern):
    if num_MT_solutions_to_plot == 1:
        if not len(radiation_pattern_MT)==0:
            curr_MT_to_plot = radiation_pattern_MT
            ax = plot_nodal_planes_for_given_NED_sixMT(ax, curr_MT_to_plot, bounding_circle_path, lower_upper_hemi_switch, alpha_nodal_planes=0.3)
    # else if number of samples > 1:
    else:
        if len(MTs_to_plot[0,:]) > num_MT_solutions_to_plot:
            sample_indices = random.sample(range(len(MTs_to_plot[0,:])),num_MT_solutions_to_plot) # Get random sample of MT solutions to plot
        else:
            sample_indices = range(len(MTs_to_plot[0,:]))
        # Loop over MT solutions, plotting nodal planes:
        for i in sample_indices:
            # Get current mt:
            curr_MT_to_plot = MTs_to_plot[:,i]
            # And try to plot current MT nodal planes:
            print "Attempted to plot solution", i
            ax = plot_nodal_planes_for_given_NED_sixMT(ax, curr_MT_to_plot, bounding_circle_path, lower_upper_hemi_switch, alpha_nodal_planes=0.3)
            
    # Plot stations (if provided):
    if not len(stations) == 0:
        # Loop over stations:
        for station in stations:
            station_name = station[0][0]
            # Get params for station:
            # If from MTFIT analysis:
            if str(type(station[1][0][0])) == "<type 'numpy.float64'>":
                azi=(station[1][0][0]/360.)*2.*np.pi + np.pi
                toa=(station[2][0][0]/360.)*2.*np.pi
                polarity = station[3][0][0]
            # Else if from python FW inversion:
            else:
                azi=(float(station[1][0])/360.)*2.*np.pi + np.pi
                toa=(float(station[2][0])/360.)*2.*np.pi
                polarity = int(station[3][0])
            # And get 3D coordinates for station (and find on 2D projection):
            theta = np.pi - toa # as +ve Z = down
            phi = azi
            if lower_upper_hemi_switch=="upper":
                theta = np.pi-theta
                phi = phi-np.pi
            if theta>np.pi/2.:
                theta = theta - np.pi
                phi=phi+np.pi
            r = 1.0 # as on surface of focal sphere
            x,y,z = convert_spherical_coords_to_cartesian_coords(r, theta, phi)
            X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
            # And plot based on polarity:
            if polarity == 1:
                ax.scatter(Y,X,c="r",marker="^",s=30,alpha=1.0, zorder=100)
            elif polarity == -1:
                ax.scatter(Y,X,c="b",marker="v",s=30,alpha=1.0, zorder=100)
            elif polarity == 0:
                ax.scatter(Y,X,c="#267388",marker='o',s=30,alpha=1.0, zorder=100)
            # And plot station name:
            plt.sca(ax) # Specify axis to work on
            plt.text(Y,X,station_name,color="k", fontsize=10, horizontalalignment="left", verticalalignment='top',alpha=1.0, zorder=100)
            
            # And plot waveforms (real and synthetic):
            # Get current real and synthetic waveforms:
            for wfs_dict_station_idx in range(len(wfs_dict.keys())):
                if wfs_dict.keys()[wfs_dict_station_idx].split(",")[0] == station_name:
                    real_wf_current_stat = wfs_dict[wfs_dict.keys()[wfs_dict_station_idx]]['real_wf']
                    synth_wf_current_stat = wfs_dict[wfs_dict.keys()[wfs_dict_station_idx]]['synth_wf']
            # Get coords to plot waveform at:
            r = 2.0 # as want to plot waveforms beyond extent of focal sphere
            theta = np.pi/2. # Set theta to pi/2 as want to just plot waveforms in horizontal plane
            x,y,z = convert_spherical_coords_to_cartesian_coords(r, theta, phi)
            X_waveform_loc, Y_waveform_loc = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
            data_xy_coords = (Y_waveform_loc, X_waveform_loc)
            disp_coords = ax.transData.transform(data_xy_coords) # And transform data coords into display coords
            fig_inv = fig.transFigure.inverted() # Create inverted figure transformation
            fig_coords = fig_inv.transform((disp_coords[0],disp_coords[1])) # Transform display coords into figure coords (for adding axis)
            # Plot line to waveform:
            ax.plot([Y,Y_waveform_loc],[X,X_waveform_loc],c='k',alpha=0.6)
            # And plot waveform:
            left, bottom, width, height = [fig_coords[0]-0.15, fig_coords[1]-0.1, 0.3, 0.2]
            inset_ax_tmp = fig.add_axes([left, bottom, width, height])
            #inset_ax1 = inset_axes(ax,width="10%",height="5%",bbox_to_anchor=(0.2,0.4))
            inset_ax_tmp.plot(real_wf_current_stat,c='k', alpha=0.6) # Plot real data
            inset_ax_tmp.plot(synth_wf_current_stat,c='#E83313',linestyle="--", alpha=0.6) # Plot synth data
            plt.axis('off')
            
            

    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
    else:
        plt.show()
        
        
# ------------------- Main script for running -------------------
if __name__ == "__main__":
    
    # Plot for inversion:
    print "Plotting data for inversion"

    # Specify MT data dir (containing MTINV solutions):
    #MT_data_filenames = glob.glob("./python_FW_outputs/*FW_DC.pkl")
    MT_data_filename = "./python_FW_outputs/20171222022435216400_FW_DC.pkl"
    MT_waveforms_data_filename = "./python_FW_outputs/20171222022435216400_FW_DC.wfs"

    print "Processing data for:", MT_data_filename

    # Import MT data and associated waveforms:
    uid, MTp, MTs, stations = load_MT_dict_from_file(MT_data_filename)
    wfs_dict = load_MT_waveforms_dict_from_file(MT_waveforms_data_filename)
    
    # Get most likely solution:
    index_MT_max_prob = np.argmax(MTp) # Index of most likely MT solution
    MT_max_prob = MTs[:,index_MT_max_prob]
    # And get full MT matrix:
    full_MT_max_prob = get_full_MT_array(MT_max_prob)
    print "Full MT (max prob.):"
    print full_MT_max_prob
    print "(For plotting radiation pattern)"
    
    # Plot MT solutions and radiation pattern of most likely on sphere:
    MTs_to_plot = full_MT_max_prob #MTs_max_gau_loc
    radiation_pattern_MT = MT_max_prob # 6 moment tensor to plot radiation pattern for
    figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+".png"
    plot_full_waveform_result_beachball_DC(MTs_to_plot, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, unconstrained_vs_DC_switch="DC")
    
    print "Finished processing unconstrained inversion data for:", MT_data_filename
    
    print "Finished"
    
    
    
    





