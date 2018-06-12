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
        stations = np.array(FW_dict["stations"])
        
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

def convert_spherical_coords_to_cartesian_coords(r,theta,phi):
    """Function to take spherical coords and convert to cartesian coords. (theta between 0 and pi, phi between 0 and 2pi)"""
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D equal area projection."""
    X = x * np.sqrt(1/(1+z))
    Y = y * np.sqrt(1/(1+z))
    return X,Y

def rotate_threeD_coords_about_spec_axis(x, y, z, rot_angle, axis_for_rotation="x"):
    """Function to rotate about x-axis (right-hand rule). Rotation angle must be in radians."""
    if axis_for_rotation == "x":
        x_rot = x
        y_rot = (y*np.cos(rot_angle)) - (z*np.sin(rot_angle))
        z_rot = (y*np.sin(rot_angle)) + (z*np.cos(rot_angle))
    elif axis_for_rotation == "y":
        x_rot = (x*np.cos(rot_angle)) + (z*np.sin(rot_angle))
        y_rot = y
        z_rot = (z*np.cos(rot_angle)) - (x*np.sin(rot_angle))
    elif axis_for_rotation == "z":
        x_rot = (x*np.cos(rot_angle)) - (y*np.sin(rot_angle))
        y_rot = (x*np.sin(rot_angle)) + (y*np.cos(rot_angle))
        z_rot = z
    return x_rot, y_rot, z_rot

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

def plot_radiation_pattern_for_given_NED_DC_sixMT(ax, radiation_pattern_MT, bounding_circle_path, lower_upper_hemi_switch="lower", radiation_MT_phase="P", unconstrained_vs_DC_switch="unconstrained", plot_plane="EN"):
    """Function to plot radiation pattern on axis ax, given 6 MT describing MT to plot radiation pattern for and other args.
    Outputs axis ax with radiation pattern plotted."""
    # Get MT to plot radiation pattern for:
    ned_mt = radiation_pattern_MT

    # Get spherical points to sample for radiation pattern:
    theta = np.linspace(0,np.pi,100)
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
        # Perform rotation of plot plane if required:
        if plot_plane == "EZ":
            x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
        elif plot_plane == "NZ":
            x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
            x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
        X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
        # And plot (but ONLY if within bounding circle):
        if bounding_circle_path.contains_point((X[0],Y[0]), radius=0):
            poly_corner_coords = [(Y[0],X[0]), (Y[2],X[2]), (Y[3],X[3]), (Y[1],X[1])]
            if unconstrained_vs_DC_switch == "DC":
                polygon_curr = Polygon(poly_corner_coords, closed=True, facecolor=matplotlib.cm.binary(128 + int(disp_magn[b]*128)), alpha=0.6)
            elif unconstrained_vs_DC_switch == "unconstrained":
                polygon_curr = Polygon(poly_corner_coords, closed=True, facecolor=matplotlib.cm.RdBu(128 - int(disp_magn[b]*128)), alpha=0.6)
            ax.add_patch(polygon_curr)
    # Plot final point (theta,phi=0,0) (beginning point):
    if unconstrained_vs_DC_switch == "DC":
        centre_area = Circle([0.,0.], radius=theta_spacing/2., facecolor=matplotlib.cm.binary(128 + int(disp_magn[b]*128)), alpha=0.6)
    elif unconstrained_vs_DC_switch == "unconstrained":
        centre_area = Circle([0.,0.], radius=theta_spacing/2., facecolor=matplotlib.cm.RdBu(128 - int(disp_magn[b]*128)), alpha=0.6)
    ax.add_patch(centre_area)
    
    return ax

def plot_nodal_planes_for_given_NED_sixMT(ax, MT_for_nodal_planes, bounding_circle_path, lower_upper_hemi_switch="lower", alpha_nodal_planes=0.3, plot_plane="EN"):
    """Function for plotting nodal planes on axis ax, for given 6MT in NED format for nodal planes."""

    ned_mt = MT_for_nodal_planes

    # Get 3D nodal planes:
    plane_1_3D, plane_2_3D = get_nodal_plane_xyz_coords(ned_mt)
    # And switch vertical if neccessary:
    if lower_upper_hemi_switch=="upper":
        plane_1_3D[2,:] = -1*plane_1_3D[2,:] # as positive z is down, therefore down gives spherical projection
        plane_2_3D[2,:] = -1*plane_2_3D[2,:] # as positive z is down, therefore down gives spherical projection
    # Specify 3D coords explicitely:
    x1, y1, z1 = plane_1_3D[0],plane_1_3D[1],plane_1_3D[2]
    x2, y2, z2 = plane_2_3D[0],plane_2_3D[1],plane_2_3D[2]
    # Perform rotation of plot plane if required:
    if plot_plane == "EZ":
        x1, y1, z1 = rotate_threeD_coords_about_spec_axis(x1, y1, z1, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
        x2, y2, z2 = rotate_threeD_coords_about_spec_axis(x2, y2, z2, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
    elif plot_plane == "NZ":
        x1,y1,z1 = rotate_threeD_coords_about_spec_axis(x1, y1, z1, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x1,y1,z1 = rotate_threeD_coords_about_spec_axis(x1, y1, z1, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
        x2,y2,z2 = rotate_threeD_coords_about_spec_axis(x2, y2, z2, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x2,y2,z2 = rotate_threeD_coords_about_spec_axis(x2, y2, z2, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
    # And convert to 2D:
    X1,Y1 = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x1, y1, z1)
    X2,Y2 = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x2, y2, z2)

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

def plot_nodal_planes_for_given_single_force_vector(ax, single_force_vector_to_plot, alpha_single_force_vector=0.8, plot_plane="EN"):
    """Function for plotting single force vector on beachball style plot."""
    
    # normalise:
    single_force_vector_to_plot = single_force_vector_to_plot/np.max(np.absolute(single_force_vector_to_plot))
    
    # Convert 3D vector to 2D plane coords:
    # Note: Single force vector in is is NED format
    x = np.array([single_force_vector_to_plot[1]])
    y = np.array([single_force_vector_to_plot[0]])
    z = np.array([single_force_vector_to_plot[2]])*-1. # -1 factor as single force z is down (?) whereas 
    # Perform rotation of plot plane if required:
    if plot_plane == "EZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
    elif plot_plane == "NZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
    X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
    
    # And plot:
    ax.quiver([0.],[0.],Y,X,color="#0B7EB3",alpha=alpha_single_force_vector, angles='xy', scale_units='xy', scale=1)
    
    return ax

def plot_full_waveform_result_beachball(MTs_to_plot, wfs_dict, radiation_pattern_MT=[], stations=[], lower_upper_hemi_switch="lower", figure_filename=[], num_MT_solutions_to_plot=20, inversion_type="unconstrained", radiation_MT_phase="P", plot_plane="EN"):
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
    ax.quiver([-2.6],[2.5],[0.5],[0.],color="k",alpha=0.8, angles='xy', scale_units='xy', scale=1) # Plot x direction label
    ax.quiver([-2.5],[2.4],[0.],[0.5],color="k",alpha=0.8, angles='xy', scale_units='xy', scale=1) # Plot y direction label
    if plot_plane=="EN":
        plt.text(-2.0,2.5,"E",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # x label
        plt.text(-2.5,3.0,"N",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # y label
    elif plot_plane=="EZ":
        plt.text(-2.0,2.5,"E",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # x label
        plt.text(-2.5,3.0,"Z",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # y label
    elif plot_plane=="NZ":
        plt.text(-2.0,2.5,"N",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # x label
        plt.text(-2.5,3.0,"Z",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # y label
    plt.axis('off')
    
    # Setup bounding circle and create bounding path from circle:
    ax, bounding_circle_path = create_and_plot_bounding_circle_and_path(ax)
    
    
    # Plot radiation pattern and nodal planes if inversion_type is DC or unconstrained:
    if inversion_type == "DC" or inversion_type == "unconstrained":
        unconstrained_vs_DC_switch = inversion_type
        
        # Plot radiation pattern if provided with radiation pattern MT to plot:
        if not len(radiation_pattern_MT)==0:
            plot_radiation_pattern_for_given_NED_DC_sixMT(ax, radiation_pattern_MT, bounding_circle_path, lower_upper_hemi_switch=lower_upper_hemi_switch, radiation_MT_phase=radiation_MT_phase, unconstrained_vs_DC_switch=unconstrained_vs_DC_switch, plot_plane=plot_plane) # Plot radiation pattern
    
        # Plot MT nodal plane solutions:
        # Get samples to plot:
        # IF single sample, plot most likely (assocaited with radiation pattern):
        if num_MT_solutions_to_plot == 1:
            if not len(radiation_pattern_MT)==0:
                curr_MT_to_plot = radiation_pattern_MT
                ax = plot_nodal_planes_for_given_NED_sixMT(ax, curr_MT_to_plot, bounding_circle_path, lower_upper_hemi_switch, alpha_nodal_planes=0.3, plot_plane=plot_plane)
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
                ax = plot_nodal_planes_for_given_NED_sixMT(ax, curr_MT_to_plot, bounding_circle_path, lower_upper_hemi_switch, alpha_nodal_planes=0.3, plot_plane=plot_plane)
    
    # Or plot single force vector if inversion_type is single_force:
    elif inversion_type == "single_force":
        single_force_vector_to_plot = radiation_pattern_MT
        ax = plot_nodal_planes_for_given_single_force_vector(ax, single_force_vector_to_plot, alpha_single_force_vector=0.8, plot_plane=plot_plane)
                
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
            # And correct for points below horizontal plane:
            # Note: Could correct for each plane, but currently only correct to place stations for EN plane, regardless of rotation.
            if theta>np.pi/2.:
                theta = theta - np.pi
                phi = phi + np.pi
            if plot_plane == "EZ":
                if (phi>0. and phi<=np.pi/2.) or (phi>3.*np.pi/2. and phi<=2.*np.pi):
                    theta = theta + np.pi/2.
                    phi = phi + np.pi
            elif plot_plane == "NZ":
                if phi>np.pi and phi<=2.*np.pi:
                    theta = theta + np.pi/2.
                    phi = phi + np.pi
            r = 1.0/np.sqrt(2.) # as on surface of focal sphere (but sqrt(2) as other previous plotting reduces size of sphere.)
            x,y,z = convert_spherical_coords_to_cartesian_coords(r, theta, phi)
            # Perform rotation of plot plane if required:
            if plot_plane == "EZ":
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
            elif plot_plane == "NZ":
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
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
            # Note: Will get all components for current station
            real_wfs_current_station = []
            synth_wfs_current_station = []
            wfs_component_labels_current_station = []
            for wfs_key in wfs_dict.keys():
                if station_name in wfs_key:
                    real_wfs_current_station.append(wfs_dict[wfs_key]['real_wf']) # Append current real waveforms to wfs for current station
                    synth_wfs_current_station.append(wfs_dict[wfs_key]['synth_wf']) # Append current synth waveforms to wfs for current station
                    wfs_component_labels_current_station.append(wfs_key.split(", ")[1]) # Get current component label
            # and reorder if have Z,R and T components:
            wfs_component_labels_current_station_sorted = list(wfs_component_labels_current_station)
            wfs_component_labels_current_station_sorted.sort()
            if wfs_component_labels_current_station_sorted == ['R','T','Z']:
                real_wfs_current_station_unsorted = list(real_wfs_current_station)
                synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
                idx_tmp = wfs_component_labels_current_station.index("R")
                real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
                synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
                idx_tmp = wfs_component_labels_current_station.index("T")
                real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
                synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
                idx_tmp = wfs_component_labels_current_station.index("Z")
                real_wfs_current_station[2] = real_wfs_current_station_unsorted[idx_tmp]
                synth_wfs_current_station[2] = synth_wfs_current_station_unsorted[idx_tmp]
                wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
            elif wfs_component_labels_current_station_sorted == ['L','Q','T']:
                real_wfs_current_station_unsorted = list(real_wfs_current_station)
                synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
                idx_tmp = wfs_component_labels_current_station.index("L")
                real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
                synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
                idx_tmp = wfs_component_labels_current_station.index("Q")
                real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
                synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
                idx_tmp = wfs_component_labels_current_station.index("T")
                real_wfs_current_station[2] = real_wfs_current_station_unsorted[idx_tmp]
                synth_wfs_current_station[2] = synth_wfs_current_station_unsorted[idx_tmp]
                wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
            # for wfs_dict_station_idx in range(len(wfs_dict.keys())):
            #     if wfs_dict.keys()[wfs_dict_station_idx].split(",")[0] == station_name:
            #         real_wf_current_stat = wfs_dict[wfs_dict.keys()[wfs_dict_station_idx]]['real_wf']
            #         synth_wf_current_stat = wfs_dict[wfs_dict.keys()[wfs_dict_station_idx]]['synth_wf']
            # Get coords to plot waveform at:
            if plot_plane == "EN":
                theta = np.pi/2. # Set theta to pi/2 as want to just plot waveforms in horizontal plane (if plot_plane == "EN")
                r = 2.0 # as want to plot waveforms beyond extent of focal sphere
            elif plot_plane == "EZ":
                if theta == np.pi/2. and (phi==np.pi or phi==2.*np.pi):
                    phi = np.pi
                    r = 2.
                else:
                    r = np.sqrt(25./((np.cos(theta)**2) + (np.sin(phi)**2))) # as want to plot waveforms beyond extent of focal sphere
            elif plot_plane == "NZ":
                if theta == np.pi/2. and (phi==np.pi/2. or phi==3.*np.pi/2.):
                    phi = np.pi
                    r = 2.
                else:
                    r = np.sqrt(25./((np.cos(theta)**2) + (np.cos(phi)**2))) # as want to plot waveforms beyond extent of focal sphere
            x,y,z = convert_spherical_coords_to_cartesian_coords(r, theta, phi)
            # Perform rotation of plot plane if required:
            if plot_plane == "EZ":
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
            elif plot_plane == "NZ":
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
            X_waveform_loc, Y_waveform_loc = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
            data_xy_coords = (Y_waveform_loc, X_waveform_loc)
            disp_coords = ax.transData.transform(data_xy_coords) # And transform data coords into display coords
            fig_inv = fig.transFigure.inverted() # Create inverted figure transformation
            fig_coords = fig_inv.transform((disp_coords[0],disp_coords[1])) # Transform display coords into figure coords (for adding axis)
            # Plot if waveform exists for current station:
            if len(real_wfs_current_station)>0:
                # Plot line to waveform:
                ax.plot([Y,Y_waveform_loc],[X,X_waveform_loc],c='k',alpha=0.6)
                # And plot waveform:
                left, bottom, width, height = [fig_coords[0]-0.15, fig_coords[1]-0.1, 0.3, 0.2]
                # for each wf component:
                if len(real_wfs_current_station)>1:
                    for k in range(len(real_wfs_current_station)):
                        bottom_tmp = bottom + k*height/len(real_wfs_current_station)
                        inset_ax_tmp = fig.add_axes([left, bottom_tmp, width, height/len(real_wfs_current_station)])
                        #inset_ax1 = inset_axes(ax,width="10%",height="5%",bbox_to_anchor=(0.2,0.4))
                        inset_ax_tmp.plot(real_wfs_current_station[k],c='k', alpha=0.6, linewidth=0.75) # Plot real data
                        inset_ax_tmp.plot(synth_wfs_current_station[k],c='#E83313',linestyle="--", alpha=0.6, linewidth=0.5) # Plot synth data
                        # inset_ax_tmp.set_ylabel(wfs_component_labels_current_station[k],loc="left",size=10)
                        plt.title(wfs_component_labels_current_station[k],loc="left",size=8)
                        plt.axis('off')
                elif len(real_wfs_current_station)==1:
                    inset_ax_tmp = fig.add_axes([left, bottom, width, height])
                    #inset_ax1 = inset_axes(ax,width="10%",height="5%",bbox_to_anchor=(0.2,0.4))
                    inset_ax_tmp.plot(real_wfs_current_station[0],c='k', alpha=0.6, linewidth=0.75) # Plot real data
                    inset_ax_tmp.plot(synth_wfs_current_station[0],c='#E83313',linestyle="--", alpha=0.6, linewidth=0.75) # Plot synth data
                    plt.axis('off')
        
    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
    else:
        plt.show()
        
def plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=[], inversion_type=""):
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
    
    # Set first and final bins equal to twice the bin value (as they only get values rounded from half the region of other bins):
    probability_percentage_DC_all_solns_bins[0] = probability_percentage_DC_all_solns_bins[0]*2.
    probability_percentage_DC_all_solns_bins[-1] = probability_percentage_DC_all_solns_bins[-1]*2.
    probability_percentage_SF_all_solns_bins[0] = probability_percentage_SF_all_solns_bins[0]*2.
    probability_percentage_SF_all_solns_bins[-1] = probability_percentage_SF_all_solns_bins[-1]*2.
    
    # And plot results:
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    plt.plot(percentage_DC_all_solns_bins[:], probability_percentage_DC_all_solns_bins[:], c='#D94411')
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
        ax1.set_xlabel("Percentage DC")
    elif inversion_type == "single_force_crack_no_coupling":
        ax1.set_xlabel("Percentage crack")
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
        
# Lune plotting functions:
def get_frac_of_MTs_using_MT_probs(MTs, MTp, frac_to_sample):
    """Function to return fraction of MTs based on highet probabilities."""
    num_events_to_sample = int(len(MTp)*frac_to_sample) # Take top 1 % of samples
    sorted_indices = np.argsort(MTp)[::-1] # reorder into descending order
    # Find indices of solutions in sample:
    sample_indices = sorted_indices[0:num_events_to_sample]
    MTs_sample = MTs[:,sample_indices]
    print "Sampled",len(MTs_sample[0,:]),"out of",len(MTs[0,:]),"events"
    return MTs_sample

def find_delta_gamm_values_from_sixMT(sixMT):
    """Function to find delta and gamma given 6 moment tensor."""
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
    # print (lambda1**2 + lambda2**2 + lambda3**2)**0.5 # Should = 1 if normallised correctly
    gamma = np.arctan(((-1*lambda1) + (2*lambda2) - lambda3)/((3**0.5)*(lambda1 - lambda3))) # eq. 20a (Tape and Tape 2012)
    beta = np.arccos((lambda1+lambda2+lambda3)/((3**0.5)*((lambda1**2 + lambda2**2 + lambda3**2)**0.5))) # eq. 20b (Tape and Tape 2012)
    delta = (np.pi/2.) - beta # eq. 23 (Tape and Tape 2012)

    return delta, gamma
    
def get_binned_MT_solutions_by_delta_gamma_dict(MTs, return_all_switch=False):
    """Function to get binned MT solutions by delta and gamma value. Input is array of MTs (in (6,n) shape).
    Output is binned dictionary containing bin values of delta and gamma and all MT solutions that are in the bin."""
    
    # Set up store for binned MT data:
    gamma_delta_binned_MT_store = {} # Will have the entries: gamma_delta_binned_MT_store[delta][gamma][array of MTs (shape(6,n))]

    # Setup delta-gamma bins for data:
    bin_size_delta = np.pi/120. #np.pi/60.
    bin_size_gamma = np.pi/120. #np.pi/60.
    bin_value_labels_delta = np.arange(-np.pi/2,np.pi/2+bin_size_delta, bin_size_delta)
    bin_value_labels_gamma = np.arange(-np.pi/6,np.pi/6+bin_size_gamma, bin_size_gamma)
    bins_delta_gamma = np.zeros((len(bin_value_labels_delta), len(bin_value_labels_gamma)), dtype=float) # array to store bin values (although can also obtain from dictionary sizes)
    
    # And setup dict for all binned values:
    for delta in bin_value_labels_delta:
        for gamma in bin_value_labels_gamma:
            try:
                gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)] = {}
            except KeyError:
                gamma_delta_binned_MT_store["delta="+str(delta)] = {}
                gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)] = {}
    
    # Loop over events (binning each data point):
    for a in range(len(MTs[0,:])):
        # Get delta and gamma values for sixMT:
        MT_current = MTs[:,a]
        delta, gamma = find_delta_gamm_values_from_sixMT(MT_current)

        # And bin solution into approriate bin:
        idx_delta = (np.abs(bin_value_labels_delta-delta)).argmin()
        idx_gamma = (np.abs(bin_value_labels_gamma-gamma)).argmin()
        bins_delta_gamma[idx_delta,idx_gamma] += 1. # Append 1 to bin
        
        # And add to dictionary:
        delta_bin_label_tmp = bin_value_labels_delta[idx_delta]
        gamma_bin_label_tmp = bin_value_labels_gamma[idx_gamma]
        try:
            tmp_MT_stacked_array = gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"]
            gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"] = np.hstack((tmp_MT_stacked_array, MT_current.reshape(6,1)))
        except KeyError:
            gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"] = np.array(MT_current.reshape(6,1)) # If doesnt exist, create new MT store entry
    
    if return_all_switch:
        return gamma_delta_binned_MT_store, bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma
    else:
        return gamma_delta_binned_MT_store

def twoD_Gaussian((X, Y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    """Function describing 2D Gaussian. Pass initial guesses for gaussian parameters. Returns 1D ravelled array describing 2D Gaussian function.
    Based on code: https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
    X, Y are 2D np grids (from np.meshgrid)."""
    xo = float(xo)
    yo = float(yo)
    amplitude = float(amplitude)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    gau = amplitude*np.exp( - (a*((X-xo)**2) + 2*b*(X-xo)*(Y-yo) + c*((Y-yo)**2)))
    #gau_out = gau.ravel() # Makes 2D gau array 1D, as otherwise fitting curve function won't work!
    gau_out = np.ravel(gau) # Makes 2D gau array 1D, as otherwise fitting curve function won't work!
    return gau_out

def fit_twoD_Gaussian(x, y, data, initial_guess_switch=False, initial_guess=(1,1,1,1,1)):
    """Function to fit 2D Gaussian to a dataset. x, y are 1D data arrays, data is a 2D array, described by x and y as labels.
    Based on code from:
    https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m"""    
    
    # Mesh grid for 2D Gaussian fit:
    Y, X = np.meshgrid(y, x)
    
    # Fit Gaussian to data:
    data_ravelled = np.ravel(data)
    if initial_guess_switch:
        print "Initial guess parameters for 2D gaussian fit:"
        print initial_guess
        popt, pcov = opt.curve_fit(twoD_Gaussian, (X, Y), data_ravelled, p0=initial_guess)
    else:
        popt, pcov = opt.curve_fit(twoD_Gaussian, (X, Y), data_ravelled)
    print "And final parameters derived:"
    print popt
    
    # Get fitted data:
    data_fitted = twoD_Gaussian((X, Y), *popt) # Get 2D Gaussian
    data_fitted = np.reshape(data_fitted, np.shape(data)) # and reshape to original data dimensions
    
    return data_fitted
    
def equal_angle_stereographic_projection_conv_YZ_plane(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D equal area projection."""
    Y = y/(1+x)
    Z = z/(1+x)
    return Y,Z

def plot_Lune(MTs, MTp, six_MT_max_prob=[], frac_to_sample=0.1, figure_filename=[]):
    """Function to plot Lune plot for certain inversions (if Lune plot is relevent, i.e. not DC constrained or single-force constrained).
    Will plot sampled MT solutions on Lune, binned. Will also fit gaussian to this and return the maximum location of the gaussian and the contour coordinates. Also outputs saved figure."""
    
    # Get sample of MT solutions for fitting Gaussian to:
    MTs_sample = get_frac_of_MTs_using_MT_probs(MTs, MTp, frac_to_sample)
    
    # Get bin values for delta-gamma space (for plotting Lune):
    gamma_delta_binned_MT_store, bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma = get_binned_MT_solutions_by_delta_gamma_dict(MTs_sample, return_all_switch=True)

    # Fit 2D gaussian to delta-gamma Lune data:
    # Define initial guess params:
    amplitude = np.max(bins_delta_gamma)
    xo = 0.0
    yo = 0.0
    sigma_x = np.pi/6.
    sigma_y = np.pi/8.
    theta = np.pi/2.
    initial_guess=(amplitude, xo, yo, sigma_x, sigma_y, theta) # Define initial guess values from data
    # And fit gaussian:
    bins_delta_gamma_gau_fitted = fit_twoD_Gaussian(bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma, initial_guess_switch=True, initial_guess=initial_guess)
    
    # Get location of maximum of Gaussian fit and 1 stdev contour:
    # Get location of maximum:
    max_bin_delta_gamma_indices = np.where(bins_delta_gamma_gau_fitted==np.max(bins_delta_gamma_gau_fitted))
    max_bin_delta_gamma_values = [bin_value_labels_delta[max_bin_delta_gamma_indices[0][0]], bin_value_labels_gamma[max_bin_delta_gamma_indices[1][0]]]
    # Get contour:
    contour_val = bins_delta_gamma_gau_fitted[max_bin_delta_gamma_indices[0][0], max_bin_delta_gamma_indices[1][0]]/2. #np.std(bins_delta_gamma_gau_fitted)
    plus_minus_range = 0.05
    contour_delta_values_indices = []
    contour_gamma_values_indices = []
    contour_delta_values = []
    contour_gamma_values = []
    for i in range(len(bins_delta_gamma_gau_fitted[:,0])):
        for j in range(len(bins_delta_gamma_gau_fitted[0,:])):
            if bins_delta_gamma_gau_fitted[i,j]<=contour_val*(1.+plus_minus_range) and bins_delta_gamma_gau_fitted[i,j]>=contour_val*(1.-plus_minus_range):
                # Find contour values:
                contour_delta_values_indices.append(i)
                contour_gamma_values_indices.append(j)
                contour_delta_values.append(bin_value_labels_delta[i])
                contour_gamma_values.append(bin_value_labels_gamma[j])        
    # And sort contour points into clockwise order:
    pts = []
    for i in range(len(contour_delta_values)):
        pts.append([contour_delta_values[i], contour_gamma_values[i]])
    origin = pts[0]
    refvec = [0, 1]
    # Define function for plotting contour:
    def clockwiseangle_and_distance(point):
        """Function to order points in clockwise order. Needs origin and refvec defined.
        Code from: https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python"""
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector
    contour_bin_delta_gamma_values_sorted = sorted(pts, key=clockwiseangle_and_distance) # Sorts points into clockwise order
    contour_bin_delta_gamma_values_sorted.append(contour_bin_delta_gamma_values_sorted[0]) # Append first point again to make circle
    
    # And plot:
    print "Plotting Lune with fitted Gaussian"
    # Set up figure:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    # Plot major gridlines:
    for phi in [-np.pi/6., np.pi/6.]:
        theta_range = np.linspace(0.0,np.pi,180)
        phi_range = np.ones(len(theta_range))*phi
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black")
    # Plot horizontal minor grid lines:
    minor_horiz_interval = np.pi/12.
    for theta in np.arange(0.+minor_horiz_interval, np.pi+minor_horiz_interval, minor_horiz_interval):
        phi_range = np.linspace(-np.pi/6,np.pi/6,90)
        theta_range = np.ones(len(phi_range))*theta
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black", linestyle="--", alpha=0.5)
    # Plot vertical minor gridlines:
    minor_vert_interval = np.pi/24.
    for phi in np.arange(-np.pi/6+minor_vert_interval, np.pi/6, minor_vert_interval):
        theta_range = np.linspace(0.0,np.pi,180)
        phi_range = np.ones(len(theta_range))*phi
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black", linestyle="--", alpha=0.5)

    # And plot binned data, colored by bin value:
    bins_delta_gamma_normallised = bins_delta_gamma/np.max(bins_delta_gamma) # Normalise data
    # Loop over binned data points:
    for i in range(len(bin_value_labels_delta)):
        for j in range(len(bin_value_labels_gamma)):
            delta = bin_value_labels_delta[i]
            gamma = bin_value_labels_gamma[j]
            # And plot data coord:
            x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
            Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
            ax.scatter(Y,Z, color = matplotlib.cm.jet(int(bins_delta_gamma_normallised[i,j]*256)), alpha=0.6,s=50)
        print i

    # # Plot maximum location and associated contours associated with Guassian fit:
    # # Plot maximum location:
    # delta = max_bin_delta_gamma_values[0]
    # gamma = max_bin_delta_gamma_values[1]
    # x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
    # Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    # ax.scatter(Y,Z, color = "green", alpha=1.0,s=50, marker="X")
    # # And plot 1 stdev contour:
    # contour_bin_delta_values_sorted = []
    # contour_bin_gamma_values_sorted = []
    # for i in range(len(contour_bin_delta_gamma_values_sorted)):
    #     contour_bin_delta_values_sorted.append(contour_bin_delta_gamma_values_sorted[i][0])
    #     contour_bin_gamma_values_sorted.append(contour_bin_delta_gamma_values_sorted[i][1])
    # delta = np.array(contour_bin_delta_values_sorted)
    # gamma = np.array(contour_bin_gamma_values_sorted)
    # x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
    # Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    # ax.plot(Y,Z, color = "green", alpha=0.5)
    
    # Plot location of maximum probability single MT solution (passed as argument):
    if len(six_MT_max_prob)>0:
        delta, gamma = find_delta_gamm_values_from_sixMT(six_MT_max_prob)
        # And plot data coord:
        x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
        Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.scatter(Y,Z, c="gold", alpha=0.8,s=250, marker="*")
    
    # And Finish plot:
    # Plot labels for various defined locations (locations from Tape and Tape 2012, table 1):
    plt.scatter(0.,1.,s=50,color="black")
    plt.text(0.,1.,"Explosion", fontsize=12, horizontalalignment="center", verticalalignment='bottom')
    plt.scatter(0.,-1.,s=50,color="black")
    plt.text(0.,-1.,"Implosion", fontsize=12, horizontalalignment="center", verticalalignment='top')
    x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - np.arcsin(5/np.sqrt(33)),-np.pi/6.)
    Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    plt.scatter(Y,Z,s=50,color="red")
    plt.text(Y,Z,"TC$^+$",color="red", fontsize=12, horizontalalignment="right", verticalalignment='bottom')
    x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) + np.arcsin(5/np.sqrt(33)),np.pi/6.)
    Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    plt.scatter(Y,Z,s=50,color="red")
    plt.text(Y,Z,"TC$^-$",color="red", fontsize=12, horizontalalignment="left", verticalalignment='top')
    plt.scatter(0.,0.,s=50,color="red")
    plt.text(0.,0.,"DC",color="red", fontsize=12, horizontalalignment="center", verticalalignment='top')
    # Various tidying:
    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.,1.)
    plt.axis('off')
    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
    else:
        plt.show()
    
    # And return MT data at maximum (and mts within contour?!):
    print "And getting MT data at maximum of gaussian to return (and mts within contour?!)"
    # Get all solutions associated with bins inside contour on Lune plot:
    gamma_delta_binned_MT_store = get_binned_MT_solutions_by_delta_gamma_dict(MTs_sample) # Returns dictionary of all MTs binned by gamma, delta value
    # And get all values associated with gaussian maximum on Lune plot:
    max_bin_delta_gamma_indices = np.where(bins_delta_gamma_gau_fitted==np.max(bins_delta_gamma_gau_fitted))
    max_bin_delta_gamma_values = [bin_value_labels_delta[max_bin_delta_gamma_indices[0][0]], bin_value_labels_gamma[max_bin_delta_gamma_indices[1][0]]]
    delta = max_bin_delta_gamma_values[0]
    gamma = max_bin_delta_gamma_values[1]
    MTs_max_gau_loc = gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)]["MTs"] # MT solutions associated with gaussian maximum (note: may be different to maximum value due to max value being fit rather than real value)
    
    return MTs_max_gau_loc
        

def run(inversion_type, event_uid, datadir, radiation_MT_phase="P", plot_Lune_switch=True):
    """Function to run main script."""
    
    # Plot for inversion:
    print "Plotting data for inversion"
    
    # Get inversion filenames:
    MT_data_filename = datadir+"/"+event_uid+"_FW_"+inversion_type+".pkl" #"./python_FW_outputs/20171222022435216400_FW_DC.pkl"
    MT_waveforms_data_filename = datadir+"/"+event_uid+"_FW_"+inversion_type+".wfs"  #"./python_FW_outputs/20171222022435216400_FW_DC.wfs"

    print "Processing data for:", MT_data_filename

    # Import MT data and associated waveforms:
    uid, MTp, MTs, stations = load_MT_dict_from_file(MT_data_filename)
    wfs_dict = load_MT_waveforms_dict_from_file(MT_waveforms_data_filename)
    
    
    # Get most likely solution and plot:
    index_MT_max_prob = np.argmax(MTp) # Index of most likely MT solution
    MT_max_prob = MTs[:,index_MT_max_prob]
    
    if inversion_type == "full_mt":
        inversion_type = "unconstrained"
        # And get full MT matrix:
        full_MT_max_prob = get_full_MT_array(MT_max_prob)
        # Plot MT solutions and radiation pattern of most likely on sphere:
        MTs_to_plot = full_MT_max_prob #MTs_max_gau_loc
        radiation_pattern_MT = MT_max_prob # 6 moment tensor to plot radiation pattern for
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+".png"
            plot_full_waveform_result_beachball(MTs_to_plot, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type=inversion_type, radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
        # And plot Lune for solution:
        if plot_Lune_switch:
            plot_Lune(MTs, MTp, six_MT_max_prob=radiation_pattern_MT, frac_to_sample=0.1, figure_filename="Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_Lune.png")
    
    elif inversion_type == "DC":
        # And get full MT matrix:
        full_MT_max_prob = get_full_MT_array(MT_max_prob)
        # Plot MT solutions and radiation pattern of most likely on sphere:
        MTs_to_plot = full_MT_max_prob #MTs_max_gau_loc
        radiation_pattern_MT = MT_max_prob # 6 moment tensor to plot radiation pattern for
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+".png"
            plot_full_waveform_result_beachball(MTs_to_plot, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type=inversion_type, radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
    
    elif inversion_type == "single_force":
        full_MT_max_prob = MT_max_prob
        # Plot MT solutions and radiation pattern of most likely on sphere:
        MTs_to_plot = full_MT_max_prob #MTs_max_gau_loc
        radiation_pattern_MT = MT_max_prob # 6 moment tensor to plot radiation pattern for
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+".png"
            plot_full_waveform_result_beachball(MTs_to_plot, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type=inversion_type, radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
    
    elif inversion_type == "DC_single_force_couple":
        full_MT_max_prob = get_full_MT_array(MT_max_prob[0:6])
        radiation_pattern_MT = MT_max_prob[0:6]
        single_force_vector_max_prob = MT_max_prob[6:9]
        amp_prop_DC = MT_max_prob[9] # Proportion of amplitude that is DC
        # Plot MT solutions and radiation pattern of most likely on sphere:
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_DC_component.png"
            plot_full_waveform_result_beachball(full_MT_max_prob, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="DC", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_SF_component.png"
            plot_full_waveform_result_beachball(single_force_vector_max_prob, wfs_dict, radiation_pattern_MT=single_force_vector_max_prob, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="single_force", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
        # And plot probability distribution for DC vs. single force:
        figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+"DC_vs_SF_prob_dist.png"
        plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=figure_filename)
    
    elif inversion_type == "DC_single_force_no_coupling":
        full_MT_max_prob = get_full_MT_array(MT_max_prob[0:6])
        radiation_pattern_MT = MT_max_prob[0:6]
        single_force_vector_max_prob = MT_max_prob[6:9]
        amp_prop_DC = MT_max_prob[9] # Proportion of amplitude that is DC
        # Plot MT solutions and radiation pattern of most likely on sphere:
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_DC_component.png"
            plot_full_waveform_result_beachball(full_MT_max_prob, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="DC", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_SF_component.png"
            plot_full_waveform_result_beachball(single_force_vector_max_prob, wfs_dict, radiation_pattern_MT=single_force_vector_max_prob, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="single_force", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
        # And plot probability distribution for DC vs. single force:
        figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+"DC_vs_SF_prob_dist.png"
        plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=figure_filename)
    
    elif inversion_type == "DC_crack_couple":
        full_MT_max_prob = get_full_MT_array(MT_max_prob[0:6])
        radiation_pattern_MT = MT_max_prob[0:6]
        amp_prop_DC = MT_max_prob[-1] # Proportion of amplitude that is DC
        # Plot MT solutions and radiation pattern of most likely on sphere:
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+".png"
            plot_full_waveform_result_beachball(full_MT_max_prob, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="unconstrained", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
        # And plot Lune for solution:
        if plot_Lune_switch:
            plot_Lune(MTs[0:6,:], MTp, six_MT_max_prob=radiation_pattern_MT, frac_to_sample=0.1, figure_filename="Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_Lune.png")    
        
    elif inversion_type == "single_force_crack_no_coupling":
        full_MT_max_prob = get_full_MT_array(MT_max_prob[0:6])
        radiation_pattern_MT = MT_max_prob[0:6]
        single_force_vector_max_prob = MT_max_prob[6:9]
        amp_prop_SF = MT_max_prob[9] # Proportion of amplitude that is DC
        # Plot MT solutions and radiation pattern of most likely on sphere:
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_crack_component.png"
            plot_full_waveform_result_beachball(full_MT_max_prob, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="unconstrained", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
            figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_SF_component.png"
            plot_full_waveform_result_beachball(single_force_vector_max_prob, wfs_dict, radiation_pattern_MT=single_force_vector_max_prob, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="single_force", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane)
        # And plot probability distribution for DC vs. single force:
        figure_filename = "Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+"crack_vs_SF_prob_dist.png"
        plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=figure_filename, inversion_type=inversion_type)
        # And plot Lune for solution:
        if plot_Lune_switch:
            plot_Lune(MTs[0:6,:], MTp, six_MT_max_prob=radiation_pattern_MT, frac_to_sample=0.1, figure_filename="Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_Lune.png")
    
    print "Full MT (max prob.):"
    print full_MT_max_prob
    print "(For plotting radiation pattern)"
    
    print "Finished processing unconstrained inversion data for:", MT_data_filename
    
    print "Finished"
    
       
# ------------------- Main script for running -------------------
if __name__ == "__main__":
    
    # Specify event and inversion type:
    inversion_type = "single_force_crack_no_coupling" # can be: full_mt, DC, single_force, DC_single_force_couple, DC_single_force_no_coupling, DC_crack_couple, or single_force_crack_no_coupling.
    event_uid = "20180214185538374893" #"20140629184210365600" #"20090121042009165190" #"20171222022435216400" # Event uid (numbers in FW inversion filename)
    datadir = "./python_FW_outputs"
    radiation_MT_phase="P" # Radiation phase to plot (= "P" or "S")
    plot_Lune_switch = True # If True, plots Lune
    
    run(inversion_type, event_uid, datadir, radiation_MT_phase="P", plot_Lune_switch=plot_Lune_switch)

    
    
    
    





