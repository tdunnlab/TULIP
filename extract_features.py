#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:16:23 2024

@author: sihan

This demo code is for extracting 25 gait features from gait 3d poses in Tulip setting experiment. 
You can find more information in Tulip project page 
https://www.tulipproject.net/

All the keypoints format is halpe26, which you can find detailed information in their github page
https://github.com/Fang-Haoshu/Halpe-FullBody
"""

import numpy as np
import os
import pickle
from scipy.signal import find_peaks
from scipy.signal import medfilt, savgol_filter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def cal_distance_2d(x1,x2):
    '''
    Input:
    - x1: the coordinates of a 2D point.
    - x2: the coordinates of another 2D point.
    
    Output:
    - The Euclidean distance between the points x1 and x2.
    '''
    return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

def cal_distance_2d_with_sign(x1,x2,direction):
    '''
    Input: 
    - x1: numpy array containing 2 elements which represents a 2d point
    - x2: numpy array containing 2 elements which represents a 2d point
    - direction: 'R2L' or 'L2R' which represents the direction of subjects' walking
    
    Output:
    - the distance between x1 and x2 with sign. 
    
    x1 can be left/right toe/heel and x2 is always the hip.
        
    Take left heel as an example. If left heel is in front of the hip, then we want the distance from left
    heel to the hip to be positive. Otherwise, if the left heel is behind the hip, we want the distance from the
    left heel to the hip to be negative. 
        
    In Tulip setting, if the walking direction is from right to left (R2L), then the subject is walking on the opposite 
    durection of x-axis. Thus, if left_heel_x > hip_x, it means the left heel is behind the hip, so the distance 
    is negative. 
    If the walking direction is from left to right (L2R), then the subject is walking on the direction of 
    x-axis. If left_heel_x > hip_x, it means the left heel is in front of the hip, so the distance 
    is positive. And vice versa.
    '''
    if direction == 'R2L':# based on camera2 view
        if x1[0]>x2[0]:
            return -np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        else:
            return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
    elif direction =='L2R':
        if x1[0]>x2[0]:
            return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        else:
            return -np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        
def calculate_angle(line_list1, line_list2,input_3d_array):
    '''
    Input:
    - line_list1: A list containing the indices of two keypoints that define the first line.
    - line_list2: A list containing the indices of two keypoints that define the second line.
    Notice that if line_list2[1] == 100, if means we are calculating angles between line1 and z-axis.
    - input_3d_array: A 3D array representing the 3D pose data, from which the angles between 
    the two lines will be calculated.
    
    Output:
    - angles: A list of angles between the two lines, computed for each frame in the input_3d_array.
    '''
    angles = []
    for i in range(input_3d_array.shape[0]):
        kpts_3d = input_3d_array[i,:,:]
        vec1 = [kpts_3d[line_list1[0],0] - kpts_3d[line_list1[1],0],kpts_3d[line_list1[0],1] - kpts_3d[line_list1[1],1],\
                kpts_3d[line_list1[0],2] - kpts_3d[line_list1[1],2]]
        if line_list2[1] != 100:
            vec2 = [kpts_3d[line_list2[0],0] - kpts_3d[line_list2[1],0],kpts_3d[line_list2[0],1] - kpts_3d[line_list2[1],1],\
                    kpts_3d[line_list2[0],2] - kpts_3d[line_list2[1],2]]
        elif line_list2[1] == 100: # if line_list2[1] == 100, calculate the angles between the line 1 and the z axis
            vec2 = [kpts_3d[line_list2[0],0] - kpts_3d[line_list2[0],0],kpts_3d[line_list2[0],1] - kpts_3d[line_list2[0],1],\
                    kpts_3d[line_list2[0],2] - 0]        
        angle = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) * 180 / np.pi
        angles.append(angle)
    return angles


def cal_projected_length(p1,p2,slope,intercept,direction='none'): # hip is p2

    '''
    Input:
    - p1: numpy array representing the coordinates of the first point (2D).
    - p2: numpy array representing the coordinates of the second point (2D), typically the hip in most cases
    - slope: float, the slope of the line onto which the points are projected.
    - intercept: float, the y-intercept of the line.
    - direction: string, specifies if the projection should consider directionality 
    ('none' for unsigned distance, or 'L2R'/'R2L' for signed distance).

    Output:
    - The projected distance between p1 and p2 along the line. If direction is 'none', it returns the unsigned distance.
      Otherwise, it returns a signed distance based on the given direction.
    '''

    slope_p1 = np.array([0,intercept])
    slope_p2 = np.array([2000,float(2000 * slope + intercept)])
    length = np.sum((slope_p1-slope_p2)**2)
    
    t1 = np.sum((p1 - slope_p1) * (slope_p2 - slope_p1)) / length
    projection_p1 = slope_p1 + t1 * (slope_p2 - slope_p1)
    
    t2 = np.sum((p2 - slope_p1) * (slope_p2 - slope_p1)) / length
    projection_p2 = slope_p1 + t2 * (slope_p2 - slope_p1)
    

    if direction == 'none':
        return cal_distance_2d(projection_p1, projection_p2)
    else:
        return cal_distance_2d_with_sign(projection_p1, projection_p2,direction)


def smooth_3d_filter(input_3d_estimation_array,median_kernel_size=15,savgol_kernel_size=17):
    '''
    Input:
    - input_3d_estimation_array: numpy array of shape (n_frames, n_kpts, 3), where each frame contains 3D coordinates (x, y, z) of keypoints.
    - median_kernel_size: int, kernel size for the median filter. Default is 15.
    - savgol_kernel_size: int, window length for the Savitzky-Golay filter. Default is 17.
    
    Output:
    - smoothed_data: numpy array of the same shape as the input, containing the smoothed 3D coordinates.
    '''
    
    data_shape = np.shape(input_3d_estimation_array) # shape is n_frames, n_kpts, 3
    smoothed_data = np.zeros(data_shape)
    
    for i in range(data_shape[1]):
        for axis in range(3):
            temp_filtered_data = input_3d_estimation_array[:,i,axis]
            after_median_filter = medfilt(temp_filtered_data, kernel_size=median_kernel_size)  
            after_savgol_filter = savgol_filter(after_median_filter, window_length=savgol_kernel_size, polyorder=3)
            smoothed_data[:,i,axis] = after_savgol_filter

    
    return smoothed_data


def split_walking_period(kpts_3d,fps,save_path):
    '''
    Input: 
    - kpts_3d: numpy array of shape (frame_number, keypoints_number, 3(xyz Dimensions), where each frame contains 
    3D coordinates (x, y, z) of keypoints for gait.
    - fps: recording frames per second
    - save_path: the folder to save the plot that shows trace of hip point and the split timepoints
    
    Output:
    - gait_each_period_kpts: a list containing each period of linear walking bout. The shape 
    of the element in this list is [period_frame_number, keypoints_number, 3(xyz Dimensions)]
    - split_timepoints: a list containing the time points at which the gait sequence 
    is segmented into distinct linear walking bouts.
    '''
    
    frames_number = np.shape(kpts_3d)[0] - 1
    
    # calculate middle hip to split each walking period, when subejects changed their walking 
    # direction, the hip distance changing in x direction will change the sign. Thus, we can
    # find the splitting point when the hip_x_dis changing the sign
    hip_x_dis = []
    for i in range(frames_number):
        # 6 is the index of mid hip keypoint, 0 is x axis.
        # In Tulip setting, all subjects walking through x direction
        temp_dist = kpts_3d[i+1,6,0] - kpts_3d[i,6,0] 
        hip_x_dis.append(temp_dist)
    
    hip_x_dis = savgol_filter(np.array(hip_x_dis),201,2)
    split_timepoints = np.where(np.diff(np.sign(hip_x_dis)))[0]
    
    # plot to check if the split timepoints are correctly found.
    if save_path != 0:
        plt.plot(hip_x_dis)
        hip_x_dis = savgol_filter(np.array(hip_x_dis),201,2)
        plt.plot(hip_x_dis)
        split_timepoints = np.where(np.diff(np.sign(hip_x_dis)))[0]
        plt.plot(split_timepoints, hip_x_dis[split_timepoints], "xr")
        plt.savefig(os.path.join(save_path, 'split_walking_period.jpg'))
        plt.clf()
    
    
    
    # extract each linear walking bout based on the split_timepoints
    gait_each_period_kpts = []
    three_seconds = 3 * fps
    
    for i in range(len(split_timepoints)):
        if i == 0:
            if split_timepoints[i] <= three_seconds:
                continue
            elif split_timepoints[i] > three_seconds:
                temp_duration_list = [0,split_timepoints[i]]
        else:
            temp_duration_list = [split_timepoints[i-1],split_timepoints[i]]
        
        gait_each_period_kpts.append(kpts_3d[temp_duration_list[0]:temp_duration_list[1],:,:])
    
    if frames_number - split_timepoints[-1] > 400:
        gait_each_period_kpts.append(kpts_3d[split_timepoints[-1]:,:,:])


    return gait_each_period_kpts, split_timepoints


def calculate_gait_events(gait_list,fps,save_path): # using fps and -fps to do linear regression

    '''
    Input: 
    - gait_list: a list containing each period of linear walking bout. The shape 
    of the element in this list is [period_frame_number, keypoints_number, 3(xyz Dimensions)]
    - fps: recording frames per second
    - save_path: the folder to save the plot that shows traces of left&right toe&heel and the gait-event timepoints
        
    Output:
    - each_walking_period_with_info: a dictionary contains all linear walking bout information.
    The structure is in the following:
        
    each_walking_period_with_info[
        '0':[
            'event_name': all toe-offs and heel-strikes detected in '0' walking bout
            'event_timepoints': corresponding index for the event_name 
            'walking_period_data': np array about poses. Shape:[period0_frame_number, keypoints_number, 3(xyz Dimensions)]
            ]
        '1':[...]
        ...
        
        ]
    '''

    peak_detection_distance = 40 # define the ditance for find_peaks function

    
    count = 0
    each_walking_period_with_info = {}
    for walking_period in gait_list:
        each_walking_period_with_info[str(count)] = {}
        each_walking_period_with_info[str(count)]['event_name'] = []
        
        # In camera2 perspective, all subjects first start walking from the right side 
        # of the screen to the left side. Based on this, we can know which direction 
        # they are heading to.
        
        if count%2 == 0:
            direction = 'R2L'
        elif count%2 == 1:
            direction = 'L2R'
        
        # calculate linear regression
        
        # avoid some trun-around noise
        real_walking = walking_period[fps:-fps,:,:]
        
        # use linear regresion to find out the precise walking direction
        during_walking_leftheels = real_walking[:,24,:]
        during_walking_rightheels = real_walking[:,25,:]
        x_coordinates_heels = np.hstack((during_walking_leftheels[:,0],during_walking_rightheels[:,0]))
        y_coordinates_heels = np.hstack((during_walking_leftheels[:,1],during_walking_rightheels[:,1]))
        linear_model = LinearRegression().fit(x_coordinates_heels.reshape(-1,1),y_coordinates_heels)
        slope = linear_model.coef_
        intercept = linear_model.intercept_
        
        xdiff_lefttoe_hip = []
        xdiff_righttoe_hip = []
        xdiff_leftheel_hip = []
        xdiff_rightheel_hip = []
        
        # calculate the projected distance between hip and toes/heels and find peaks or valleies to 
        # find the gait events
        for frame_num in range(np.shape(walking_period)[0]):
            left_toe = walking_period[frame_num,20,:2]
            left_heel = walking_period[frame_num,24,:2]
            right_toe = walking_period[frame_num,21,:2]
            right_heel = walking_period[frame_num,25,:2]
            hip = walking_period[frame_num,19,:2]
            
            # We need to identify the extrema (maximum and minimum points) in the 
            # anterior-posterior (front-to-back) trajectories of the heels and toes 
            # relative to the hip. These extrema help define important gait events, 
            # such as toe-off and heel-strike. By analyzing the walking direction, 
            # we can determine whether the toe or heel is positioned in front of or 
            # behind the hip at any given moment.
            
            xdiff_lefttoe_hip.append(cal_projected_length(left_toe,hip,slope,intercept,direction))
            xdiff_leftheel_hip.append(cal_projected_length(left_heel,hip,slope,intercept,direction))
            xdiff_righttoe_hip.append(cal_projected_length(right_toe,hip,slope,intercept,direction))
            xdiff_rightheel_hip.append(cal_projected_length(right_heel,hip,slope,intercept,direction))
        

        xdiff_lefttoe_hip = np.array(xdiff_lefttoe_hip)
        xdiff_leftheel_hip = np.array(xdiff_leftheel_hip)
        xdiff_righttoe_hip = np.array(xdiff_righttoe_hip)
        xdiff_rightheel_hip = np.array(xdiff_rightheel_hip)

        
        left_heel_peaks, _ = find_peaks(xdiff_leftheel_hip, prominence=20,distance=peak_detection_distance)
        left_toe_valleys, _ = find_peaks(-xdiff_lefttoe_hip, prominence=20,distance=peak_detection_distance)
        right_heel_peaks, _ = find_peaks(xdiff_rightheel_hip, prominence=20,distance=peak_detection_distance)
        right_toe_valleys, _ = find_peaks(-xdiff_righttoe_hip, prominence=20,distance=peak_detection_distance)
        
        # summarize all gait events and name and save them into dictionary
        each_walking_period_with_info[str(count)]['event_timepoints'] = \
            np.sort(np.hstack((left_heel_peaks,left_toe_valleys,right_heel_peaks,right_toe_valleys)))
        for timepoint in each_walking_period_with_info[str(count)]['event_timepoints']:
            if timepoint in left_heel_peaks:
                each_walking_period_with_info[str(count)]['event_name'].append('l_heel_strike')
            elif timepoint in left_toe_valleys:
                each_walking_period_with_info[str(count)]['event_name'].append('l_toe_off')
            elif timepoint in right_heel_peaks:
                each_walking_period_with_info[str(count)]['event_name'].append('r_heel_strike')
            elif timepoint in right_toe_valleys:
                each_walking_period_with_info[str(count)]['event_name'].append('r_toe_off')
         
        each_walking_period_with_info[str(count)]['walking_period_data'] = walking_period
        count += 1  
        
        # plot to check if the gait event timepoints are correctly found.
        plt.plot(xdiff_rightheel_hip,label='rheel2hip',c='b')
        plt.plot(right_heel_peaks, xdiff_rightheel_hip[right_heel_peaks], "xb")
        plt.plot(xdiff_righttoe_hip,label='rtoe2hip',c='cyan')
        plt.plot(right_toe_valleys, xdiff_righttoe_hip[right_toe_valleys], "xc")
        
        plt.plot(xdiff_leftheel_hip,label='lheel2hip',c='g')
        plt.plot(left_heel_peaks, xdiff_leftheel_hip[left_heel_peaks], "xg")
        plt.plot(xdiff_lefttoe_hip,label='ltoe2hip',c='r')
        plt.plot(left_toe_valleys, xdiff_lefttoe_hip[left_toe_valleys], "xr")
        plt.legend(loc=1)

    
        plt.savefig(os.path.join(save_path, 'gait_events_period{}.jpg'.format(count)))
        plt.clf()
        
        
        
    return each_walking_period_with_info
            
        

def calculate_basic_features(each_walking_info_dict,fps):
    
    
    '''
    Input: 
    - each_walking_period_with_info: aa dictionary contains all linear walking bout information. 
    Generated from calculate_gait_events function.
    
    The structure is in the following:
    each_walking_period_with_info[
        '0':[
            'event_name': all toe-offs and heel-strikes detected in '0' walking bout
            'event_timepoints': corresponding index for the event_name 
            'walking_period_data': np array about poses. Shape:[period0_frame_number, keypoints_number, 3(xyz Dimensions)]
            ]
        '1':[...]
        ...
        
        ]
    - fps: recording frames per second
        
    Output:
    - basic_features_dict: a dictionary containing all gait features for this subject. Here we calculate 25 features listed 
    in the following: start with l/r means left/right. The keys are feature names and the values are lists that save all features
        
    lstep_duration, rstep_duration, lstride_duration, rstride_duration, lsingle_support_time, rsingle_support_time,
    double_support_time, lstance_time, rstance_time, lswing_time, rswing_time, cadence, lstep_length, rstep_length,
    lstride_length, rstride_length, step_width, average_velocity, lankle_angle, rankle_angle, lknee_angle, rknee_angle, 
    lhip_angle, rhip_angle, legs_angle
    '''
    

    # delete some gait events since at the start and end of a walking bout, subjects may slow down
    # which may lead to some noises in the gait features
    delete_turning_around = 2    
    
    
    ## all features
    basic_features_dict = {}
    basic_features_dict['lstep_duration'] = []
    basic_features_dict['rstep_duration'] = []
    basic_features_dict['lstride_duration'] = []
    basic_features_dict['rstride_duration'] = []
    basic_features_dict['lsingle_support_time'] = []
    basic_features_dict['rsingle_support_time'] = []
    basic_features_dict['double_support_time'] = []
    basic_features_dict['lstance_time'] = []
    basic_features_dict['rstance_time'] = []
    basic_features_dict['lswing_time'] = []
    basic_features_dict['rswing_time'] = []
    basic_features_dict['cadence'] = []
    basic_features_dict['lstep_length'] = []
    basic_features_dict['rstep_length'] = []
    basic_features_dict['lstride_length'] = []
    basic_features_dict['rstride_length'] = []
    basic_features_dict['step_width'] = []    
    basic_features_dict['average_velocity'] = []     
    
    basic_features_dict['lankle_angle'] = []     
    basic_features_dict['rankle_angle'] = []     
    basic_features_dict['lknee_angle'] = []     
    basic_features_dict['rknee_angle'] = []     
    basic_features_dict['lhip_angle'] = []     
    basic_features_dict['rhip_angle'] = []     
    basic_features_dict['legs_angle'] = []     
    
    # count = 0
    
    for walking_period_id in each_walking_info_dict:
        
        walking_period = each_walking_info_dict[walking_period_id]['walking_period_data'] # shape [n_frames,26,3]
        event_timepoints = each_walking_info_dict[walking_period_id]['event_timepoints'][delete_turning_around:-delete_turning_around]
        event_name = each_walking_info_dict[walking_period_id]['event_name'][delete_turning_around:-delete_turning_around]
        
        # calculate linear regression
        real_walking = walking_period[fps:-fps,:,:]
        during_walking_leftheels = real_walking[:,24,:]
        during_walking_rightheels = real_walking[:,25,:]
        x_coordinates_heels = np.hstack((during_walking_leftheels[:,0],during_walking_rightheels[:,0]))
        y_coordinates_heels = np.hstack((during_walking_leftheels[:,1],during_walking_rightheels[:,1]))
        linear_model = LinearRegression().fit(x_coordinates_heels.reshape(-1,1),y_coordinates_heels)
        slope = linear_model.coef_
        intercept = linear_model.intercept_
        vertical_slope = -1 / slope
        
        steps_count = 0
        steps_timepoints = []
        
        for timepoint_id, event in enumerate(event_name):
            
            # event is r-contact ############################################################################
            if event == 'r_heel_strike':
                
                # r-contact to l-leave: double support time 
                if timepoint_id+1 < len(event_name):
                    if event_name[timepoint_id+1] == 'l_toe_off':
                        basic_features_dict['double_support_time'].append((event_timepoints[timepoint_id+1] - event_timepoints[timepoint_id])/fps)
                    else:
                        print('something wrong with walking period'+walking_period_id)
                        
                # r-contact to l-contact: l setp 
                if timepoint_id+2 < len(event_name):
                    if event_name[timepoint_id+2] == 'l_heel_strike':
                        basic_features_dict['lstep_duration'].append((event_timepoints[timepoint_id+2] - event_timepoints[timepoint_id])/fps)
                        step_length = abs(cal_projected_length(walking_period[event_timepoints[timepoint_id+2],24,:2],# left heel
                                                                walking_period[event_timepoints[timepoint_id],25,:2],# right heel
                                                                slope,intercept))
                        basic_features_dict['lstep_length'].append(step_length)
                        step_width = abs(cal_projected_length(walking_period[event_timepoints[timepoint_id+2],24,:2],# left heel
                                                                walking_period[event_timepoints[timepoint_id],25,:2],# right heel
                                                                vertical_slope,intercept))
                        basic_features_dict['step_width'].append(step_width) 
                        steps_count += 1
                        steps_timepoints.append(event_timepoints[timepoint_id+2])
                        steps_timepoints.append(event_timepoints[timepoint_id])
                        
                    else:
                        print('something wrong with walking period'+walking_period_id)
                        
                # r-contact to r-leave: r stance 
                if timepoint_id+3 < len(event_name):
                    if event_name[timepoint_id+3] == 'r_toe_off':
                        basic_features_dict['rstance_time'].append((event_timepoints[timepoint_id+3] - event_timepoints[timepoint_id])/fps)
                    else:
                        print('something wrong with walking period'+walking_period_id)
                        
                # r-contact to r-contact: r stride 
                if timepoint_id+4 < len(event_name):
    
                    if event_name[timepoint_id+4] == 'r_heel_strike':
                        basic_features_dict['rstride_duration'].append((event_timepoints[timepoint_id+4] - event_timepoints[timepoint_id])/fps)
                        step_length = abs(cal_projected_length(walking_period[event_timepoints[timepoint_id+4],25,:2],# right heel
                                                                walking_period[event_timepoints[timepoint_id],25,:2],# right heel
                                                                slope,intercept))
                        basic_features_dict['rstride_length'].append(step_length)
                    else:
                        print('something wrong with walking period'+walking_period_id)
    
            # event is r-leave ############################################################################
            if event == 'r_toe_off':
                
                # r-leave to r-contact: r swing and l single support time 
                if timepoint_id+1 < len(event_name):
                    if event_name[timepoint_id+1] == 'r_heel_strike':
                        basic_features_dict['lsingle_support_time'].append((event_timepoints[timepoint_id+1] - event_timepoints[timepoint_id])/fps)
                        basic_features_dict['rswing_time'].append((event_timepoints[timepoint_id+1] - event_timepoints[timepoint_id])/fps)
                    else:
                        print('something wrong with walking period'+walking_period_id)
            
            # event is l-contact ############################################################################
            if event == 'l_heel_strike':
                
                # l-contact to r-leave: double support time 
                if timepoint_id+1 < len(event_name):
                    if event_name[timepoint_id+1] == 'r_toe_off':
                        basic_features_dict['double_support_time'].append((event_timepoints[timepoint_id+1] - event_timepoints[timepoint_id])/fps)
                    else:
                        print('something wrong with walking period'+walking_period_id)
                        
                # l-contact to r-contact: r step 
                if timepoint_id+2 < len(event_name):
                    if event_name[timepoint_id+2] == 'r_heel_strike':
                        basic_features_dict['rstep_duration'].append((event_timepoints[timepoint_id+2] - event_timepoints[timepoint_id])/fps)
                        step_length = abs(cal_projected_length(walking_period[event_timepoints[timepoint_id+2],25,:2],# right heel
                                                                walking_period[event_timepoints[timepoint_id],24,:2],# left heel
                                                                slope,intercept))
                        basic_features_dict['rstep_length'].append(step_length)
                        step_width = abs(cal_projected_length(walking_period[event_timepoints[timepoint_id+2],25,:2],# right heel
                                                                walking_period[event_timepoints[timepoint_id],24,:2],# left heel
                                                                vertical_slope,intercept))
                        basic_features_dict['step_width'].append(step_width)   
                        steps_count += 1
                        steps_timepoints.append(event_timepoints[timepoint_id+2])
                        steps_timepoints.append(event_timepoints[timepoint_id])
                    else:
                        print('something wrong with walking period'+walking_period_id)
                        
                # l-contact to l-leave: l stance 
                if timepoint_id+3 < len(event_name):
                    if event_name[timepoint_id+3] == 'l_toe_off':
                        basic_features_dict['lstance_time'].append((event_timepoints[timepoint_id+3] - event_timepoints[timepoint_id])/fps)
                    else:
                        print('something wrong with walking period'+walking_period_id)
                        
                # l-contact to l-contact: l stride 
                if timepoint_id+4 < len(event_name):
    
                    if event_name[timepoint_id+4] == 'l_heel_strike':
                        basic_features_dict['lstride_duration'].append((event_timepoints[timepoint_id+4] - event_timepoints[timepoint_id])/fps)
                        step_length = abs(cal_projected_length(walking_period[event_timepoints[timepoint_id+4],24,:2],# left heel
                                                                walking_period[event_timepoints[timepoint_id],24,:2],# left heel
                                                                slope,intercept))
                        basic_features_dict['lstride_length'].append(step_length)
                    else:
                        print('something wrong with walking period'+walking_period_id)
    
            # event is l-leave ############################################################################
            if event == 'l_toe_off':
                
                # l-leave to l-contact: l swing and r single support time 
                if timepoint_id+1 < len(event_name):
                    if event_name[timepoint_id+1] == 'l_heel_strike':
                        basic_features_dict['rsingle_support_time'].append((event_timepoints[timepoint_id+1] - event_timepoints[timepoint_id])/fps)
                        basic_features_dict['lswing_time'].append((event_timepoints[timepoint_id+1] - event_timepoints[timepoint_id])/fps)
                    else:
                        print('something wrong with walking period'+walking_period_id)
        
        
        all_steps_duration = (max(steps_timepoints) - min(steps_timepoints)) / fps
        cadence = steps_count * 60 / all_steps_duration
        basic_features_dict['cadence'].append(cadence)         
        
        real_walking_duration =  (max(event_timepoints) - min(event_timepoints)) / fps
        real_walk_length = abs(cal_projected_length(walking_period[max(event_timepoints),19,:2],# hip
                                                walking_period[min(event_timepoints),19,:2],# hip
                                                slope,intercept))
        walking_velocity = real_walk_length / real_walking_duration
        basic_features_dict['average_velocity'].append(walking_velocity) 
        
        # start calculating each angles #
        selected_walking_period = walking_period[min(event_timepoints):max(event_timepoints),:,:]
        basic_features_dict['lankle_angle'] += calculate_angle([15,13],[15,20],selected_walking_period)
        basic_features_dict['rankle_angle'] += calculate_angle([16,14],[16,21],selected_walking_period)
        basic_features_dict['lknee_angle'] += calculate_angle([13,15],[13,11],selected_walking_period)
        basic_features_dict['rknee_angle'] += calculate_angle([14,16],[14,12],selected_walking_period)
        basic_features_dict['lhip_angle'] += calculate_angle([11,13],[11,100],selected_walking_period) # 100 means z-vector
        basic_features_dict['rhip_angle'] += calculate_angle([12,14],[12,100],selected_walking_period) # 100 means z-vector
        basic_features_dict['legs_angle'] += calculate_angle([11,13],[12,14],selected_walking_period)

        
    return basic_features_dict






if __name__ == '__main__':
    
    # define the path of subjects poses pkl file, figure save path and feature save path
    base_dir = os.getcwd()
    
    subjects_poses_path = 'ThreeSubjects_GaitPoses.pkl'
    validation_figures_folder_path = os.path.join(base_dir,'validation_figures/')
    feature_save_path =  os.path.join(base_dir,'ThreeSubjects_GaitFeatures.pkl')
   

    
    # define the fps of the recording, fps will be used to calculate temporal features
    fps = 80
    
    # load subjects 3d poses
    with open(subjects_poses_path, 'rb') as f:
        subjects_poses = pickle.load(f)
    
    # pre-define the shape of the final feature matrix
    gait_features_dict = {}
    
    # start loop to calculate gait features one by one
    for count, sub_id in enumerate(subjects_poses.keys()):
        
        # create a folder to save all validation figures for each step
        temp_saving_path = os.path.join(validation_figures_folder_path, 'sub{}/'.format(sub_id))
        if not os.path.exists(temp_saving_path):
            os.makedirs(temp_saving_path)
        
        temp_sub_pose = subjects_poses[sub_id]
        
        # smooth 3d poses
        kpts_3d_smoothed = smooth_3d_filter(temp_sub_pose)
        
        # splitting linear gait bout from the continuous cycle walking 
        splited_gait, spliting_timepoints = split_walking_period(kpts_3d_smoothed,fps,temp_saving_path) 
        
        # extract gait events from each linear gait bout
        each_walking_period_with_info = calculate_gait_events(splited_gait,fps,temp_saving_path)
        
        # calculate all gait features based on gait events
        basic_features = calculate_basic_features(each_walking_period_with_info,fps)
        
        # save features into a dict
        gait_features_dict[sub_id] = basic_features
    
    # save the gait feautres dict
    with open(feature_save_path,'wb') as f:
        pickle.dump(gait_features_dict,f)
        

    





