#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:16:23 2024

@author: sihan

This demo code is for plotting angles between two legs during gait in Tulip setting experiment. 
You can find more information in Tulip project page 
https://www.tulipproject.net/

All the keypoints format is halpe26, which you can find detailed information in the github page
https://github.com/Fang-Haoshu/Halpe-FullBody
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from extract_features import smooth_3d_filter, split_walking_period, calculate_gait_events, calculate_angle



if __name__ == '__main__':
    
    base_dir = os.getcwd()
    
    # define the path of subjects poses pkl file, figure save path and feature save path
    subjects_poses_path = 'ThreeSubjects_GaitPoses.pkl'
    plt_save_path = os.path.join(base_dir,'angles_visualization.jpg')

    # define the fps of the recording, fps will be used to calculate temporal features
    fps = 80
    
    # load subjects 3d poses
    with open(subjects_poses_path, 'rb') as f:
        subjects_poses = pickle.load(f)
    
    # define the angle
    angle_point1 = [11,13] # refer to the left leg
    angle_point2 = [12,14] # refer to the right leg
    
    score0 = [7] # sub7 gait UPDRS score is 0
    score1 = [8] # sub8 gait UPDRS score is 1
    score2 = [13] # sub13 gait UPDRS score is 2
    
    color_list = [
    [53/255,163/255,223/255],
    [112/255,181/255,52/255],
    [181/255,91/255,52/255]   
    ]
    
    fig = plt.figure()
    
    for group_id in range(3):
        if group_id == 0:
            the_group = score0
            group_label = 'Score 0'
        elif group_id == 1:
            the_group = score1
            group_label = 'Score 1'
        elif group_id == 2:
            the_group = score2
            group_label = 'Score 2'
            
        right_stride_gait = []
        left_stride_gait = []
        
        for sub_id in the_group:
            print('Processing group {} subject {}'.format(group_label,sub_id))
        
            kpts_3d = subjects_poses[str(sub_id)]
            kpts_3d_smoothed = smooth_3d_filter(kpts_3d) # this step is smooth
            splited_gait, spliting_timepoints = split_walking_period(kpts_3d_smoothed,fps,0) # this step is split each walking period
            each_walking_period_with_info = calculate_gait_events(splited_gait,fps,0) # this step is calculating each gait event
            
            delete_turning_around = 2    # delete first 2 and last 2 events to avoid incluing turning aroung affect the data
        
            for walking_period_id in each_walking_period_with_info:
                
                walking_period = each_walking_period_with_info[walking_period_id]['walking_period_data'] # shape [n_frames,26,3]
                event_timepoints = each_walking_period_with_info[walking_period_id]['event_timepoints'][delete_turning_around:-delete_turning_around]
                event_name = each_walking_period_with_info[walking_period_id]['event_name'][delete_turning_around:-delete_turning_around]
                
                # calculate linear regression
                real_walking = walking_period[fps:-fps,:,:] # avoid turning around
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
                        # r-contact to r-contact: r stride 
                        if timepoint_id+4 < len(event_name):
                            if event_name[timepoint_id+4] == 'r_heel_strike':
                                ###### calculate angles here
                                period_angle = calculate_angle(angle_point1,angle_point2,\
                                                               walking_period[event_timepoints[timepoint_id]:event_timepoints[timepoint_id+4],:,:])
                                right_stride_gait.append(period_angle)
        
                            else:
                                print('something wrong with walking period'+walking_period_id)
            
                    # event is l-contact ############################################################################
                    if event == 'l_heel_strike':
                        # l-contact to l-contact: l stride 
                        if timepoint_id+4 < len(event_name):
                            if event_name[timepoint_id+4] == 'l_heel_strike':
                                period_angle = calculate_angle(angle_point1,angle_point2,\
                                                               walking_period[event_timepoints[timepoint_id]:event_timepoints[timepoint_id+4],:,:])
                                left_stride_gait.append(period_angle)
                            else:
                                print('something wrong with walking period'+walking_period_id)
    
    
        # resample all angle_period length into 100
        after_norm = []
        for angle_period in left_stride_gait: #  left_stride_gait or right_stride_gait
            angle_period_after_norm = []
            for i in range(100):
                temp_percent = len(angle_period) * i / 100
                temp_value = angle_period[round(temp_percent)]
                angle_period_after_norm.append(temp_value)
            after_norm.append(angle_period_after_norm)   

        # reshape the list to a 2d np array
        angle_array = np.zeros((100,len(after_norm)))
        for j, angle_period in enumerate(after_norm):
            angle_array[:,j] = np.array(angle_period)
        
        # calculate the mean and 95% confidence interval on each percentage
        mean = []
        ci_95 = []
        for i in range(100):
            temp_mean = np.mean(angle_array[i,:])
            mean.append(temp_mean)
            temp_std = np.std(angle_array[i,:])
            ci_95.append(1.96*temp_std/np.sqrt(len(after_norm)))
        mean = np.array(mean)
        ci_95 = np.array(ci_95)
        
        
        # plot the angles
        plt.plot(mean,label=group_label,color=color_list[group_id])
        plt.fill_between(range(100), mean+ci_95, mean-ci_95,alpha=0.5,color=color_list[group_id])

    plt.legend(loc='lower right')
    plt.ylabel('Angles [°]')
    plt.xlabel('Percentage of Gait Cycle [%]')
    plt.savefig(plt_save_path)
    

        
        

    





