'''
I will turn this into a function. I'll call the fn
9 times to get the data for each of the 9 points
To do: Get rid out outliers and calculate the avg
gaze angles angle_x and angle_y for each of the points.

'''
'''
1) Setup file to get actual averages
2) Call calibrate from main.py
3) Now draw over an image.
Then convert coords to pixel vals or smth
ThenDraw heatmap?
Compare results
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import statistics

def data_preprocess():
    # Take in gaze angle x and y from main
    SLICE_NUM = 5
    path = 'GAZE-TRACKING-TRACKEREYES\calibration'
    extension = 'csv'
    os.chdir(path)
    print (path)
    result = glob.glob('*.{}'.format(extension))
    # Result holds names of all csv files
    #os.chdir('../')
    os.getcwd()
    # Remember to do os.chdir('../') to get back to Gaze-Estimation2 directory
    # If putting this in actual program

    # Point is a list storing gaze_angle values for all 9 points
    point = []
    for i in range(9):
        point.append(get_csv(result[i]))

        # Get averages gaze angles for each point
        # Get rid of first and last 5 co-ords 
        averages = []
    for j, _ in enumerate(point):
        
        averages.append(average(point[j],SLICE_NUM))
    np_averages= np.array([np.array(xi) for xi in averages])  
    os.chdir('../')
    #plot_scatter(point)
    return np_averages

def return_coordinates(np_averages,gaze_angle_x, gaze_angle_y):   
    # Get it into numpy array    
    # 2-D array of normalised screen coordinates
    point_coords= np.array([[0.1,0.1], [0.5,0.1], [0.9,0.1],
    [0.1,0.5], [0.5, 0.5], [0.9, 0.5],
    [0.1,0.9], [0.5,0.9], [0.9,0.9]])#

    # Full quadrant

    g22,x22 = get_gaze_and_coords(np_averages,point_coords,0,0,2 ,0)
    g23,y23 = get_gaze_and_coords(np_averages,point_coords,2,1,8,1)

    g22 = np.flip(g22, 0)
    g23 = np.flip(g23, 0)
    #print(g22)
    x1 = np.interp(gaze_angle_x, g22,x22)
    y1 = np.interp(gaze_angle_y, g23,y23)
    #print("Coords",x1,y1)
    coords = x1,y1
    # x1,y1 are screen coordinates
    return(x1,y1)

def average(lst,num):
    # Slice the list for starting and ending program
    lst = lst[num:-num]
    return sum(lst) / len(lst)

def get_gaze_and_coords(np_averages,point_coords,i,j,k,l):
        x1 = [np_averages[i][j],np_averages[k][l]]
        x2= [point_coords[i][j],point_coords[k][l]]
        return x1,x2
        

def get_csv(filename):
    # Reminding myself of fieldnames in csv file
    field_names = (['Point','Frame','Time','Gaze_angle_x','Gaze_angle_y'])
    # Now get columns as lists
    data = pd.read_csv(filename, header=0)

    # Now get columns as lists
    point = list(map(int,data.Point))
    frame_num = list(map(int,data.Frame))
    time = list(map(float, data.Time))
    gaze_x = list(map(float, data.Gaze_angle_x))
    gaze_y = list(map(float, data.Gaze_angle_y))
    gaze_angles= np.column_stack((gaze_x, gaze_y))

    x = gaze_x
    y = gaze_y
    H = gaze_angles
    return H



def plot_scatter(list):
    count=1
    plt.subplots()
    for i in list:
        
        plt.scatter(i[:,0], i[:,1])
        plt.xlabel('Gaze angle x(rad)') 
        plt.ylabel('Gaze angle y(rad)') 
        plt.title("Gaze Angles for nine Calibration oints ")
        plt.show
        count+=1
        i = i+3

    plt.show()
np_avgs = data_preprocess()

def plot_scatter1(list): 
    plt.scatter(list[:,0], list[:,1])
    plt.legend()
    plt.show()

'''
coord_x,coord_y = return_coordinates(np_avgs,0.01, -0.38)
eye0 = get_csv("calibration/point_1.csv")
print(eye0[:10])
plot_scatter1(eye0)
print(np_avgs)
'''