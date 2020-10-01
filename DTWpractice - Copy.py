#!/usr/bin/env python
import csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline



#first I want to take the two signals and write them to arrays in python that I can apply the DTW to
#to do this to the the csv I will import the CH2 column from both of the files and then run my analysis

csv = np.genfromtxt('Fadi0.csv', delimiter = ",", header = 0)
fadi1 = csv[:,1]

csv = np.genfromtxt('Fadi4.csv', delimiter = ",", header = 0)
fadi2 = csv[:,1]

# I now have the data in my two np arrays we can now plot them to see wat they will look like
plt.plot(fadi1, 'r', label = 'Fadi 0')
plt.plot(fadi2, 'g', label = 'Fadi 4')

#From here we want to see if they are similar.  We will do this by creating a mapping of between all of the points in the tow signals
#to compute the distance e have to make a 2d matrix
distances = np.zeros((len(fadi2), len(fadi1)))

for i in range(len(fadi2)):
    for j in range(len(fadi1)):
        distances[i,j] = (fadi1[j]-fadi2[i])**2  
#for this distance we can choose between a couple different options but we will opt to use the Euclidean distance
#we now want to plot the distance that we just calculted and we can use a heat map to do this

def distance_cost_plot(distances):
    im = plt.imshow(distances, interpolation='nearest', cmap='Reds')
    plt.gca().invert_yaxis()
    plt.xlabel("Fadi1")
    plt.ylabel("Fadi4")
    plt.grid()
    plt.colorbar()


distance_cost_plot(distances)


accumulated_cost = np.zeros((len(fadi2), len(fadi1)))
accumulated_cost[0,0] = distances[0,0]
#distance_cost_plot(accumulated_cost)

for i in range(1, len(fadi1)):
    accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]

#distance_cost_plot(accumulated_cost)

for i in range(1, len(fadi2)):
    accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0]  

#distance_cost_plot(accumulated_cost)

for i in range(1, len(fadi2)):
    for j in range(1, len(fadi1)):
        accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]

distance_cost_plot(accumulated_cost)


path = [[len(fadi1)-1, len(fadi2)-1]]
i = len(fadi2)-1
j = len(fadi1)-1
while i>0 and j>0:
    if i==0:
        j = j - 1
    elif j==0:
        i = i - 1
    else:
        if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
            i = i - 1
        elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
            j = j-1
        else:
            i = i - 1
            j= j- 1
    path.append([j, i])
path.append([0,0])

path

path_x = [point[0] for point in path]
path_y = [point[1] for point in path]
distance_cost_plot(accumulated_cost)
plt.plot(path_x, path_y)

def path_cost(fadi1, fadi2, accumulated_cost, distances):
    path = [[len(fadi1)-1, len(fadi2)-1]]
    cost = 0
    i = len(fadi2)-1
    j = len(fadi1)-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j= j- 1
        path.append([j, i])
    path.append([0,0])
    for [fadi2, fadi1] in path:
        cost = cost +distances[fadi1, fadi2]
    return path, cost 

path, cost = path_cost(fadi1, fadi2, accumulated_cost, distances)
print(path)
print(cost)

#this is an implementation that we have created for this problem but we can also try it using a library that pythn has ad see the difference

#attempt using the mlpy library

import mlpy

dist, cost, path = mlpy.dtw_std(fadi1, fadi2, dist_only = False)

import matplotlib.cm as cm
fig = plt.figure(1)
ax = fig.add_subplor(11)
plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plot2= plt.plot(path[0], path[1], 'w')
xlim = ax.set_xlim((-.05, cost.shape[0]-.05))
ylim = ax.set_ylim((-0.5, cost.shape[1]-0.5))

dist

plt.plot(fadi1, 'bo-' ,label='Fadi 1')
plt.plot(fadi2, 'g^-', label = 'Fadi 4')
plt.legend()
paths = path_cost(fadi1, fadi2, accumulated_cost, distances)[0]
for [map_x, map_y] in paths:
    print map_x, fadi1[map_x], ":", map_y, fadi2[map_y]
    
    plt.plot([map_x, map_y], [fadi1[map_x], fadi2[map_y]], 'r')

idx = np.linspace(0, 6.28, 100)
fadi1 = np.sin(idx)
fadi2 = np.cos(idx)
distances = np.zeros((len(fadi2), len(fadi1)))

for i in range(len(fadi2)):
    for j in range(len(fadi2)):
        distances = np.zeros((len(fadi2), len(fadi1)))

distances_cost_plot(distances)

accumulated_cost = np.zeros((len(fadi2), len(fadi1)))
accumulated_cost[0,0] = distances[0,0]
for i in range(1, len(fadi2)):
    accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0]
for i in range(1, len(fadi1)):
    accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1] 
for i in range(1, len(fadi2)):
    for j in range(1, len(fadi1)):
        accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]z




plt.plot(fadi1, 'bo-' ,label='Fadi 1')
plt.plot(fadi2, 'g^-', label = 'Fadi 4')
plt.legend()
paths = path_cost(fadi1, fadi2, accumulated_cost, distances)[0]
for [map_x, map_y] in paths:
    #print map_x, fadi1[map_x], ":", map_y, fadi2[map_y]
    
    plt.plot([map_x, map_y], [fadi1[map_x], fadi2[map_y]], 'r')

