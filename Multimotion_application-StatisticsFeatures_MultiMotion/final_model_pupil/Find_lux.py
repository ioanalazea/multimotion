# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:56:35 2024

@author: zp20945
"""

import pandas as pd
import bisect
from itertools import product

def find_lux_from_file(R, G, B, file_path):
    df = pd.read_csv(file_path)

    lux_poss = []
    close = []
    mult = []
    lux_array = []
    closeness = []
    
    array_1 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    array_2 = [65, 75, 85, 95]
    array_3 = [0, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    array_4 = [0, 10, 20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    array_5 = [0, 60, 70, 80, 90, 100]

    def find_interval(z, array):
        for i in range(len(array)):
            if z not in array:
                if array[i] <= z <= array[i + 1]:
                    return array[i], array[i + 1]
            elif z == array[i]:
                return array[i]

    red_poss = find_interval(R, array_4)
    green_poss = find_interval(G, array_4)
    blue_poss = find_interval(B, array_4)
    #print(red_poss)
    #print(green_poss)
    #print(blue_poss)

    def find_lux(filtered_df, x, y, z):
        if not filtered_df.empty:
            for index, row in filtered_df.iterrows():
                lux_poss.append(row['lux'])
                distance_red = (x - combination[0])**4
                distance_green = (y - combination[1])**4
                distance_blue = (z - combination[2])**4
                total_distance = (distance_red + distance_green + distance_blue)
                if total_distance != 0:
                    closeness = 1 / total_distance
                    close.append(closeness)
                    mult.append(closeness * row['lux'])
                else:
                    close.append(0)
            return lux_poss, mult, close

    if isinstance(red_poss, int):
        red_poss_combinations = [red_poss]
    else:
        red_poss_combinations = red_poss

    if isinstance(green_poss, int):
        green_poss_combinations = [green_poss]
    else:
        green_poss_combinations = green_poss

    if isinstance(blue_poss, int):
        blue_poss_combinations = [blue_poss]
    else:
        blue_poss_combinations = blue_poss

    combinations = list(product(red_poss_combinations, green_poss_combinations, blue_poss_combinations))

    fina_comb = []

    for combination in combinations:
        combination = list(combination)
        if combination[0] in array_2:
            
            if combination[1] in array_3:
                
                if combination[2] in array_3:
                    fina_comb.append(combination)
                else:
                
                    insertion_point = bisect.bisect_right(array_1, combination[0])
                    combination[0] = array_1[insertion_point - 1]
                    insertion_point = bisect.bisect_right(array_1, combination[1])
                    combination[1] = array_1[insertion_point - 1]
                    fina_comb.append(combination)
            else:
            
                insertion_point = bisect.bisect_right(array_1, combination[0])
                combination[0] = array_1[insertion_point - 1]
                insertion_point = bisect.bisect_right(array_1, combination[1])
                combination[1] = array_1[insertion_point - 1]
                insertion_point = bisect.bisect_right(array_1, combination[2])
                combination[2] = array_1[insertion_point - 1]
                fina_comb.append(combination)
                
        elif combination[0] in array_5:
            
            if combination[2] != 0:
                if combination[1] <= 60:
                    #print(2)
                    insertion_point = bisect.bisect_right(array_1, combination[1])
                    combination[1] = array_1[insertion_point - 1]
                    insertion_point = bisect.bisect_right(array_1, combination[2])
                    combination[2] = array_1[insertion_point - 1]
                    fina_comb.append(combination)
                else:
                    #print(3)
                    if combination[2] >= 60:
                        fina_comb.append(combination)
                    else:
                        insertion_point = bisect.bisect_right(array_1, combination[1])
                        combination[1] = array_1[insertion_point - 1]
                        fina_comb.append(combination)
            else:
                fina_comb.append(combination)
        else:
            #print(3)
            insertion_point = bisect.bisect_right(array_1, combination[1])
            combination[1] = array_1[insertion_point - 1]
            insertion_point = bisect.bisect_right(array_1, combination[2])
            combination[2] = array_1[insertion_point - 1]
            fina_comb.append(combination)
    #print(fina_comb)        
    unique_combination = [list(comb) for comb in set(tuple(row) for row in fina_comb)]
    #print(unique_combination)
    for combination in unique_combination:
        #print(combination)
        filtered_df = df[(df["R_%"] == combination[0]) & (df["G_%"] == combination[1]) & (df["B_%"] == combination[2])]
        lux_poss, lux_array, closeness = find_lux(filtered_df, R, G, B)

    if sum(closeness) != 0:
        lux = sum(lux_array) / sum(closeness)
    else:
        lux = sum(lux_poss) / len(lux_poss)
    return lux

"""
# Example usage:
R = 0
G = 0
B = 68
file_path = 'C:/Users/zp20945/OneDrive - University of Essex/luminance/Excel_sheet/lux/important_files/all_pssobilities_lux_v4.csv'
lux_value = find_lux_from_file(R, G, B, file_path)
print("Lux value:", lux_value)
"""
"""
R = 64
G = 0
B = 0
file_path = 'C:/Users/zp20945/OneDrive - University of Essex/luminance/Excel_sheet/lux/important_files/all_pssobilities_lux_color.csv'
df = pd.read_csv(file_path)
df = df[(df["R_%"] == R) & (df["G_%"] == G) & (df["B_%"] == B)]
print("Lux value:", df['lux'])
"""
