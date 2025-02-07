# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:41:06 2024

@author: zp20945
"""

def append_based_on_last_values(color_list, threshold=0.01, factor_same=0.90, factor_diff=0.80):
    """
    Appends a value to the given color list based on whether the last three values are approximately the same.
    
    Parameters:
    color_list (list): The list of color values.
    threshold (float): The threshold for considering values "approximately the same".
    factor_same (float): The factor to multiply by if the values are the same.
    factor_diff (float): The factor to multiply by if the values are different.
    """
    if len(color_list) >= 3:
        last_three = color_list[-3:]
        # Check if the last three values are approximately the same
        if abs(last_three[0] - last_three[1]) < threshold and abs(last_three[1] - last_three[2]) < threshold:
            color_list.append(factor_same * last_three[0])
        else:
            color_list.append(factor_diff * last_three[0])
    else:
        color_list.append(factor_diff * color_list[0])