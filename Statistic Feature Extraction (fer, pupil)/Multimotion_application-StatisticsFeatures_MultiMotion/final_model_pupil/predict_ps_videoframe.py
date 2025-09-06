# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:17:18 2024

@author: zeelp
"""

from final_model_pupil import Find_lux as FL
from final_model_pupil import Normal_RGB_model as RGB_N

def predict_ps(results_df, popt_r, popt_g, popt_b, popt_w, file_path):
    
    color_based, grey_based, r_ps, g_ps, b_ps, lux = [], [], [], [], [], []
    
    # Calculate lux and pupil size values
    for rgb in results_df['RGB']:
        R, G, B = rgb
        
        # Lux values for individual color channels
        lux_value_r = FL.find_lux_from_file(R, 0, 0, file_path)
        lux_value_g = FL.find_lux_from_file(0, G, 0, file_path)
        lux_value_b = FL.find_lux_from_file(0, 0, B, file_path)
    
        
    
        # Grey-based calculation
        lux_value = FL.find_lux_from_file(R, G, B, file_path)
        
        lux.append(lux_value)
    
        # Calculate color-based pupil size
        Total = R + G + B
        if Total != 0:
            contro_r = R / Total
            contro_g = G / Total
            contro_b = B / Total
            
            
            # Pupil size for each color
            ps_r = RGB_N.objective(lux_value_r, *popt_r)
            ps_g = RGB_N.objective(lux_value_g, *popt_g)
            ps_b = RGB_N.objective(lux_value_b, *popt_b)
            
            #colour based
            Total_ps = (ps_r * contro_r) + (ps_g * contro_g) + (ps_b * contro_b)
            
            
            
            
        else:
            # Pupil size for each color
            ps_r = RGB_N.objective(0, *popt_w)
            ps_g = RGB_N.objective(0, *popt_w)
            ps_b = RGB_N.objective(0, *popt_w)
            
            
            # color based
            Total_ps = RGB_N.objective(0, *popt_w)
            
        r_ps.append(ps_r)
        g_ps.append(ps_g)
        b_ps.append(ps_b)
        color_based.append(Total_ps)    
            
        grey_based.append(RGB_N.objective(lux_value, *popt_w))
            
            
    
        
    
        
        #print(R,G,B, (RGB_N.objective(lux_value, *popt_w)), ps_r, ps_g, ps_b)
    
    results_df['color_based'] = color_based
    results_df['grey_based'] = grey_based
    results_df['Lux'] = lux
    results_df['r_ps'] = r_ps
    results_df['g_ps'] = g_ps
    results_df['b_ps'] = b_ps
   
    
    return results_df, (RGB_N.objective(lux_value, *popt_w))