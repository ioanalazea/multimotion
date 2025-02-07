# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:35:23 2024

@author: zeelp
"""
import numpy as np
import random
import cv2


def get_10_points_average_rgb_within_circle(img, x_center, y_center, radius):
    # Convert the image to a NumPy array for faster access
    img_array = np.array(img)
    image_height, image_width = img_array.shape[:2]

    # Create a meshgrid of x, y coordinates for the bounding box of the circle
    y_grid, x_grid = np.ogrid[max(0, y_center - radius):min(image_height, y_center + radius),
                              max(0, x_center - radius):min(image_width, x_center + radius)]
    
    # Compute the distance from the center for each point in the grid
    dist_from_center = (x_grid - x_center) ** 2 + (y_grid - y_center) ** 2
    
    # Create a mask for pixels inside the circle
    mask = dist_from_center <= radius ** 2

    # Apply the mask to get the valid pixels inside the circle
    valid_pixels = img_array[max(0, y_center - radius):min(image_height, y_center + radius),
                             max(0, x_center - radius):min(image_width, x_center + radius)][mask]

    
            # Extract R, G, B values and compute their means
    r_avg = np.mean(valid_pixels[:, 0])
    g_avg = np.mean(valid_pixels[:, 1])
    b_avg = np.mean(valid_pixels[:, 2])

    return r_avg, g_avg, b_avg

# Optimized function to get the average RGB values within a circular region
def get_points_average_rgb_within_circle(img_array, x_center, y_center, radius, num_points=50):
    image_height, image_width = img_array.shape[:2]
    
    # Lists to store the RGB values
    r_values, g_values, b_values = [], [], []
    
    # Generate random points within the bounding box of the circle
    count = 0
    while count < num_points:
        # Generate random points in the bounding box
        x = random.randint(max(0, x_center - radius), min(image_width - 1, x_center + radius))
        y = random.randint(max(0, y_center - radius), min(image_height - 1, y_center + radius))
        
        # Check if the point (x, y) is inside the circle
        if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
            r, g, b = img_array[y, x, :3]  # Directly access the RGB values using NumPy array
            r_values.append(r)
            g_values.append(g)
            b_values.append(b)
            count += 1

    # Compute the mean R, G, B values
    r_avg = np.mean(r_values) if r_values else 0
    g_avg = np.mean(g_values) if g_values else 0
    b_avg = np.mean(b_values) if b_values else 0

    
    
    
    return r_avg, g_avg, b_avg

# Optimized function to compute average RGB for the entire image
def get_average_rgb(img_array):
    # Calculate the average directly from the NumPy array
    r_avg, g_avg, b_avg = np.mean(img_array[:, :, :3], axis=(0, 1))
    return r_avg, g_avg, b_avg
