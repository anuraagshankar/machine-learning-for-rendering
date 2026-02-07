from AppRenderer import scene, DIRECTORY, FILENAME
from PyRT_Integrators import CMCIntegrator
import cv2
import numpy as np
import os

n_values = [10, 50, 100, 500]

for n in n_values:
    print(f"Generating image for n = {n}")
    if not os.path.exists(f"out/cmc_MC_{n}_samples.png"):
        integrator = CMCIntegrator(n, DIRECTORY + FILENAME)
        integrator.add_scene(scene)
        integrator.render()

def rmse_error(image_path_1, image_path_2):
    image_1 = cv2.imread(image_path_1)
    image_2 = cv2.imread(image_path_2)
    
    # Calculate per-pixel squared error across all channels
    squared_error = (image_1.astype(np.float32) - image_2.astype(np.float32)) ** 2
    
    # Mean across channels to get single-channel error map
    mse_map = np.mean(squared_error, axis=2)
    
    # Take square root to get RMSE per pixel
    rmse_map = np.sqrt(mse_map)
    
    # Min-max scaling to 0-255
    min_val = rmse_map.min()
    max_val = rmse_map.max()
    error_image = rmse_map.astype(np.uint8)
    
    # Overall RMSE
    error = np.sqrt(np.mean(squared_error))
    
    return error, error_image

ground_truth_image = "out/cmc_MC_1000_samples.png"

for n in n_values:
    image_path = f"out/cmc_MC_{n}_samples.png"
    error, error_image = rmse_error(ground_truth_image, image_path)
    print(f"n = {n}, error = {error}")
    
    # Apply colormap (hot, jet, or turbo work well for error visualization)
    error_colored = cv2.applyColorMap(error_image, cv2.COLORMAP_JET)
    
    # Save the error image
    cv2.imwrite(f"out/error_{n}_samples.png", error_colored)
    
    cv2.imshow("Error Image", error_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
