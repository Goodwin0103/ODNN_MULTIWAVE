# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:16:56 2025

@author: zhang
修改: 移除对 scipy 的依赖
"""
import os
from datetime import datetime
import numpy as np
import torch

def save_to_mat_MC(save_dir,
                mode_classification,
                num_modes,
                test_dataset,
                visibility_value,
                temp_model,
                temp_E,
                propagated_fields,
                distance_layers,
                pixel_size,
                distance_propagation,
                wavelength,
                training_loss,
                distance_first_layer=0,
                field_size=50,
                focus_radius=5,
                detectsize=15,
                epochs=1000):
    """
    Save optical simulation data and training results to a .npy file.
    (Modified to remove scipy dependency)

    Includes:
    - temp_model: phase masks (mask_0, mask_1, ...)
    - temp_test_data: input field
    - propagation_process: list of propagated fields
    - model_parameters: geometry, pixel size, wavelength, etc.
    - training_loss: loss curve during training
    """

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    filename = (
        f"{mode_classification}_M{num_modes}_{len(temp_model)}layers_"
        f"{test_dataset}_{visibility_value:.4f}_{timestamp}"
    )
    filepath = os.path.join(save_dir, filename)

    # Convert phase masks
    model_dict = {}
    for i_layer, mask in enumerate(temp_model):
        if isinstance(mask, np.ndarray):
            model_dict[f'mask_{i_layer}'] = mask.astype(np.float32)
        else:
            print(f"[Warning] temp_model[{i_layer}] is not a numpy array. Skipped.")

    # Convert propagated fields
    prop_dict = {}
    for i_field, field in enumerate(propagated_fields):
        if isinstance(field, torch.Tensor):
            prop_dict[f'field_{i_field}'] = field.detach().cpu().numpy()
        else:
            print(f"[Warning] propagated_fields[{i_field}] is not a torch.Tensor. Skipped.")

    # Convert loss list to numpy
    loss_array = np.array(training_loss, dtype=np.float32)

    # Model/physical parameters
    param_dict = {
        'distance_first_layer': distance_first_layer,
        'distance_layers': distance_layers,
        'distance_propagation': distance_propagation,
        'pixel_size': pixel_size,
        'wavelength': wavelength,
        'field_size': field_size,
        'focus_radius': focus_radius,
        'detectsize': detectsize,
        'epochs': epochs
    }

    # Combine all data into one dictionary
    all_data = {
        'temp_model': model_dict,
        'temp_test_data': temp_E.detach().cpu().numpy() if isinstance(temp_E, torch.Tensor) else temp_E,
        'propagation_process': prop_dict,
        'model_parameters': param_dict,
        'training_loss': loss_array
    }

    # Save as .npy file (instead of .mat)
    np.save(filepath + ".npy", all_data)
    
    # Also save as PyTorch file for any tensor data
    torch_data = {}
    if isinstance(temp_E, torch.Tensor):
        torch_data['temp_test_data'] = temp_E
    
    # Add any other tensor data that might be useful
    if torch_data:
        torch.save(torch_data, filepath + ".pt")

    print(f"✅ Data saved: {filepath}.npy (替代 .mat 格式)")
    return filepath
