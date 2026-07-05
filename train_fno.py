import torch
import scipy.io
import numpy as np

print("1. Loading the dataset...")
data = scipy.io.loadmat('matlab/thermal_dataset.mat')

k_s_array = data['k_s'].flatten() 
k_c_array = data['k_c'].flatten() 
T_data = data['T_data']           

num_samples = len(k_s_array)

print("2. Painting the Input Images (Conductivity Maps)...")
input_images = np.zeros((num_samples, 64, 64))

for i in range(num_samples):
    input_images[i, :, :] = k_c_array[i]

    input_images[i, 16:48, 16:48] = k_s_array[i]

T_data = np.transpose(T_data, (2, 0, 1))

inputs = torch.tensor(input_images, dtype=torch.float32).unsqueeze(1)
outputs = torch.tensor(T_data, dtype=torch.float32).unsqueeze(1)

print(f"Final Input shape: {inputs.shape}")
print(f"Final Output shape: {outputs.shape}")
