import scipy.io
import numpy as np
import matplotlib.pyplot as plt

data = scipy.io.loadmat('matlab/thermal_dataset.mat')
k_s_array = data['k_s'].flatten() 
k_c_array = data['k_c'].flatten() 
T_data = data['T_data']
T_data = np.transpose(T_data, (2, 0, 1))

input_image = np.zeros((64, 64))
input_image[:, :] = k_c_array[0]         
input_image[16:48, 16:48] = k_s_array[0]

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

im1 = axs[0].imshow(input_image, cmap='magma')
axs[0].set_title('Our Input: Conductivity Map')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(T_data[0], cmap='inferno')
axs[1].set_title('Our Output: MATLAB Temperature')
fig.colorbar(im2, ax=axs[1])

plt.show()
