import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from fno_model import FNO2d

print("Loading data and model...")
data_mat = scipy.io.loadmat('matlab/thermal_dataset.mat')
k_s_array = data_mat['k_s'].flatten() 
k_c_array = data_mat['k_c'].flatten() 
T_data = data_mat['T_data']           

# Paint the 500 input images
input_images = np.zeros((500, 64, 64))
for i in range(500):
    input_images[i, :, :] = k_c_array[i]
    input_images[i, 16:48, 16:48] = k_s_array[i]

T_data = np.transpose(T_data, (2, 0, 1))

# Convert to tensors
inputs = torch.tensor(input_images, dtype=torch.float32).unsqueeze(1)
outputs = torch.tensor(T_data, dtype=torch.float32).unsqueeze(1)

# NEW: Apply Normalization
norm_stats = torch.load('normalization_stats.pt')
in_mean = norm_stats['in_mean']
in_std = norm_stats['in_std']
out_mean = norm_stats['out_mean']
out_std = norm_stats['out_std']

norm_inputs = (inputs - in_mean) / in_std

# Load the trained AI Brain
model = FNO2d(modes=12, width=32)
model.load_state_dict(torch.load('fno_model_weights.pth'))
model.eval()

my_data = np.load('loss_history.npy')
plt.plot(my_data)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Convergence')
plt.yscale('log')
plt.savefig('convergence_plot.png')
plt.clf()

import time

# Predict on the last 100 test samples using NORMALIZED inputs
start_time = time.perf_counter()
with torch.no_grad():
    norm_preds = model(norm_inputs[-100:])
end_time = time.perf_counter()

print(f"Average Inference Time per Sample: {(end_time - start_time) / 100:.6f} seconds")

# UN-NORMALIZE to get real temperatures back!
test_preds = (norm_preds * out_std) + out_mean
test_preds = test_preds.numpy()
test_truth = outputs[-100:].numpy()

true_max = test_truth.reshape(100, -1).max(axis=1)
pred_max = test_preds.reshape(100, -1).max(axis=1)

min_val = min(true_max.min(), pred_max.min()) - 1
max_val = max(true_max.max(), pred_max.max()) + 1

mae = np.mean(np.abs(true_max - pred_max))
max_err = np.max(np.abs(true_max - pred_max))
ss_res = np.sum((true_max - pred_max)**2)
ss_tot = np.sum((true_max - np.mean(true_max))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R^2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Max Error: {max_err:.4f}")

plt.scatter(true_max, pred_max, color='blue', alpha=0.6)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title('Max Temperature: MATLAB vs. AI')
plt.xlabel('True MATLAB Max Temp')
plt.ylabel('AI Predicted Max Temp')
plt.savefig('parity_plot.png')
plt.clf()

# 1. Pick a single sample (the very last one)
sample_input = inputs[-1, 0].numpy()  # Real Conductivity
sample_truth = outputs[-1, 0].numpy() # Real Temp
sample_pred = test_preds[-1, 0]       # Un-normalized Predicted Temp

# 2. Calculate the Absolute Error Map
error_map = np.abs(sample_truth - sample_pred)

# 3. Paint the 4-panel canvas
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(sample_input, cmap='inferno')
axes[0].set_title("Input Conductivity")

axes[1].imshow(sample_truth, cmap='inferno')
axes[1].set_title("MATLAB Truth")

axes[2].imshow(sample_pred, cmap='inferno')
axes[2].set_title("AI Prediction")

im = axes[3].imshow(error_map, cmap='inferno')
axes[3].set_title("Absolute Error")
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('money_shot_plot.png')
plt.clf()
