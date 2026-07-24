import torch
import scipy.io
import numpy as np
import torch.utils.data as data

print("1. Loading the dataset...")
data_mat = scipy.io.loadmat('matlab/thermal_dataset.mat')

k_s_array = data_mat['k_s'].flatten() 
k_c_array = data_mat['k_c'].flatten() 
T_data = data_mat['T_data']           

num_samples = len(k_s_array)

print("2. Painting the Input Images (Conductivity Maps)...")
input_images = np.zeros((num_samples, 64, 64))

for i in range(num_samples):
    input_images[i, :, :] = k_c_array[i]
    input_images[i, 16:48, 16:48] = k_s_array[i]

T_data = np.transpose(T_data, (2, 0, 1))

inputs = torch.tensor(input_images, dtype=torch.float32).unsqueeze(1)
outputs = torch.tensor(T_data, dtype=torch.float32).unsqueeze(1)

print("3. Normalizing the Data...")
in_mean = torch.mean(inputs)
in_std = torch.std(inputs)
out_mean = torch.mean(outputs)
out_std = torch.std(outputs)

norm_inputs = (inputs - in_mean) / in_std
norm_outputs = (outputs - out_mean) / out_std

# Save normalization stats to un-normalize later
norm_stats = {
    'in_mean': in_mean, 'in_std': in_std,
    'out_mean': out_mean, 'out_std': out_std
}
torch.save(norm_stats, 'normalization_stats.pt')

from fno_model import FNO2d

print("\n4. Building the Fourier Neural Operator...")
model = FNO2d(modes=12, width=32)

print("\n5. Training the Fourier Neural Operator...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

dataset = data.TensorDataset(norm_inputs, norm_outputs)
loader = data.DataLoader(dataset, batch_size=20, shuffle=True)

epochs = 500
loss_history = []

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_in, batch_out in loader:
        optimizer.zero_grad()
        predictions = model(batch_in)
        loss = loss_fn(predictions, batch_out)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(loader)
    loss_history.append(avg_loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

print("Training Complete!")
print("Saving model and loss history...")
torch.save(model.state_dict(), 'fno_model_weights.pth')
np.save('loss_history.npy', np.array(loss_history))
print("Saved successfully!")