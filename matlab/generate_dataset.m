% 1. Define how many simulations we want
num_samples = 5; % TEST RUN: Let's do 5 first to make sure it works!

% 2. Generate 500 rows and 2 columns of perfectly spread decimals (0.0 to 1.0)
random_decimals = lhsdesign(num_samples, 2); 

% 3. Stretch the decimals to our physical material properties
k_s = 100 + (200 - 100) * random_decimals(:, 1);  % Silicon: 100 to 200
k_c = 300 + (500 - 300) * random_decimals(:, 2);  % Copper: 300 to 500

% 4. Create an empty 3D box to store our 500 images. 
T_data = zeros(64, 64, num_samples);

% 5. The Loop: Run the physics simulation 500 times
for i = 1:num_samples
    current_ks = k_s(i);
    current_kc = k_c(i);
    
    % Call the solver! Using 8 for refinement (ref=8)
    T_grid = solve_thermal_field(current_ks, current_kc, 8);
    
    % Save the resulting 64x64 grid into layer "i" of our 3D box
    T_data(:, :, i) = T_grid;
    disp(['Finished simulation ', num2str(i), ' out of ', num2str(num_samples)]);
end

% 6. Save everything into a file
save('thermal_dataset.mat', 'k_s', 'k_c', 'T_data');
disp('Data generation complete and saved to thermal_dataset.mat!');
