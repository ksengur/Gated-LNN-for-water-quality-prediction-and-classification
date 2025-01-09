clc;
clear;
warning off;
close all;

% Load data
data = table2array(readtable('.\water_huseyin.xlsx'));

% Select specific columns (features and output)
features = (data(:, 2:8));
output = (data(:, 9));

% Normalize features and output
X = zscore(features);
y = zscore(output);

% Split data into training and testing sets
num_samples = size(X, 1);
train_ratio = 0.8;
num_train = floor(train_ratio * num_samples);

X_train = X(1:num_train, :);
y_train = y(1:num_train);
X_test = X(num_train+1:end, :);
y_test = y(num_train+1:end);

% Define LNN parameters
input_dim = size(X_train, 2);
hidden_units = 1024;

% Define the LNN architecture
layers = [

    featureInputLayer(input_dim, 'Normalization', 'none', 'Name', 'Input')
    
    % Gating Mechanism
    fullyConnectedLayer(hidden_units, 'Name', 'Gate_FC1')
    sigmoidLayer('Name', 'Gate_Sigmoid')  % Gate using sigmoid
    fullyConnectedLayer(hidden_units, 'Name', 'Gate_FC2')
    tanhLayer('Name', 'Gate_Tanh')       % Activation gate using tanh
    
    % Liquid Neural Dynamics Layer
    customLiquidLayer(hidden_units, 'Name', 'Liquid_Dynamics')  
    
    fullyConnectedLayer(1, 'Name', 'Output') % Single-output layer
    regressionLayer('Name', 'RegressionLoss') % Loss for regression tasks
];

% Analyze the network
analyzeNetwork(layers);

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 2000, ...
    'InitialLearnRate', 0.0001, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ValidationData', {X_test, y_test}, ...
    'Verbose', true);

% Train the LNN model
net = trainNetwork(X_train, y_train, layers, options);

% Test the network with Monte Carlo Dropout
num_mc_samples = 50; % Number of Monte Carlo samples
y_pred_samples = zeros(size(X_test, 1), num_mc_samples);

for mc_sample = 1:num_mc_samples
    y_pred_samples(:, mc_sample) = predict(net, X_test);
end

% Compute mean and uncertainty of predictions
y_pred_mean = mean(y_pred_samples, 2); % Mean prediction
y_pred_std = std(y_pred_samples, [], 2); % Prediction uncertainty

% Evaluate performance
rmse = sqrt(mean((y_test - y_pred_mean).^2));
mae = mean(abs(y_test - y_pred_mean));
mape = mean(abs((y_test - y_pred_mean) ./ y_test)) * 100;
r2 = 1 - sum((y_test - y_pred_mean).^2) / sum((y_test - mean(y_test)).^2);

% Display the metrics
fprintf('Test RMSE: %.4f\n', rmse);
fprintf('Test MAE: %.4f\n', mae);
fprintf('Test MAPE: %.2f%%\n', mape);
fprintf('Test R²: %.4f\n', r2);

% Plot results
figure;
hold on;
plot(1:length(y_test), y_test, '-o', 'DisplayName', 'True Values');
plot(1:length(y_pred_mean), y_pred_mean, '-x', 'DisplayName', 'Predicted Values');
legend;
xlabel('Sample Index');
ylabel('WQI');
grid on;

