clc;
clear;
warning off;
close all;

% Load data
data = table2array(readtable('.\water_huseyin.xlsx'));

% Select specific columns (features and output)
features = (data(:, 2:8));
output = categorical(data(:, 11));

% Normalize features and output
  % features = (features - min(features)) ./ (max(features) - min(features));


% Normalize the data
X = zscore(features);
y = (output);
num_samples = size(X,1);
input_dim = size(X,2);
hidden_units = 1024;
% Split data into training and testing sets
train_ratio = 0.8;
num_train = floor(train_ratio * num_samples);
X_train = X(1:num_train, :);
y_train = y(1:num_train);
X_test = X(num_train+1:end, :);
y_test = y(num_train+1:end);
 % rng(42)

layers = [
    featureInputLayer(input_dim, 'Normalization', 'none', 'Name', 'Input')
    
    % Gating Mechanism
    fullyConnectedLayer(hidden_units, 'Name', 'Gate_FC1')
    sigmoidLayer('Name', 'Gate_Sigmoid')  % Gate using sigmoid
    fullyConnectedLayer(hidden_units, 'Name', 'Gate_FC2')
    tanhLayer('Name', 'Gate_Tanh')       % Activation gate using tanh
    
    % Liquid Neural Dynamics Layer
    customLiquidLayer(hidden_units, 'Name', 'Liquid_Dynamics')  
    
    % Output Layer
    fullyConnectedLayer(3, 'Name', 'Output_FC')  
    softmaxLayer('Name', 'Output_Softmax') % Classification output
    classificationLayer('Name', 'ClassificationLoss')
];


% analyzeNetwork(layers)
% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 3000, ...
    'InitialLearnRate', 0.001, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
     'ValidationData', {X_test, y_test}, ...
    'Verbose', true);

% Train the GLNN model
net = trainNetwork(X_train, y_train, layers, options);

% Test the network with Monte Carlo Dropout
y_pred = classify(net,X_test);
confusionchart(y_test, y_pred)
metrics = multiclass_metrics_common(confusionmat(y_test, y_pred))

