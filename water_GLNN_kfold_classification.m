clc;
clear;
warning off;
close all;

% Load data
data = table2array(readtable('.\water_huseyin.xlsx'));

% Select specific columns (features and output)
features = data(:, 2:8);
output = categorical(data(:, 11));

% Normalize features
X = zscore(features);
y = output;

% Define k-fold cross-validation
k = 5; % Number of folds
cv = cvpartition(y, 'KFold', k);
input_dim = size(X,2);
hidden_units = 1024;
% Initialize metrics storage
all_metrics = [];

for fold = 1:k
    fprintf('Processing fold %d/%d...\n', fold, k);
    
    % Split data into training and testing sets for this fold
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    X_train = X(trainIdx, :);
    y_train = y(trainIdx);
    X_test = X(testIdx, :);
    y_test = y(testIdx);
    
    % Define Gated Liquid Neural Network architecture
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
    classificationLayer('Name', 'ClassificationLoss')];


    % Define training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 600, ...
        'InitialLearnRate', 0.0001, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false);

    % Train the GLNN model
    net = trainNetwork(X_train, y_train, layers, options);

    % Test the network
    y_pred = classify(net, X_test);
    
    % Confusion matrix and metrics for this fold
    cm = confusionmat(y_test, y_pred);
    fold_metrics = multiclass_metrics_common(cm);
    all_metrics = [all_metrics; fold_metrics]; % Append fold metrics
    
    % Display confusion chart for this fold
    confusionchart(y_test, y_pred, 'Title', sprintf('Fold %d Confusion Matrix', fold));
end

% Compute and display average metrics across all folds
average_metrics = mean(all_metrics, 1);
disp('Average Metrics Across Folds:');
disp(average_metrics);
