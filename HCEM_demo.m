% Hierarchical Structure-Based Noisy Labels Detection
% for Hyperspectral Image Classification (JSTAR-20200512)

close all; clear all; clc
addpath ('.\common')
%% load original image
load(['.\datasets\KSC.mat']);
load(['.\datasets\KSC_gt2.mat']);

%% Preprocess the image
no_class = max(GroundT(:,2));
GroundT = GroundT';
img = KSC;
img2 = KSC;% original 
[no_row, no_col, no_bands] = size(img);
img = reshape(img, no_row * no_col, no_bands);

%% Construct training set and test set
per_class_ture = 25;
Value = ones(no_class,1).*per_class_ture; % the number of samples (24) per class
indexes = train_random_select(GroundT(2,:),Value); % based on 24 for each class
train_SL = GroundT(:,indexes);
test_SL = GroundT;
test_SL(:,indexes) = [];

train_samples = img(train_SL(1,:),:);
train_labels = train_SL(2,:);
test_samples = img(test_SL(1,:),:);
GroudTest = test_SL(2,:)';

train_samples1 = img(train_SL(1,:),:);
train_labels1 = train_SL(2,:);

%% Experiment 1 (original SVM classification result)
[ OA_1,AA_1,kappa_1,CA_1 ] = My_SVM_Classifier(img2,train_samples1,train_labels1',test_SL,GroudTest);

%% Add 5 and 15 mislabel samples
load(['.\datasets\Noise_samples_5.mat']);
train_data_record = Noise_samples_5(:,:,1);

% load(['.\datasets\Noise_samples_15.mat']);
% train_data_record = Noise_samples_15(:,:,1);
%% Experiment 2 (SVM classification with mislabel samples)
training_label_1 = train_data_record(:,1);
training_data_1 = img(train_data_record(:,3)',:);
[OA_2,AA_2,kappa_2,CA_2] = My_SVM_Classifier(img2,training_data_1,training_label_1,test_SL,GroudTest);

%% Experiment 3 (SVM classification after HCEM with mislabel samples)
[ OA_3,AA_3,kappa_3,CA_3 ] = HCEM(train_data_record,img2,test_SL,GroudTest);
