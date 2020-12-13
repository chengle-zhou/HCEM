function [OA,AA,kappa,CA] = HCEM(train_data_record,img2,test_SL,GroudTest)
%Input:
%   train_data_record: training samples with mislabel samples.
%   img2: the original image.
%   test_SL: the test set.
%   GroudTest: the turth label of all samples.
%Output:
%   the Classification accuracy OA, AA,kappa and CA.
% 
%% HCEM parameter setting
% The meaning and optimal valueof those parameters can be found in our paper.
num = 5;                %the number of mislabel samples: 5 or 15

para.detal = 3;         
para.epsilon = 1e-3;
para.max_it = 10;
para.lambda = 0.14;
para.num = num;
para.k = 0.4;           %k-rate: 0.4 or 0.6
   
%% preprocess the image
[no_row, no_col, no_bands] = size(img2);
img = img2; 
img = reshape(img, no_row * no_col, no_bands);  
img = normcol(img');
img = img';  
%% HCEM 
[training_index,detec_result] = HCEM_rate(train_data_record,img,para);
%% SVM classification
training_data = img(training_index(:,2),:);
training_label = training_index(:,1);

[training_data,M,m] = scale_func(training_data);
[img] = scale_func(img,M,m);

[Ccv2 Gcv2 cv cv_t] = cross_validation_svm(training_label,training_data);
parameter = sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv2,Gcv2); 
model = svmtrain(training_label,training_data,parameter);
SVMresult = svmpredict(ones(no_row*no_col,1),img,model); 
SVMResultTest = SVMresult(test_SL(1,:),:);

[ OA,AA,kappa,CA ] = confusion(GroudTest,SVMResultTest);
