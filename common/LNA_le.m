function [noise_train_label,noise_train_data,noise_train_data1] = LNA_kang(train_SL,Test_SL,train_labels,train_samples,Test_SL_samples,per)
%函数思路;根据每一类的噪声数据的比例（个数），从其他类中选取相应个数的像元数据。
PretreatmentLabel = [];
NoiseLD = [];
NoiseData = [];
noise_train_data1 = [];
noise_train_label = [];
LabelData =[(train_SL(2,:))',(train_SL(1,:))',train_samples]; 
MiddleData = LabelData;
% per = 10;
for i = 1:max(train_labels)
    if i>1
       MiddleData(find(MiddleData(:,1) == i-1),:) = []; 
    end
    LabelCount(i) = length(find(train_labels == i));
    NoiseCount(i) = per;
    PretreatmentLabel = [PretreatmentLabel;i*ones(NoiseCount(i),1)];  
    train_labels_1 = ones(1,LabelCount(i)+per)*i;
    noise_train_label = [noise_train_label train_labels_1];
end
NoiseLD = [(Test_SL(2,:))' (Test_SL(1,:))' Test_SL_samples];
for j = 1:max(PretreatmentLabel)
    NoiseCount1 = NoiseCount;
    NoiseLD1 = NoiseLD;
    NoiseCount1(j) = [];
    NoiseLD1(find(NoiseLD1(:,1) == j),:) = [];
    AfterLabel = randi([1,size(NoiseLD1,1)],1,NoiseCount(j));
    NoiseDataMiddle = NoiseLD1(AfterLabel,:);
    NoiseData = [NoiseData;NoiseDataMiddle];
end
NoiseData_Bfor = [PretreatmentLabel,NoiseData];
for k = 1:max(PretreatmentLabel)
    Pdata = LabelData(find(LabelData(:,1)==k),:);
    Ndata = NoiseData_Bfor(find(NoiseData_Bfor(:,1)==k),:);
    Edata = [Ndata(:,(2:end));Pdata];  
%     Edata(1:NoiseCount(k),:) = Ndata(:,2:end);
    noise_train_data1 = [noise_train_data1;Edata];
end
noise_train_data = noise_train_data1(:,3:end);
noise_train_label = noise_train_label';
end