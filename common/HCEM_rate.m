function [training_data,detec_result] = HCEM_rate(train_data_record,img,para)

detal = para.detal;
epsilon = para.epsilon;
max_it = para.max_it;
lambda = para.lambda;
num = para.num;
k = ceil(para.k*(25+num));
no_class = max(train_data_record(:,1));

training_data = [];
detec_result = [];
for c = 1:no_class
    class_noisy_data = train_data_record(find(train_data_record(:,1) == c),:);
    class_noisy_index = class_noisy_data;
    class_noisy_index(:,2) = [];
    X = img(class_noisy_data(:,3),:);
    % 目标光谱选择
    pdist_sum = sum(pdist_le(X,X,'SAM'));
    [~,sort_pdist] = sort(pdist_sum);
    k_d = X(sort_pdist(1:k),:)';
    d = mean(k_d,2);
% %     pdist_min = min(pdist_sum);
% %     d_index(c) = find( pdist_min == pdist_sum);
% %     d = X(d_index(c),:)';
    [tr_num,tr_band] = size(X);
    X = X';
    % hCEM algorithm
    % initialization
    Weight = ones(1,tr_num);
    y_old = ones(1,tr_num);
    Energy = [];
    
    % Hierarchical filtering process
    display('hierarchical filtering...');
    for T = 1:max_it
        for pxlID = 1:tr_num
            X(:,pxlID) = X(:,pxlID).*Weight(pxlID);
        end
        R = X*X'/tr_num;
        % To inrease stability, a small diagnose matrix is added
        % before the matrix inverse process.
        w = inv(R+0.0001*eye(tr_band)) * d / (d'*inv(R+0.0001*eye(tr_band)) *d);
        y = w' * X;
        W1 = exp(-detal*y);
        Weight =1 - W1./repmat(sqrt(sum(W1.*W1)),[size(W1,1) 1]);
        res = norm(y_old)^2/tr_num - norm(y)^2/tr_num;
        Energy = [Energy, norm(y)^2/tr_num];
        y_old = y;
        % stop criterion:
        if (abs(res)<epsilon)
            break;
        end
    end
    limit_y = mean(y)*lambda;
    sample_index = find(y <= limit_y);
    
    % Noisy labels检测统计：
    % 正确检测: '1' (N -> N)
    % 错误检测: '-1' (T -> N)
    % 未检测到: '0' (N -> ×)
    % 真实样本: '2' (T)
    num = para.num;
    Stat_matrix = ones(length(y),1)*2;
    Si = find(sample_index <= num);
    Stat_matrix(sample_index(Si)) = 1;
    Se = find(sample_index > num);
    Stat_matrix(sample_index(Se)) = -1;
    N = 1:num;
    [~,Ni,~] = intersect(N,sample_index);
    N(Ni) = [];
    Stat_matrix(N) = 0;
    detec_result = [detec_result,Stat_matrix];
    class_noisy_index(sample_index,:) = [];
    training_data = [training_data;class_noisy_index];
end
