close all;
clear;
clc;
%% Load data and functions
addpath('data');
addpath('utility');
load('Reuters.mat'); % Raw data 
load('shuffled_reuters.mat'); % Shuffled data
%% Initialization
nv = size(X, 2); % number of view
total_num = length(Y); % number of sample
size_window =500;  % size of window
num_windows = floor(total_num / size_window);
gnd = Y(rand_set, 1)+1';  % Label: 1*n  for Reuters, the label should plus 1 
K = length(unique(gnd));% number of clusters


final_clustering_accs = zeros(1,num_windows);
final_clustering_nmis = zeros(1,num_windows);
final_clustering_fmeasures = zeros(1,num_windows);
final_clustering_costs = zeros(1,num_windows);
iter_stop = zeros(1,num_windows);

original_data_views = cell(1, nv);
prev_window_data = cell(1,nv);
current_window_data = cell(1,nv);

%% para setting
    number_anchor = K;
    dimension_anchor = K;
    lambda = 0.0001; % tradeoff para
    beta = 0.8; % tradeoff para
    dim_idx=1;
    F_norm =0;

%% Shuffle the raw data randomly（if you need）    
% rand('state', 100);
% rand_set = randperm(total_num);
% rand_data_views = cell(1, nv);
% for nv_idx = 1 : nv 
%     rand_data_views{nv_idx} = X{nv_idx}(rand_set, :)';   % rand_data_views:d*n
% end
% save('shuffled_Animal.mat','rand_data_views','rand_set');    

%% Start:
prev_A = zeros(dimension_anchor, number_anchor); % Set of anchors from previous window
for wnd_idx = 1 : num_windows
            tic;        
            % Get the data of the current window
            start_idx = (wnd_idx - 1) * size_window + 1;
            for nv_idx = 1 : nv                                 
                current_window_data{nv_idx} = rand_data_views{nv_idx}(:, start_idx : start_idx + size_window - 1);
            end
            ground_lables = gnd(start_idx : start_idx + size_window - 1);
            
            if wnd_idx >1
                  for iv = 1:nv
                       F_norm = norm(current_window_data{iv}-prev_window_data{iv},'fro')^2;
                       F_norm = F_norm/size_window;
                  end
            end
            prev_window_data = current_window_data;
            [A,P,S,G,F,alpha,iter] = algo_MSIAL(current_window_data,prev_A,F_norm,wnd_idx,ground_lables,lambda,beta,dimension_anchor,number_anchor);
            prev_A = A;
            iter_stop(wnd_idx) = iter;
            [~,actual_ids]=max(F); % Obtain the cluster assignment
            time_cost = toc;
%             
%             num_sc_clusters = length(unique(actual_ids));
%             if num_sc_clusters ~= K
%                disp('The cluster assignment is wrong and the number of clusters is inconsistent!'); 
%             end
             [current_ground_lables, ~] = refresh_labels(ground_lables, K);
             [current_cluster_lables, ~] = refresh_labels(actual_ids(1 , 1: size_window), K);
             num_current_clusters = length(unique(current_cluster_lables));                    
             [acc, nmi, purity, fmeasure, ~, ~] = calculate_dynamic_clustering_results(current_cluster_lables, current_ground_lables, num_current_clusters);
              
             final_clustering_accs(wnd_idx) = acc;
             final_clustering_nmis(wnd_idx) = nmi;
             final_clustering_fmeasures(wnd_idx) = fmeasure;           
             final_clustering_costs(wnd_idx) = time_cost;
             fprintf('Window: %d, ACC: %.4f, NMI: %.4f, FM: %.4f, Time: %f\n',wnd_idx, acc, nmi, fmeasure, time_cost); 
end
average_iter_stop = mean(iter_stop);
average_acc =  mean(final_clustering_accs);
average_nmi =  mean(final_clustering_nmis);           
average_fm =  mean(final_clustering_fmeasures);            
average_cost = mean(final_clustering_costs);   
fprintf('Average : ACC: %.4f, NMI: %.4f, FM: %.4f, Time: %f, Iter: %d\n', average_acc, average_nmi, average_fm, average_cost,average_iter_stop);   

