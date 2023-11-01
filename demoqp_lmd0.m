clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
ds = {'proteinFold'};


dsPath = './dataset/';
resPath = './res-lmd0/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

for dsi = 1:1:length(ds)
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
    k = length(unique(Y));
    numview = length(X);
    
    %% para setting for ProteinFold
    anchor = k;
    beta = 2;
    
    %%
    for i=1:numview
        rand('twister',12);
        X{i} = mapstd(X{i}',0,1)';
        [~, A{i}] = litekmeans(X{i},anchor,'MaxIter', 100,'Replicates',3);
        AA{i} = A{i}*A{i}';
        A{i} =  A{i}';
    end
    tic;
    [U,W,Z,iter,obj] = algo_qp(X,Y,A,AA,anchor,beta);
    [res,std] = myNMIACCwithmean(U,Y,k);
    timer = toc;
    fprintf('Anchor:%d \t beta:%d\t Res:%12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[anchor beta res(1) res(2) res(3) res(4) timer]);
end