function [UU,W,Z,Zi,iter,obj] = algo_qp(X,Y,A,AA,numanchor,beta)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

m = numanchor;
numview = length(X);
numsample = size(Y,1);
k = length(unique(Y));

Z = zeros(m,numsample); % m  * n

for i = 1:numview
   di = size(X{i},1); 
   W{i} = eye(m);
   X{i} = X{i}'; % turn into d*n
   Zi{i} = zeros(m,numsample); 
end
Z(:,1:m) = eye(m);

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    %% optimize Z
    for iv=1:numview
        H = W{iv}*AA{iv}*W{iv};
        H = (H+H')/2;
        options = optimset( 'Algorithm','interior-point-convex','Display','off'); 
        Zp = zeros(m,numsample);
        parfor j=1:numsample
            ff=0;
            ff = -X{iv}(:,j)'*A{iv}*W{iv};
            Zp(:,j) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
        end
        Zi{iv} = Zp;
        zz{iv} = Zp*Zp';
    end

    %% optimize W_i
    options = optimset( 'Algorithm','interior-point-convex','Display','off'); 
    parfor iv=1:numview
        R = zz{iv}.*AA{iv}+beta*AA{iv};
        fw= zeros(m,1);
        G = Zi{iv}*X{iv}'*A{iv};
        for j = 1:m
            fw(j) = -G(j,j);
        end
        ww{iv} = quadprog(R,fw',[],[],ones(1,m),m,zeros(m,1),m*ones(m,1),[],options);
    end
    for iv = 1:numview
        for j = 1:m
             W{iv}(j,j) = ww{iv}(j);
        end
    end

    %%
    term1 = 0;
    term2 = 0;
    for iv = 1:numview
        term1 = term1 + norm(X{iv} - A{iv} * W{iv} * Zi{iv},'fro')^2;
        term2 = term2 + ww{iv}'*AA{iv}*ww{iv};
    end
    obj(iter) = term1+beta*term2;
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        Z = [];
        for iv = 1:1:numview
            Z=cat(1,Z,1/sqrt(numview)*Zi{iv});
        end
        [UU,~,~]=mySVD(Z',k);
        flag = 0;
    end
end
         
         
    
