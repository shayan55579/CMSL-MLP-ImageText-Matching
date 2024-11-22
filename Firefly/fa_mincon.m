% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    

% -------------------------------------------------------- %
% Firefly Algorithm for constrained optimization using     %
% for the design of a spring (benchmark)                   % 
% by Xin-She Yang (Cambridge University) Copyright @2009   %
% -------------------------------------------------------- %

function fa_mincon()
% parameters [n N_iteration alpha betamin gamma]
global W_dim;
global para;
format long;

help fa_mincon.m
% This demo uses the Firefly Algorithm to solve the
% [Spring Design Problem as described by Cagnina et al.,
% Informatica, vol. 32, 319-326 (2008). ]
% Define additional parameters for the cost function
filename1 = 'out1.csv';  % Update with the correct path
filename2 = 'Senten.csv';  % Update with the correct path
alpha = 1.0;  % Example values, adjust as necessary
beta = 0.5;
gamma = 0.3;
% Simple bounds/limits
disp('Start Solving Problem ...');
Lb=ones(W_dim(1),W_dim(2)).*-2;
Ub=ones(W_dim(1),W_dim(2)).*2;

% Initial random guess
W0=Lb+(Ub-Lb).*rand(size(Lb));

% Reshape Matrices
Lb=reshape(Lb,[1,W_dim(1)*W_dim(2)]);
Ub=reshape(Ub,[1,W_dim(1)*W_dim(2)]);
W0=reshape(W0,[1,W_dim(1)*W_dim(2)]);

%[u,fval,NumEval]=ffa_mincon(@cost,@constraint,W0,Lb,Ub,para);
% I change this
%[u,fval,NumEval] = ffa_mincon(@(w) cost(w, filename1, filename2, alpha, beta, gamma), @constraint, W0, Lb, Ub, para);
[u,fval,NumEval] = ffa_mincon(@(w) cost(w, filename1, filename2, alpha, beta, gamma), [], W0, Lb, Ub, para);



% Display results
result.best_solution=u;
result.best_objective=fval;
result.total_number_of_function_evaluations=NumEval;
%save(['Optimization_Results\MFA_Matrix_with_',num2str(para(1)),...
   % '_FireFly_and_',num2str(para(2)),'_Iterations.mat'],'result');
% Save the result in the current directory with the name 'file.mat'
save('file1.mat', 'result');
disp('Result file saved!!!');
disp(['Best Objective Value : ', num2str(result.best_objective)]);
end
%%% Put your own cost/objective function here --------%%%
%% Cost or Objective function
function z = cost(w, filename1, filename2, alpha, beta, gamma)
    % Read data from CSV files as single precision
	%w = reshape(w,[300,50]);
	w = reshape(w,[10,10]);
    % Print size of w
    %disp(['Size of W: ', mat2str(size(w))]);
    I = single(table2array(readtable(filename1)));
    S = single(table2array(readtable(filename2)));
	

    % Clear filenames as they are no longer needed
    clear filename1 filename2;

    % Preprocessing just for image (remove headers/metadata)
    %I(:,1) = []; I(1,:) = [];  % Remove the first column and row
    I(1,:) = [];
    S(1,:) = [];
	% Print size of I and S
    %disp(['Size of Image: ', mat2str(size(I))]);
    %disp(['Size of Sentences: ', mat2str(size(S))]);
    % Truncate I and S to only use the first 10 rows ****
    I = I(1:5, :);
    S = S(1:5, :);
    %*****************************************
    % Extract and remove labels
    LabelImage = I(:,end); I(:, end) = [];
    
    LabelSen = S(:,end); S(:, end) = [];
    

    % Normalize rows in I and S
    for j = 1:size(I,1)
        I(j,:) = I(j,:) / norm(I(j,:), 2);
    end
    for j = 1:size(S,1)
        S(j,:) = S(j,:) / norm(S(j,:), 2);
    end

    % Clear variable j as it is no longer needed
    clear j;

    
    w = single(w);
	
	% Similarity matrices with single precision weights

	%image
    Iw = I * w; 
    i = Iw * Iw';
    i = triu(i, 1); % Upper triangular to avoid redundancy

	%sentences
    Sw = S * w; 
    s = Sw * Sw';
    s = triu(s, 1); % Upper triangular to avoid redundancy

	%cross
    cross = Sw * Iw';

    % Initialize loss values as single precision
    matchLossImage = single(0);
    nonMatchLossImage = single(0);
    matchLossSentence = single(0);
    nonMatchLossSentence = single(0);
    matchLossCross = single(0);
    nonMatchLossCross = single(0);

    % Compute losses for image-image similarity
    for rowIdx = 1:size(i, 1)
        for colIdx = rowIdx + 1:size(i, 2)
            if LabelImage(rowIdx) == LabelImage(colIdx)
                matchLossImage = matchLossImage + i(rowIdx, colIdx);
            else
                nonMatchLossImage = nonMatchLossImage + i(rowIdx, colIdx);
            end
        end
    end

	% Compute losses for sentence-sentence similarity
    for rowIdx = 1:size(s, 1)
        for colIdx = rowIdx + 1:size(s, 2)
            if LabelSen(rowIdx) == LabelSen(colIdx)
                matchLossSentence = matchLossSentence + s(rowIdx, colIdx);
            else
                nonMatchLossSentence = nonMatchLossSentence + s(rowIdx, colIdx);
            end
        end
    end

	% Compute losses for cross-modal similarity
    for rowCross = 1:size(cross, 1)
        for colCross = 1:size(cross, 2)
            if LabelSen(rowCross) == LabelImage(colCross)
                matchLossCross = matchLossCross + cross(rowCross, colCross);
            else
                nonMatchLossCross = nonMatchLossCross + cross(rowCross, colCross);
            end
        end
    end


    % Merge and weight losses
    Loss1 = alpha * matchLossImage + beta * matchLossSentence + gamma * matchLossCross;
    Loss2 = alpha * nonMatchLossImage + beta * nonMatchLossSentence + gamma * nonMatchLossCross;

    % Calculate final cost
    z = (1 / (Loss1 + eps('single'))) + Loss2;
	% Convert result back to CPU array if needed
    
    % Clear large intermediate matrices to free memory
    
end

%scn_func_value = sum(sum(abs(total_scn_scn_sim_mat - (Z_prim*Z_prim')),1),2);
%z = obj_func_value + scn_func_value;
%end

% Constrained optimization using penalty methods
% by changing f to F=f+ \sum lam_j*g^2_j*H_j(g_j)
% where H(g)=0 if g<=0 (true), =1 if g is false

%%% Put your own constraints here --------------------%%%
%function [g,geq]=constraint(x)
% All nonlinear inequality constraints should be here
% If no inequality constraint at all, simple use g=[];
%g = [];

% all nonlinear equality constraints should be here
% If no equality constraint at all, put geq=[] as follows
%geq =[];
%end

%%% End of the part to be modified -------------------%%%

%%% --------------------------------------------------%%%
%%% Do not modify the following codes unless you want %%%
%%% to improve its performance etc                    %%%
% -------------------------------------------------------
% ===Start of the Firefly Algorithm Implementation ======
% Inputs: fhandle => @cost (your own cost function,
%                   can be an external file  )
%     nonhandle => @constraint, all nonlinear constraints
%                   can be an external file or a function
%         Lb = lower bounds/limits
%         Ub = upper bounds/limits
%   para == optional (to control the Firefly algorithm)
% Outputs: nbest   = the best solution found so far
%          fbest   = the best objective value
%      NumEval = number of evaluations: n*MaxGeneration
% Optional:
% The alpha can be reduced (as to reduce the randomness)
% ---------------------------------------------------------

% Start FA
function [nbest,fbest,NumEval] = ffa_mincon(fhandle,nonhandle,u0, Lb, Ub, para)
% Check input parameters (otherwise set as default values)
if nargin<6, para=[20 50 0.25 0.20 1]; end
if nargin<5, Ub=[]; end
if nargin<4, Lb=[]; end
if nargin<3
    disp('Usuage: FA_mincon(@cost, @constraint,u0,Lb,Ub,para)');
end

% n=number of fireflies
% MaxGeneration=number of pseudo time steps
% ------------------------------------------------
% alpha=0.25;      % Randomness 0--1 (highly random)
% betamn=0.20;     % minimum value of beta
% gamma=1;         % Absorption coefficient
% ------------------------------------------------
n=para(1);  MaxGeneration=para(2);
alpha=para(3); betamin=para(4); gamma=para(5);

% Total number of function evaluations
NumEval=n*MaxGeneration;

% Check if the upper bound & lower bound are the same size
if length(Lb) ~=length(Ub)
    disp('Simple bounds/limits are improper!');
    return
end

% Calcualte dimension
d=length(u0);

% Initial values of an array
zn=ones(n,1)*10^100;
% ------------------------------------------------
% generating the initial locations of n fireflies
[ns,Lightn]=init_ffa(n,d,Lb,Ub,u0);

% Iterations or pseudo time marching
for k=1:MaxGeneration    %%%%% start iterations
    %***************************** add this******
    disp(['Starting Epoch: ', num2str(k)])
% This line of reducing alpha is optional
 alpha=alpha_new(alpha,MaxGeneration);

% Evaluate new solutions (for all n fireflies)
for i=1:n
   zn(i)=Fun(fhandle,nonhandle,ns(i,:));
   Lightn(i)=zn(i);
end

% Ranking fireflies by their light intensity/objectives
[Lightn,Index]=sort(zn);
ns_tmp=ns;
for i=1:n
 ns(i,:)=ns_tmp(Index(i),:);
end

%% Find the current best
nso=ns; Lighto=Lightn;
nbest=ns(1,:); Lightbest=Lightn(1);

% For output only
fbest=Lightbest;

% Move all fireflies to the better locations
[ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,Lightbest,alpha,betamin,gamma,Lb,Ub);

%if(mod(k,500)==0)
 %   disp(['Iteration : ',num2str(k), '  --  Objective Function : ',num2str(fbest)]);
%end
%%% Change to this
if(mod(k,1)==0)  % This will always be true for each iteration
    disp(['Iteration : ', num2str(k), '  --  Objective Function : ', num2str(fbest)]);
end

disp(['End of Epoch ', num2str(k), ': Best Objective Value = ', num2str(fbest)]);
end   %%%%% end of iterations
end
% -------------------------------------------------------
% ----- All the subfunctions are listed here ------------
% The initial locations of n fireflies
function [ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)
  % if there are bounds/limits,
if length(Lb)>0
   for i=1:n
        ns(i,:)=Lb+(Ub-Lb).*rand(1,d);
   end
else
   % generate solutions around the random guess
   for i=1:n
        ns(i,:)=u0+randn(1,d);
   end
end

% initial value before function evaluations
Lightn=ones(n,1)*10^100;
end

% Move all fireflies toward brighter ones
function [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,Lightbest,alpha,betamin,gamma,Lb,Ub)
% Scaling of the system
scale=abs(Ub-Lb);

% Updating fireflies
for i=1:n
% The attractiveness parameter beta=exp(-gamma*r)
   for j=1:n
        r=sqrt(sum((ns(i,:)-ns(j,:)).^2));
        % Update moves
        if Lightn(i)>Lighto(j) % Brighter and more attractive
           beta0=1; beta=(beta0-betamin)*exp(-gamma*r.^2)+betamin;
           tmpf=alpha.*(rand(1,d)-0.5).*scale;
           ns(i,:)=ns(i,:).*(1-beta)+nso(j,:).*beta+tmpf;
        end
   end % end for j
end % end for i

% Check if the updated solutions/locations are within limits
[ns]=findlimits(n,ns,Lb,Ub);
end

% This function is optional, as it is not in the original FA
% The idea to reduce randomness is to increase the convergence,
% however, if you reduce randomness too quickly, then premature
% convergence can occur. So use with care.
function alpha=alpha_new(alpha,NGen)
% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
% alpha_0=0.9
delta=1-(10^(-4)/0.9)^(1/NGen);
alpha=(1-delta)*alpha;
end

% Make sure the fireflies are within the bounds/limits
function [ns]=findlimits(n,ns,Lb,Ub)
for i=1:n
     % Apply the lower bound
  ns_tmp=ns(i,:);
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);

  % Apply the upper bounds
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move
  ns(i,:)=ns_tmp;
end
end
% -----------------------------------------
% d-dimensional objective function
function z=Fun(fhandle,nonhandle,u)
% Objective
z=fhandle(u);

% Apply nonlinear constraints by the penalty method
% Z=f+sum_k=1^N lam_k g_k^2 *H(g_k) where lam_k >> 1
z=z+getnonlinear(nonhandle,u);
end

function Z=getnonlinear(nonhandle,u)
Z=0;
% Penalty constant >> 1
%lam=10^15; lameq=10^15;
% Get nonlinear constraints
%[g,geq]=nonhandle(u);

% Apply inequality constraints as a penalty function
%for k=1:length(g)
 %   Z=Z+ lam*g(k)^2*getH(g(k));
%end
% Apply equality constraints (when geq=[], length->0)
%for k=1:length(geq)
%   Z=Z+lameq*geq(k)^2*geteqH(geq(k));
%end
end

% Test if inequalities hold
% H(g) which is something like an index function
function H=getH(g)
if g<=0
    H=0;
else
    H=1;
end
end

% Test if equalities hold
function H=geteqH(g)
if g==0
    H=0;
else
    H=1;
end
end
%% ==== End of Firefly Algorithm implementation ======