% run_fa_mincon.m
% Script to run the Firefly Algorithm (fa_mincon)

% Set up algorithm-specific parameters (if necessary)
global W_dim; W_dim = [10 10];  % Example dimensions, adjust as needed
global para; para = [100, 500, 0.5, 0.2, 1];  % Example parameters [fire_fly_num, iter_num, alpha, beta, gamma]

% Call the fa_mincon function to start the optimization
fa_mincon();

% Additional code here if needed, such as post-processing of results
