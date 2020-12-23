function [Zk_all, time, MaximumResponses, ground_truth] = tracker(video_path, ...
    img_files, pos1, target_sz, target_sz_file, padding, kernel, ...
    lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization, ground_truth,gt, video) %#ok<INUSL>
%%%%%%%%%%%%%%%%%%%%%   HOG Response ~ Line 150
%  Process_noise = Process_noise_mag * [(dt^2/2)*randn; dt*randn];
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
% This function implements the pipeline for tracking with the KCF (by
% choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
% It is meant to be called by the interface function RUN_TRACKER, which
% sets up the parameters and loads the video information.
%
% Parameters:
%  VIDEO_PATH is location of image files (must end with a slash
%  '/' or '\').
%  IMG_FILES is a cell array of image file names.
%  Zk and TARGET_SZ are the initial position and size of the target
%  (both in format [rows, columns]).
%  PADDING is the additional tracked region, relative to target size.
%  KERNEL is a struct describing the kernel. The field TYPE must be one
%  of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%  POLY_A and POLY_B are parameters for Gaussian and Polynomial kernels.
%  OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%  target, relative to the target size.
%  INTERP_FACTOR is the adaptation rate of the tracker.
%  CELL_SIZE is number of pixels per cell (must be 1 if using raw
%  pixels).
%  FEATURES is a struct describing the used features (see GET_FEATURES).
%  SHOW_VISUALIZATION will show an interactive video if set to true.
%
% Outputs:
%  Zk_allAll is an Nx2 matrix of target positions over time (in the
%  format [rows, columns]).
%  TIME is the tracker execution time, without video loading/rendering.
%
%  Joao F. Henriques, 2014
% If large target, lower the resolution, we don't need that much detail.

 frameNum = numel(img_files);   % Counts number of images in video file
 HOGsigma = 1 ;  % Initialize
 HOGresponseMax=1;
 resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
 if resize_image
  pos1 = floor(pos1 / 2);
  target_sz = floor(target_sz / 2);
  target_sz_file = target_sz_file .* 0.5 ;
  ground_truth = ground_truth .* 0.5 ;
 end

%window size, taking padding into account
window_sz = floor(target_sz * (1 + padding));	
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
if isfield(features, 'deep') && features.deep
 yf = fft2(gaussian_shaped_labels(output_sigma, ceil(window_sz / cell_size)));
else
 yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
end
%  yf is a Gaussian filter with largest value in upper-left of 50x21 matrix
	%store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
 if show_visualization  
  update_visualization = show_video(img_files, video_path, resize_image);
 end
	%note: variables ending with 'f' are in the Fourier domain.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Pre-allocate arrays
Zk_all           = zeros(numel(img_files), 2); % All HOG meas stored
MaximumResponses = zeros(frameNum,1);     % Save Max HoG Est for frames
kalman           = zeros(frameNum,2);     % Save Kalman Gains to File
kcf              = zeros(frameNum,2);     % Save HoG Estimates to File
gtKalman         = zeros(frameNum,2);     % intermediate value only
GT_KalmanDelta   = zeros(frameNum);       % Save GrndTruth / Kalman Delta
%%   End pre-allocation

	for frame = 1:numel(img_files)
	  im = imread([video_path img_files{frame}]);
        if ~isfield(features, 'deep')
          if size(im,3) > 1
           im = rgb2gray(im);
          end
        end 
	   if resize_image
		im = imresize(im, 0.5);
       end
%%%%%%%%%%%%%%% Creates the Ground Truth bounding box for video display %%
GT_Box = [fliplr(ground_truth(frame,:)) - ...
    target_sz_file(frame,[2,1])/2, target_sz_file(frame,[2,1])];

if frame<3
    Zk = pos1 ;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  tic()
	 if frame > 1
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Zk = Xk(1:2)' +nullFactor(1:2); % patch window centered based on Kalman est 
                              % comment this part out to center on HOG
  patch = get_subwindow(im, Zk, window_sz);
  zf = fft2(get_features(patch, features, cell_size, cos_window));
  kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
  response = real(ifft2(model_alphaf .* kzf));  %equation for fast detect

  %  response(:) contains all HOG responses from patch
[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
      
  HOGresponseMax = max(response(:)) ; 
   if frame == 2    % save MaxNumber from Frame 1 to normalize
    normFactor = HOGresponseMax;
   end
    hog{frame} = response;
    cirshft_hog = circshift(response, [24 10]);
    circshiftHogNorm = cirshft_hog ./ normFactor;

MaximumResponses(frame-1,:) = HOGresponseMax;

 if vert_delta > size(zf,1)/2  %wrap to negative 1/2-space of vertical axis
	vert_delta = vert_delta - size(zf,1);  % +25 to -24
 end
 if horiz_delta > size(zf,2) / 2  %same for horizontal axis
	horiz_delta = horiz_delta - size(zf,2); % +10.5 to -9.5
 end
Zk = Zk + cell_size * [vert_delta - 1, horiz_delta - 1];
     end
     
%obtain a subwindow for training at newly estimated target position
patch = get_subwindow(im, Zk, window_sz);
xf = fft2(get_features(patch, features, cell_size, cos_window));
kf = gaussian_correlation(xf, xf, kernel.sigma);
alphaf = yf ./ (kf + lambda);   %equation for fast training

  if frame == 1  %first frame, train with a single image
	model_alphaf = alphaf;
	model_xf = xf;
  else
		         %subsequent frames, interpolate model
model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
  end
  
Zk_all(frame,:) = Zk;   % Save estimates here
  %% Function below prints the frame number & HOG response
  %% on video frames.
  FrameTitles(HOGresponseMax, frame, size(im));
	if show_visualization        
	box = [Zk([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%   KALMAN STARTS HERE  [positionX; positionY; velocityX; velocityY]
if frame == 1
posXYinitial = Zk;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initialize Kalman parameters 
dt  = 1;                   % sampling rate
sw  = [0.1 0.1 0.2 0.2]' ; % standard deviation of wk (process noise)
sv = [0.8 0.8]' ;          % standard deviation of vk (measurement noise)
sa = 0.01  ;               % acceleration noise
R  = diag(sv.^2) ;         % measurement noise COV matrix
t1 = 0.5 * dt^2 ;          
Q = t1 .* [t1  0    dt  0; ...   % process noise COV matrix      
           0  t1    0  dt; ...        
           dt  0    2  0;   ...
           0  2*dt/3 0  2] .* sa ;     
P     =   [0 0 0 0]' ;         % initial error COV matrix
Xk    =   [0 0 0 0]' ;         % initial object location
Pk    =   diag(P) ;      
%% Define update equations in 2-D! (Coefficent matrices): A physics based model for where we expect object to be [state transition (state + velocity)] + [input control (acceleration)]
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]; % State update matrix
H = [1 0 0 0; 0 1 0 0];                     % Measurement update matrix
%% initize estimation variables
KalmanXY_EstFile      = zeros(frameNum*2,1); %  position estimate
KalmanVelocityEstFile = zeros(frameNum*2,1); %  velocity estimate
KalmanGains           = zeros(frameNum,2);
CovUpdates            = zeros(frameNum,4);  
wk = sw * randn(1,frameNum) ;           % wk contains noise for all frames
vk = sv * randn(1,frameNum) ;
% % Modify groundTruth position for starting point at [0,0]
% gtNorm = gt - [Xgt1 Ygt1] ;  % Normalize ground truth here
% Xgt = gtNorm(:,1);            % Divide GT vector by initial XY values
% Ygt = gtNorm(:,2);
% Xgt1n = Xgt(1,1) ;
% Ygt1n = Ygt(2,1) ;
nullFactor = [Zk 0 0] ;
end                                     % end frame=1
Zk_norm = Zk - nullFactor(1:2) ;
%      End Initialization of Kalman variables here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% do the kalman filter   
Xk  =  A * Xk ;              
Pk  =  A * Pk * A' + Q;      
K   =  Pk*H'/(H*Pk*H'+ (R*HOGsigma) );                  
Xk  =  Xk + K*(Zk_norm' - H*Xk);
Pk  = (eye(4)-K*H)*Pk;                 % update COV estimate 
if frame >= 2
% [HOGsigma, Zk] = HOGvariance(response(:), HOGresponseMax, ...
%     frame, Xk, Zk, Zk_norm, nullFactor);
end

    %% Store data
% KalmanXY_EstFile = [KalmanXY_EstFile ; Xk(1:2)] ;
% KalmanVelocityEstFile = [KalmanVelocityEstFile; Xk(3:4)];
%   %%%%%%%%%%%%%%%%%%%%%%%%  Store KALMAN variables
% KalmanGains(frame,:) = [K(2,2) K(3,1)] ;
% CovUpdates(frame,:) = [Pk(1,1) Pk(3,1) Pk(1,3) Pk(3,3)]; 
% KalmanEstTemp = Xk(1:2,1)' ;
% KalmanEstALL(frame,:) = KalmanEstTemp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 stop = update_visualization(frame, box);
    end
Zk = Xk(1:2)' +nullFactor(1:2) ;
 boundingBoxes(Zk, frame, GT_Box, box, padding, target_sz, Xk, ...
     nullFactor) ;
 
 if frame == frameNum
plotsHOGvGT(frame,response,Zk_all,gt,ground_truth,... 
      Zk, video, MaximumResponses) ;
end
 
 
% %% Save Measurements to Spreadsheet
%   if frame == numel(img_files)
%    filename = 'hogData.xlsx';
%    data=cat(2,MaximumResponses,gt,kalman,kcf,GT_KalmanDelta, ...
%             GT_KCF_Delta,KCF_KalmanDelta,KalmanGains,CovUpdates);
%    col_header={'HOGmax','G_Truth X','G_Truth Y' ...
%          'Kalman X','Kalman Y','KCF X','KCF Y','GT_Kalman Delta', ...
%          'GT_KCF Delta','KCF_Kalman Delta','Kalman1', ...
%          'Kalman2','Cov1','Cov2','Cov3','Cov4'};
% %     row_header(1:frame,1) = {'Frame Number'} ;
%    data(1,:) = [];  % Remove first row
% %    xlswrite(filename, data,'Sheet1','B2');
% %    xlswrite(filename,col_header,'Sheet1','B1');
%    gtKalman = num2str(mean(GT_KalmanDelta(:,1)));
%    fprintf('RMS Delta between Ground Truth and Kalman is: %s\n',gtKalman);
%    gtKCF = num2str(mean(GT_KCF_Delta(frame,:)));
%    fprintf('RMS Delta between Ground Truth and HOG is: %s\n',gtKCF);
%    kcfKalman = num2str(mean(KCF_KalmanDelta(:,1)));
%    fprintf('RMS Delta between HOG and Kalman Filter is: %s\n',kcfKalman);
%     figure;
%     plot(data(:,1))
%     grid
%     xlabel('Frame Number');
%     ylabel('Max  HOG  Response')
%     title('HOG Response vs Frame Number')
%     set(gca,'XMinorTick','on','YMinorTick','on')
%   end
    end
end

