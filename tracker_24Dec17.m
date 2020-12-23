function [HOGestAll, time, MaximumResponses] = tracker(video_path, img_files, HOGest, target_sz, ...
	target_sz_file, padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization, ground_truth)
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
%  HOGest and TARGET_SZ are the initial position and size of the target
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
%  HOGestAll is an Nx2 matrix of target positions over time (in the
%  format [rows, columns]).
%  TIME is the tracker execution time, without video loading/rendering.
%
%  Joao F. Henriques, 2014
% If large target, lower the resolution, we don't need that much detail.
 frameNum = numel(img_files);   % Counts number of images in video file
 HOGresponseMax=1;
 resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
 if resize_image
  HOGest = floor(HOGest / 2);
  target_sz = floor(target_sz / 2);
 end

%window size, taking padding into account
window_sz = floor(target_sz * (1 + padding));
	
% we could choose a size that is a power of two, for better FFT
% performance. in practice it is slower, due to the larger window size.
% window_sz = 2 .^ nextpow2(window_sz);
	
% create regression labels, gaussian shaped, with a bandwidth
% proportional to target size   
    
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
if isfield(features, 'deep') && features.deep
 yf = fft2(gaussian_shaped_labels(output_sigma, ceil(window_sz / cell_size)));
%   sz = ceil(window_sz/cell_size)-1+4-4;
%   yf = fft2(gaussian_shaped_labels(output_sigma, sz));
else
 yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
end
%  yf is a Gaussian filter with largest value in upper-left of 50x21 matrix
	%store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
 if show_visualization  %create video interface
  update_visualization = show_video(img_files, video_path, resize_image);
 end
	%note: variables ending with 'f' are in the Fourier domain.
  time = 0;  %to calculate FPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Pre-allocate arrays
HOGestAll = zeros(numel(img_files), 2); % to calculate precision
MaximumResponses = zeros(frameNum,1);   % Save Max HoG Est for each frame
gt     = zeros(frameNum,2);             % Save Ground Truth to File
kalman = zeros(frameNum,2);             % Save Kalman Gains to File
kcf    = zeros(frameNum,2);             % Save HoG Estimates to File
gtKalman = zeros(frameNum,2);           % intermediate value only
GT_KalmanDelta = zeros(frameNum);       % Save GrndTruth / Kalman Delta
% gtKCF   = zeros(frameNum,2)
% GT_KCF_Delta = zeros(frameNum);
% kcfKalman = zeros(frameNum,2);
% KCF_KalmanDelta = zeros(frameNum);
%%   End pre-allocation
	for frame = 1:numel(img_files)
	  im = imread([video_path img_files{frame}]);
        if ~isfield(features, 'deep')
          if size(im,3) > 1
           im = rgb2gray(im);
          end
        end  
%    im = Occlusion(im, 10, 10, frame, HOGest, window_sz);
	   if resize_image
		im = imresize(im, 0.5);
       end
%  im = im*3; 
%  Ground_Truth has target center point (XY) for each frame
%  GT_Box- Horiz-Vert corner of bounding box; Width-Height if box
GT_Box = [fliplr(ground_truth(frame,:)) - ...
    target_sz_file(frame,[2,1])/2, target_sz_file(frame,[2,1])];

% rg = insertShape(im,'FilledRectangle',GT_Box,'Color','w');
%   *******************************************************
%  Add noise to image;  Noise can be Gaussian, Salt/Pepper, Speckle
%  Format: (image,'gaussian',Mean,Sigma); 
%         (I,'salt & pepper',d)  d=noise intensity
%         (I,'speckle',v) adds multiplicative noise v is variance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
noise='gaussian'; mean1=0; sigma1=1; 
% if frame ==100
% %     z=rectangle('Position',GT_Box,'FaceColor','w');
%    im=imnoise(im, 'gaussian', 0, sigma);
% %     im = insertShape(im,'FilledRectangle',GT_Box,'Color','w');
% end
% noise='salt & pepper';  d=.1;
% noise='speckle'; v=.1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  tic()
	 if frame > 1
    if frame==269   % 724
     f=3;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Noise_Add(frame, noise_type, param1, param2); 
%     if frame >= 420 & frame <=440
%       Noise_Add = text(160,15,strcat(upper(noise),{' Noise Added   '}, ...
%          {' Mean= '},num2str(mean1),{'   Sigma= '},num2str(sigma)));
%       Noise_Add.Color='red';
%        im=imnoise(im, 'gaussian', 0, sigma); 
%     end
  % obtain a subwindow for detection at the position from last
  % frame, and convert to Fourier domain (its size is unchanged)
% patch- Search Window - HOGest is target center (YX) position

  patch = get_subwindow(im, HOGest, window_sz);
  zf = fft2(get_features(patch, features, cell_size, cos_window));
			
	%calculate response of the classifier at all shifts
  switch kernel.type
	case 'gaussian'
	kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
	case 'polynomial'
	kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
    case 'linear'
	kzf = linear_correlation(zf, model_xf);
	end
response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection

	%target location is at the maximum response. we must take into
	%account the fact that, if the target doesn't move, the peak
	%will appear at the top-left corner, not at the center (this is
	%discussed in the paper). the responses wrap around cyclically.
    % [vert,horz] is Row,Col position in response matrix with
    % largest value
 [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
      
  HOGresponseMax = max(response(:)) ; 
   if frame == 2    % save MaxNumber from Frame 1 to normalize
    normFactor = HOGresponseMax;
   end
    hog{frame} = response;
    cirshft_hog = circshift(response, [24 10]);
    circshiftHogNorm = cirshft_hog ./ normFactor;
% surf(circshiftHogNorm);
% colorbar ;
% if frame > 2
%  figure(frame)
%  surf(circshiftHogNorm);
%  colorbar;
% end
%%  Measurements saved with each iteration into arrays...
%   gt==GroundTruth;  kalman==KalmanEstimate;  kcf==HOGestimate
%
% circshiftHogZoomNorm = circshiftHogNorm(21:29, 7:15);%save HOGresponseMax
MaximumResponses(frame-1,:) = HOGresponseMax;
gt(frame-1,:) = [GT_Box(1) GT_Box(2)] ;
%     gtRMS(frame,:) = sqrt(GT_Box(1)^2 + GT_Box(2)^2);
kalman(frame-1,:) = [round(Kalman_Box(1)) round(Kalman_Box(2))];
%     kalmanRMS(frame,:) = sqrt(Kalman_Box(1)^2 + Kalman_Box(2)^2);
%     kcf(frame,:) = [HOGest(2) HOGest(1)] ;
 kcf(frame-1,:) = round([HOGest([2,1]) - target_sz([2,1])/2]);
%     kcfRMS(frame,:) = sqrt(kcf(frame,1)^2 + kcf(frame,2)^2) ;
clc
gtKalman(frame-1,:)=(gt(frame-1,:)-kalman(frame-1,:)).^2 ;
GT_KalmanDelta(frame-1,1) = sqrt(gtKalman(frame-1,1)+gtKalman(frame-1,2));
 
gtKCF(frame,:)= (gt(frame,:)-kcf(frame,:)).^2 ;
GT_KCF_Delta(frame,:)= sqrt(gtKCF(frame,1)+gtKCF(frame,2));
 
kcfKalman(frame,:)= (kcf(frame,:)-kalman(frame,:)).^2 ;
KCF_KalmanDelta(frame,1)= sqrt(kcfKalman(frame,1)+kcfKalman(frame,2)) ;
  % In this case, if vert_delta > 25  or horiz_delta > 10.5
 if vert_delta > size(zf,1)/2  %wrap to negative 1/2-space of vertical axis
	vert_delta = vert_delta - size(zf,1);  % +25 to -24
 end
 if horiz_delta > size(zf,2) / 2  %same for horizontal axis
	horiz_delta = horiz_delta - size(zf,2); % +10.5 to -9.5
 end
HOGest = HOGest + cell_size * [vert_delta - 1, horiz_delta - 1];
     end
     
%obtain a subwindow for training at newly estimated target position
patch = get_subwindow(im, HOGest, window_sz);
%  FHOG is located in get_features function
xf = fft2(get_features(patch, features, cell_size, cos_window));

%Kernel Ridge Regression, calculate alphas (in Fourier domain)
  switch kernel.type
	case 'gaussian'
	  kf = gaussian_correlation(xf, xf, kernel.sigma);
	case 'polynomial'
	  kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
	case 'linear'
	  kf = linear_correlation(xf, xf);
  end
	alphaf = yf ./ (kf + lambda);   %equation for fast training

  if frame == 1  %first frame, train with a single image
	model_alphaf = alphaf;
	model_xf = xf;
  else
		%subsequent frames, interpolate model
	model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
	model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
  end

		%save position and timing
HOGestAll(frame,:) = HOGest;   % Estimate of object positions for frames in video
time = time + toc();

        if  frame == 725
            fid = fopen('kcf_positions.txt', 'wt');
            fprintf(fid, ' %d  %d \n', HOGestAll');
            fclose(fid);
%             save kcf_positions.txt positions -ascii
        end
  %% Function below prints the frame number & HOG response
  %% on video frames.
  FrameTitles(HOGresponseMax, frame);
  
%   if frame ~= 1
%        delete(FrameNum);
%        delete(HOGnum) ;
%   end   
%       str = strcat([' Frame ', ' ',num2str(frame)]);
%       FrameNum = text(500,15,str);
%       FrameNum.Color='red';
%    if frame==1  
%    str1 = ' ' ;
%    end
%    if frame ~= 1
%    txt = sprintf(' %.2f ', HOGresponseMax);    
%    str1 = strcat([' HOG Response ', ' ',txt]);
%    end
%    HOGnum = text(430,35,str1);
%    HOGnum.Color='red';
		%visualization
	if show_visualization        
	box = [HOGest([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
%   end
%             box1 = [box(1)+10,box(2),box(3),box(4)];
%             box1 = [box(1)+10,box(2),box(3),box(4)];
%             box2 = [150,150,40,85];
%             delete(rec)
%
%   KALMAN STARTS HERE
%%   load ground truth from Basketball video file from Henriques
% load('kcf_positions.txt') ;
% HOGest(:,[1 2]) = kcf_positions(:,[2 1]); % x  and y shifted since Henriques uses [y x height width]
% CM_idx = [(groundtruth_rect(:,1)+floor(groundtruth_rect(:,3)/2)), ...
%     (groundtruth_rect(:,2)+floor(groundtruth_rect(:,4)/2))];
if frame == 1
posXYinitial = HOGest;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define main variables
                  % S_frame = 1;     
dt      = 1;      % our sampling rate
u       = .005;   % define acceleration magnitude
%  Init state--4 components: [positionX; positionY; velocityX; velocityY]
Q          = [posXYinitial(1); posXYinitial(2); 0; 0]; 
KalmanEst  = Q;             % Estimate initial object location 
Ex_mag     = 0.1;           % Kalman Process Noise:
KalmanMeasNoiseX = 1;       % Measurement noise for X-direction 
KalmanMeasNoiseY = 1;       % Measurement noise in  Y-direction 
Ez = [KalmanMeasNoiseX 0; 0 KalmanMeasNoiseY];
Ex = [dt^4/4 0 dt^3/2 0; ...
      0 dt^4/4 0 dt^3/2; ...
      dt^3/2 0 dt^2 0;   ...
      0 dt^3/2 0 dt^2] .* Ex_mag^2;    % Kalman Process Noise
P =   Ex;   % estimate of initial position variance (covariance matrix)

%% Define update equations in 2-D! (Coefficent matrices): A physics based model for where we expect object to be [state transition (state + velocity)] + [input control (acceleration)]
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]; % State update matrice
B = [(dt^2/2); (dt^2/2); dt; dt];
C = [1 0 0 0; 0 1 0 0];                     % Kalman measurement function
%% initize result variables
HOGestTranspose = [];                 % object track from HOG tracker
%% initize estimation variables
KalmanXY_EstFile = zeros(frameNum*2,1); %  position estimate
KalmanVelocityEstFile   = zeros(frameNum*2,1); %  velocity estimate
KalmanGains    = zeros(frameNum,2);
CovUpdates     = zeros(frameNum,4);  
               % r = 5; % r is the radius of the plotting circle
               % j=0:.01:2*pi; %to make the plotting circle
end   
%      End Initialization of Kalman variables here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i = S_frame:length(HOGest)
    
%     load the image
%     img_tmp = double(imread(f_list(t).name));
%     img = img_tmp(:,:,1);
%     load the given tracking
%     HOGestTranspose(:,i) = HOGestTranspose(i,2); % Contains HOGest'
HOGestTranspose(:,frame) = HOGest' ;  % Equals HOGest'  (transposed)
    
    %% do the kalman filter   
    
    % Predict next state of object with the last state and predicted motion.
        %%%  Project the state ahead
 if frame==269   % 724
         f=3;
     end
%%%%%% Capture the HOG response from reference
HOGsigma = 1 ;
% if frame > 1
%     if frame == 2
%     maxHOGnorm = 1/(HOGresponseMax+1e-10);
%     end
[HOGsigma, HOGest] = HOGvariance(HOGresponseMax, frame, KalmanEst, HOGest);
% HOGsigma=1 ;
% end
KalmanEst = A * KalmanEst + B * u;   % KalmanEst initially is 1st frame GT
P = A * P * A' + (Ex*HOGsigma);      % P is covariance matrix...predicted
K = P*C'/(C*P*C'+Ez);                % Kalman Gain    
 if ~isnan(HOGestTranspose(:,frame)) % Update state & measurement estimate
  KalmanEst = KalmanEst + K*(HOGestTranspose(:,frame) - C*KalmanEst);
 end
P =  (eye(4)-K*C)*P;                 % update COV estimate    
    %% Store data
KalmanXY_EstFile = [KalmanXY_EstFile ; KalmanEst(1:2)] ;
KalmanVelocityEstFile = [KalmanVelocityEstFile; KalmanEst(3:4)];
  %%%%%%%%%%%%%%%%%%%%%%%%  Store KALMAN variables
KalmanGains(frame,:) = [K(2,2) K(3,1)] ;
CovUpdates(frame,:) = [P(1,1) P(3,1) P(1,3) P(3,3)]; 
    
 %   box2 = [KalmanEst([2,1]) - target_sz([2,1])'/2 ; target_sz([2,1])']';
  %%%%%%%%%%%%%%%%%%%%%%%
  %  KALMAN STOPS HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 stop = update_visualization(frame, box);
%   imshow(im); 
    if frame ~= 1
       delete(gt_rec);
       delete(gt_title);
       delete(HoG_title);
       delete(Kalman_rec);
       delete(Kalman_title);
       delete(SearchTitle);
       delete(sb);
    end
 %% GROUND TRUTH Bounding Box and Title 
   gt_rec=rectangle('Position',GT_Box,'EdgeColor','r', ...
       'LineWidth',1);
   gt_title = text(GT_Box(1)+20,GT_Box(2)-6,'GT');
   gt_title.Color='r';
%% HoG Bounding Box and Title
   HoG_title = text(box(1)-20,box(2)-6,'HoG');
   HoG_title.Color='g' ;
%% PATCH Bounding Box and Title  (Search window)
   SearchBox =  [HOGest(2) - (padding+1)*target_sz(2)/2  ...
       HOGest(1) - (padding+1)*target_sz(1)/2 (padding+1)*target_sz([2,1])];
   sb=rectangle('Position',SearchBox, 'EdgeColor', 'magenta','LineWidth',1);
   sbTitle = ['Search Window ' num2str(padding+1) 'x'];
   SearchTitle = text(SearchBox(1)-10,SearchBox(2)-6,sbTitle);
   SearchTitle.Color='magenta';
%% KALMAN Bounding Box and Title
   Kalman_Box=[KalmanEst(2)-target_sz(2)/2 ...
       KalmanEst(1)-target_sz(1)/2 target_sz([2,1])];
   Kalman_rec=rectangle('Position',Kalman_Box, ...
       'EdgeColor','yellow','LineWidth',1);
   Kalman_title = text(Kalman_Box(1)-10,Kalman_Box(2)+90,'Kalman');
   Kalman_title.Color='yellow';
%%
%  im = insertShape(im,'FilledRectangle',GT_Box,'Color','w');           
	if stop, break, end  %user pressed Esc, stop early
	drawnow
% 			pause(0.05)  %uncomment to run slower
    end
 %  SAVE parameters here
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save Measurements to Spreadsheet
  if frame == numel(img_files)
   filename = 'hogData.xlsx';
   data=cat(2,MaximumResponses,gt,kalman,kcf,GT_KalmanDelta, ...
            GT_KCF_Delta,KCF_KalmanDelta,KalmanGains,CovUpdates);
   col_header={'HOGmax','G_Truth X','G_Truth Y' ...
         'Kalman X','Kalman Y','KCF X','KCF Y','GT_Kalman Delta', ...
         'GT_KCF Delta','KCF_Kalman Delta','Kalman1', ...
         'Kalman2','Cov1','Cov2','Cov3','Cov4'};
%     row_header(1:frame,1) = {'Frame Number'} ;
   data(1,:) = [];  % Remove first row
   xlswrite(filename, data,'Sheet1','B2');
   xlswrite(filename,col_header,'Sheet1','B1');
   gtKalman = num2str(mean(GT_KalmanDelta(:,1)));
   fprintf('RMS Delta between Ground Truth and Kalman is: %s\n',gtKalman);
   gtKCF = num2str(mean(GT_KCF_Delta(frame,:)));
   fprintf('RMS Delta between Ground Truth and HOG is: %s\n',gtKCF);
   kcfKalman = num2str(mean(KCF_KalmanDelta(:,1)));
   fprintf('RMS Delta between HOG and Kalman Filter is: %s\n',kcfKalman);
%     figure;
%     plot(data(:,1))
%     grid
%     xlabel('Frame Number');
%     ylabel('Max  HOG  Response')
%     title('HOG Response vs Frame Number')
%     set(gca,'XMinorTick','on','YMinorTick','on')
  end
% 	end
	if resize_image
	HOGestAll = HOGestAll * 2;
	end
end

