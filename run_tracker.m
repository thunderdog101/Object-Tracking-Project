%
function [precision, fps] = run_tracker(video,kernel_type, ...
    feature_type,show_visualization,show_plots) %#ok<INUSD>
    clc
%   for j=1:24
   base_path = 'D:/Research/Videos_VOT14/';
   videoList = allVideos() ;   % reads function for all videos in dBase
   
	%default settings
   show_visualization = 1 ;           % displays video on-screen if '1'
   show_plots = 1 ;                   % shows precision plots if '1'
   

for k = 1:1  % 16:16  % 13:numel(videoList)  do not use 12 (David)
  video = char(videoList(k)) ; 

[features,kernel,kernel_type,feature_type,interp_factor,padding, ...
    lambda,output_sigma_factor,cell_size] = hogKernel_parameters; %#ok<ASGLU>

	% ground_truth=object center; gt=top-left of object bounding box
[img_files, pos1, target_sz, target_sz_file, ground_truth, ...
    video_path, gt] = load_video_info(base_path, video);

    % TRACKER FUNCTION - Most processing done here
[Zk_all, MaximumResponses, ground_truth] = tracker(video_path, ...
    img_files, pos1, target_sz, target_sz_file, padding, ...
    kernel, lambda, output_sigma_factor, interp_factor, ...
	cell_size, features, show_visualization, ground_truth, ...
    gt, video);

precisions = precision_plot(Zk_all, ground_truth, ...
    video, show_plots, MaximumResponses);
        

   fid = fopen('kcf_positions.txt', 'wt');
   fprintf(fid, ' %d  %d \n', Zk_all');
   fclose(fid);
                  %  save kcf_positions.txt positions -ascii

clear results

        results{1}.type = 'rect';
        x = Zk_all(:,2);
        y = Zk_all(:,1);
        w = target_sz(2) * ones(size(x, 1), 1);
        h = target_sz(1) * ones(size(x, 1), 1);
        results{1}.res = [x-w/2, y-h/2, w, h];
        
        results{1}.startFame = 1;
        results{1}.annoBegin = 1;
        results{1}.len = size(x, 1);
        
        frames = {'David', 300, 465;%770
            'Football1', 1, 74;
            'Freeman3', 1, 460;
            'Freeman4', 1, 283};
        
        idx = find(strcmpi(video, frames(:,1)));
        
        if ~isempty(idx)
            results{1}.startFame = frames{idx, 2};
            results{1}.len = frames{idx, 3} - frames{idx, 2}; +1;
        end
        res_path = ['KCF_results_' feature_type '/'];
        if ~isfolder(res_path)
            mkdir(res_path);
        end
     save([res_path lower(video) '_kcf_' feature_type '8.mat'], 'results');
        
end         
close all
end
