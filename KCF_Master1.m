clear all
clc
%                                         ************* show_visualization, show_plots
 [precision, fps] = run_tracker('choose', 'gaussian', 'hog', false, false);

 video_name = choose_video('C:\Videos');
 [img_files, pos, target_sz, ground_truth, video_path] = load_video_info('C:\Videos', video_name);
% 
% resize_image = true;
 update_visualization_func = show_video(img_files, video_path, true);
% %store one instance per frame
% 	num_frames = numel(img_files);
% 	boxes = cell(num_frames,1);
% 
% 	%create window
% 	[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
% 	set(fig_h, 'Number','off', 'Name', ['Tracker - ' video_path])
% 	axis off;