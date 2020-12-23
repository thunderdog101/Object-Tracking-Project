function [img_files, pos1, target_sz, target_sz_file, ground_truth, video_path, gt] = load_video_info(base_path, video)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%

	%see if there's a suffix, specifying one of multiple targets, for
	%example the dot and number in 'Jogging.1' or 'Jogging.2'.
		suffix = '';

	%full path to the video's files
	video_path = [base_path video '/'];

	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path 'groundtruth_rect' suffix '.txt'];
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height]
  try
    ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
  catch %ok, try different format  (no commas)
    frewind(f) ;
    ground_truth = textscan(f, '%f %f %f %f') ;
  end  
	ground_truth = cat(2, ground_truth{:});
	fclose(f);
	
%             fid = fopen('kcf_groundTruth.txt', 'wt');
%             fprintf(fid, ' %d  %d  %d  %d \n', ground_truth');
%             fclose(fid);  
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];
    target_sz_file = [ground_truth(:,4), ground_truth(:,3)];
	pos1 = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
	
	%from now on, work in the subfolder where all the images are
	video_path = [video_path 'img/'];
	
	%for these sequences, we must limit ourselves to a range of frames.
	%for all others, we just load all png/jpg files in the folder.
	frames = {'David', 300, 465; %770
			  'Football1', 1, 74;
			  'Freeman3', 1, 460;
			  'Freeman4', 1, 283};
	
	idx = find(strcmpi(video, frames(:,1)));
	
	if isempty(idx),
		%general case, just list all images
		img_files = dir([video_path '*.png']);
		if isempty(img_files),
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
		end
		img_files = sort({img_files.name});
        target_sz = [ground_truth(1,4), ground_truth(1,3)];
	    pos1 = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
    else
        
        target_sz = [ground_truth(frames{idx, 2},4), ground_truth(frames{idx, 2},3)];
        pos1 = [ground_truth(frames{idx, 2},2), ground_truth(frames{idx, 2},1)] + floor(target_sz/2);
		%list specified frames. try png first, then jpg.
		if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
			
		elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
			
		else
			error('No image files to load.')
		end
		
		img_files = cellstr(img_files);
    end
    
    	if size(ground_truth,1) == 1,
		%we have ground truth for the first frame only (initial position)
		ground_truth = [];
	else
		%store positions instead of boxes
        gt = ground_truth(:,[1,2]);
		ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
	end
	
end

