function precisions = precision_plot(positions, ground_truth, ...
    videoTitle, show, MaximumResponses)
%PRECISION_PLOT
%   Calculates precision for a series of distance thresholds (percentage of
%   frames where the distance to the ground truth is within the threshold).
%   The results are shown in a new figure if SHOW is true.
%
%   Accepts positions and ground truth as Nx2 matrices (for N frames), and
%   a title string.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
% Video list is indexed so that spreadsheet measurements are saved in
%  separate columns.
videoList = {'basketball', 'bolt', 'boy', 'car4', 'carDark', 'carScale', ...
	'coke', 'couple', 'crossing', 'david2', 'david3', 'david', 'deer', ...
	'dog1', 'doll', 'dudek', 'faceocc1', 'faceocc2', 'fish', 'fleetface', ...
	'football', 'football1', 'freeman1', 'freeman3', 'freeman4', 'girl', ...
	'ironman', 'jogging', 'jumping', 'lemming', 'liquor', 'matrix', ...
    'mhyang', 'motorRolling', 'mountainBike', 'shaking', 'singer1', ...
	'singer2', 'skating1', 'skiing', 'soccer', 'subway', 'suv', 'sylvester', ...
	'tiger1', 'tiger2', 'trellis', 'walking', 'walking2', 'woman'};
    
	max_threshold = 50;  %used for graphs in the paper
	precisions = zeros(max_threshold, 1);
	
	if size(positions,1) ~= size(ground_truth,1),
% 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
		
		%just ignore any extra frames, in either results or ground truth
		n = min(size(positions,1), size(ground_truth,1));
		positions(n+1:end,:) = [];
		ground_truth(n+1:end,:) = [];
	end
	
	%calculate distances to ground truth over all frames
	distances = sqrt((positions(:,1) - ground_truth(:,1)).^2 + ...
				 	 (positions(:,2) - ground_truth(:,2)).^2);
	distances(isnan(distances)) = [];

	%compute precisions
	for p = 1:max_threshold,
		precisions(p) = nnz(distances <= p) / numel(distances);
         if p == 20
         thresholdPercent = precisions(p);
         end
	end
	
	%plot the precisions 
% Max HOG Response; Deltas   
	if show == 1
    capTitle=strcat('''',upper(videoTitle(1)),videoTitle(2:end),'''');
    hFig = figure;
%     hFig = figure(1);    % Left,Bottom,Width,Height
    set(hFig, 'position', [250 80 800 500]) 
    subplot(2,2,1);
    plot(MaximumResponses(1:end-1,1))
    grid
    xlabel('Frame Number');
    ylabel('Max  HOG  Response')
    title(strcat(capTitle,'   Video Plot'))
    set(gca,'XMinorTick','on','YMinorTick','on')
      subplot(2,2,3);
% 		figure('NumberTitle','off', 'Name',['Precisions - ' videoTitle])
        plot(distances, 'k-', 'LineWidth',0.8)
		xlabel('Frame Number'), ylabel('Distance')
        title('Distance  (GroundTruth v Tracker)')
%         b=sprintf(' %.2f%% <= 20 pixel threshold',thresholdPercent*100);
%         legend(b);
        grid
        set(gca,'XMinorTick','on','YMinorTick','on')
   %%%%%%%%%%%%%%%%%%%%%%%
         subplot(2,2,2);
% 		figure('NumberTitle','off', 'Name',['Precisions - ' videoTitle])
 		plot(precisions, 'k-', 'LineWidth',2)
%         plot(distances, 'k-', 'LineWidth',1)
		xlabel('Threshold (pixels)'), ylabel('Precision')
        title('Tracker v GroundTruth Dist <= Threshold')
        b=sprintf(' %.2f%% <= 20 pixel threshold',thresholdPercent*100);
        legend(b);
        grid
        set(gca,'XMinorTick','on','YMinorTick','on')
    end
    
% filename = 'test.xlsx' ;
% worksheet1_name='Precision';  worksheet2_name='HOG Maximums';
% sheet1=1; sheet2=2;
% xlsdir = sprintf(['C:\\Program Files\\MATLAB\\R2017a\\bin' ...
%     '\\Asilomar-Research\\KCF Practice\\KCF-master_Software']);
% filename = strcat(xlsdir,'\',filename);
% 
% warning( 'off', 'MATLAB:xlswrite:AddSheet' );
% xlswrite(filename,{'Precision'},sheet1, 'A1');
% xlswrite(filename, {videoTitle},sheet1,'A2');
% xlswrite(filename,precisions,sheet1,'A3');
% 
% xlswrite(filename, {'HOG Maximums'},sheet1,'B1');
% % xlswrite(filename, {videoTitle},sheet1,'B2');
% xlswrite(filename,MaximumResponses(1:end-1,1),sheet1,'B3');

end

