function [  ] = plotsHOGvGT( frame,response,HOGestAll,gt, ...
    ground_truth,HOGest, video, MaximumResponses )
% Function designed to plot HOG response and Ground Truth on same axis
%   
frameNum = length(ground_truth) ;
f = 1:frameNum-1 ;
HOGestRMS = sqrt(HOGestAll(1:end-1,1).^2 + HOGestAll(1:end-1,2).^2);
gtRMS = sqrt(ground_truth(1:end-1,1).^2 + ground_truth(1:end-1,2).^2);
distance = sqrt( (HOGestAll(1:end-1,1)-ground_truth(1:end-1,1)).^2 + ...
    (HOGestAll(1:end-1,2)-ground_truth(1:end-1,2)).^2) ;

fpath = 'C:/GroundTruth_HogPlots' ;
fpath1 = 'C:/GroundTruth_HogPlots/Figures' ;
fpath2 = 'C:/GroundTruth_HogPlots/GroundTruthHogResponses1';
fpath3 = 'C:/GroundTruth_HogPlots/Figures1' ;
fpath4 = 'C:/ResearchPlots/ScatterPlots' ;

close
vidName = strcat(video,'_scatterplot');
figure
 plot(MaximumResponses(1:end-1),distance,'*')
 xlabel('Maximum HOG Response per Frame');
 ylabel('Error (HOG-GroundTruth RMS Difference)');
 title([video ' Video: Error vs Max HOG Response']);
%  saveas(gcf, fullfile(fpath4, vidName),'png');



% figure
% plot(f,MaximumResponses(1:end-1),'b','LineWidth',1.25)
% yyaxis left
%  xlabel('Frame Number');
%  ylabel('Maximum HOG Response');
% hold on
% yyaxis right
%  plot(f,distance,'r','LineWidth',1.25)
%  ylabel('RMS Distance: HOG Estimate vs Ground Truth') ;
%  title([video ' Video: Max HOG Response vs HOG Position Error']);
%  legend('Max HOG Response', 'HOG Position Error')
%  set(gca,'Color',[.95 .97 .95]);
%  saveas(gcf, fullfile(fpath, video),'png');
%  saveas(gcf, fullfile(fpath1, video),'fig');
%  %  *** Plot below show Maximum HOG response for each frame
%  %  vs the RMS distance (between HOG position estimate and
%  %  Ground Truth
%  %  *******************************************************
%  figure
%  plot(f,HOGestRMS,'b',f,gtRMS,'r-.','LineWidth', 0.75)
% %     hold on
% yyaxis left
%  xlabel('Frame Number');
%  ylabel('Object Position (XY RMS)');
% yyaxis right
%  plot(f,distance,'g')
%  ylabel('RMS Distance HOG v Ground Truth') ;
%  title([video ' Video - Ground Truth vs HOG Estimate']);
%  legend('HOG estimate', 'Ground Truth', 'RMS Distance')
%  set(gca,'Color',[.95 .97 .95]);
%    saveas(gcf, fullfile(fpath2, video),'png');
%    saveas(gcf, fullfile(fpath3, video),'fig');
%  ******************************************************* 
end


 
%     clf

