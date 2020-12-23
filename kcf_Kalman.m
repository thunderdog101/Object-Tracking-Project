clear all
clc
empty_pic = ones(576,432);   % creates white canvass background
figure
imshow(empty_pic);
title('KCF Approximation (red) & Ground Truth (blue)');
xlabel(' Pixels ');  ylabel(' Pixels ');
xlim([0 432]);  ylim([0 576]);
% legend ('Tracker', 'Ground Truth');
hold on;
axis on;

load kcf_positions.txt ;    % computed from Henriques KCF example
load kcf_groundTruth.txt ;
pos = kcf_positions;
pos_truth = horzcat(floor(kcf_groundTruth(:,1)+kcf_groundTruth(:,3)/2), ...
          floor(kcf_groundTruth(:,2)+kcf_groundTruth(:,4)/2));
for i = 1:length(pos)
h = rectangle('Position', [pos(i,2),pos(i,1),34,81]);
h.LineWidth = 1; h.EdgeColor = [1 0 0]; % h.LineStyle = '--' ;
h1 = rectangle('Position', [pos_truth(i,1),pos_truth(i,2), 34, 81]);
h1.LineWidth=1; h1.EdgeColor = [0 0 1];
pause(0.05)
h.EdgeColor =[1 1 1];
h1.EdgeColor =[1 1 1];
% fprintf('Iteration %d ', i, 'of 765'); 
i
end
clc
pos(:,[1 2]) = pos(:,[2 1]);
Precision = sum( sqrt((pos_truth - pos).^2));


