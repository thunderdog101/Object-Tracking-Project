clear all
clc
%  HOG Representation
img = imread('C:\Videos\Basketball\img\0002.jpg');
img = rgb2gray(img);
[featureVector,hogVisualization] = extractHOGFeatures(img);
% figure;
% imshow(img);
% hold on;
% plot(hogVisualization);

% [hog1,visualization] = extractHOGFeatures(img,'CellSize',[32 32]);
% subplot(1,2,1);
% imshow(img);
% subplot(1,2,2);
% plot(visualization);

% x = randn(10000,1);
% x = [1 1 1 1 2 3 3 3 4 4 5 6 6 6 6 6];
% h = histogram(x)
% h.FaceColor = 'red' ;
% % h.NumBins = 9;

I = [1 2 2 4; 5 6 9 8; 9 3 11 1; 1 6 3 4];
BW = edge(I, 'Sobel')