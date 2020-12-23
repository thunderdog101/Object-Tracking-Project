clear all
clc

%      'C:\Videos\Basketball\img\0001.jpg'

 I=imread('C:/Videos/Basketball/img/0001.jpg'/255,[480 640]);
%  tic, for i=1:100, H=fhog(I,8,9); end; disp(100/toc) % >125 fps
%  figure(1); 
%  im(I); 
%  V=hogDraw(H,25,1); 
%  figure(2); 
%  im(V)

%  I=imResample(single(imread('peppers.png'))/255,[480 640]);
%  tic, for i=1:100, H=fhog(I,8,9); end; disp(100/toc) % >125 fps
%  figure(1); im(I); V=hogDraw(H,25,1); figure(2); im(V)