clear all
clc
mu1 = 0;        m1=num2str(mu1);    % Process Noise
sigma1 = 1.5;   s1=num2str(sigma1);
mu2 = 2;        m2=num2str(mu2);    % Measurement Noise
sigma2 = 2.5;   s2=num2str(sigma2);
mu3 = 1;     m3=num2str(mu3);       % HOG Mean
sigma3 = 1;  s3=num2str(sigma3);    % HOG Variance
s1sq = sigma1^2; s2sq = sigma2^2;  s3sq = sigma3^2;
% KALMAN Combination of Gaussians shown below
meanK = (mu1*s2sq + mu2*s1sq) / (s1sq + s2sq); % Mean for Kalman
sK2 = (s1sq*s2sq) / (s1sq + s2sq);             % Sigma^2 for Kalman
sK=sqrt(sK2);                                  % Sigma for Kalman
mucomb = num2str(meanK);  scomb=num2str(sK2);
% 
mKH = (mu3*sK2 + meanK*s3sq) / (s3sq + sK2);   % Mean for HOG_Kalman
sKH2 = (s3sq*sK2) / (s3sq + sK2);              % Sigma^2 for HOG_Kal
sKH = sqrt(sKH2);                              % Sigma for HOG_Kalman
mucombHOGkalman = num2str(mKH);  scomb1=num2str(sK2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = -4*sigma2:1e-3:4*sigma2; 
y1 = pdf('normal', x, mu1, sigma1);
y2 = pdf('normal', x, mu2, sigma2);
y3 = pdf('normal', x, meanK, sK2);
y4 = pdf('normal', x, mKH, sKH2);
plot(x, y1,'LineWidth',1.5)
hold on 
plot(x, y2,'m','LineWidth',1.5)
plot(x, y3,'g','LineWidth', 2)
plot(x, y4,'r', 'LineWidth', 2)
% legend(['\mu_1=' m1 '  \sigma_1=' s1],['\mu_2=' m2 '  \sigma_2=' s2], ...
%        ['\mu_3=' mucomb '  \sigma_3=' scomb]) ;
   legend('Kalman Proc N_o','Kalman Meas N_o', ...
       'Kalman Comb N_o', 'HOG+Kalman') ;
title('Density functions : Kalman + HOG')
xticks([-4*sigma2:1:4*sigma2]);
str = sprintf(['Mean1 = %1.2f  Sigma1 = %1.2f \nMean2 = %1.2f' ...
 '   Sigma2 = %1.2f\nHOG Mean = %1.2f   HOG Sigma = %1.2f  \n\n' ... 
 'Kalman Mean =    %1.2f     Kalman Sigma =    %1.2f' ...
 '  \nHogKalman Mean = %1.2f     HogKalman Sigma = %1.2f'],mu1, ...
     sigma1,mu2,sigma2,mu3,sigma3,meanK,sK,mKH,sKH) 
   


