function [ sv2 ] = HOGvariance( response, HOGresponseMax, ...
    frame, Xk, Zk, sv2old, posDeltas)
% This function converts HOG confidence factor into a change in the
%  process noise variation [Q]
%   
persistent maxHOGnorm ;
HOGvar = 1 ;
maxHOGnorm = 1;


if frame > 1
%     if frame == 2
%     maxHOGnorm = 1/(HOGresponseMax+1e-10);
%     end
 %  A low HOGvariance indicates little confidence in measurement.
 %   Noise is minimized for Kalman process noise.
 max_responseNorm = maxHOGnorm * HOGresponseMax;
 
   if HOGresponseMax < 0.001  % 0.1  was 0.3
     sv2 = sv2old ;  % 0.001;
%      Zk = Xk(1:2,1)' + nullFactor(1:2) ;
   else
       sv2 = sv2old;
   end

% HOGvar = (1 ./ (1+exp(-30*(max_responseNorm-0.25))))*0.3;
% Zk = KalmanEst(1:2,1)' ;
end

% if (frame>60) && (frame<70)
%     sv2 = .09 ;
% end

end


