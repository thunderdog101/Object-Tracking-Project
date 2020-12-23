function [] = boundingBoxes(Zk, frame, GT_Box, box, ...
    padding, target_sz, Xk, nullFactor)
% Code listed hear to create bounding boxes (for displayed video) for
%  Ground Truth, HoG without Kalman and Kalman Estimate
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent gt_rec gt_title HoG_title Kalman_rec Kalman_title SearchTitle sb
    if frame ~= 1
       delete(gt_rec);
       delete(gt_title);
       delete(HoG_title);
       delete(Kalman_rec);
       delete(Kalman_title);
       delete(SearchTitle);
       delete(sb);
    end
    %% GROUND TRUTH Bounding Box and Title
        %  Matlab rectangle:  [x y width height]  x,y: lower-left
   gt_rec=rectangle('Position',GT_Box,'EdgeColor','r', ...
       'LineWidth',1);
   gt_title = text(GT_Box(1)+20,GT_Box(2)-6,'GT');
   gt_title.Color='r';
%% HoG Bounding Box and Title
   HoG_title = text(box(1)-20,box(2)-6,'HoG');
   HoG_title.Color='g' ;
%% PATCH Bounding Box and Title  (Search window)
   SearchBox =  [Zk(2) - (padding+1)*target_sz(2)/2  ...
       Zk(1) - (padding+1)*target_sz(1)/2 (padding+1)*target_sz([2,1])];
   sb=rectangle('Position',SearchBox, 'EdgeColor', 'magenta','LineWidth',1);
   sbTitle = ['Search Window ' num2str(padding+1) 'x'];
   SearchTitle = text(SearchBox(1)-10,SearchBox(2)-6,sbTitle);
   SearchTitle.Color='magenta';                  % aka purple
%% KALMAN Bounding Box and Title
   Kalman_Box=[Xk(2)-target_sz(2)/2 ...
       Xk(1)-target_sz(1)/2 target_sz([2,1])];
   Kalman_rec=rectangle('Position',Kalman_Box, ...
       'EdgeColor','yellow','LineWidth',1);
   Kalman_title = text(Kalman_Box(1)-10,...
       Kalman_Box(2)+Kalman_Box(4)+11,'Kalman');
   Kalman_title.Color='yellow';
%%
%  im = insertShape(im,'FilledRectangle',GT_Box,'Color','w');           
% 	if stop, break, end  %user pressed Esc, stop early
	drawnow
% 			pause(0.05)  %uncomment to run slower
end

