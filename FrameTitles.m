function [ ] = FrameTitles( max_response_number, frame, sizeImage )
%Prints the Frame number and HOG Response on the video frames
% 
persistent HOGnum FrameNum 
 printFramePosX = sizeImage(2) - (sizeImage(2)*0.18) ;
 printFramePosY = sizeImage(1) * 0.015 ;
 printHogPosX = sizeImage(2) - (sizeImage(2)*0.25) ;
 printHogPosY = sizeImage(1) * 0.065 ;
 
   if frame==1  
   str1 = ' ' ;
   FrameNum = text(printFramePosX,printFramePosY,' ');
   HOGnum = text(printHogPosX,printHogPosY,' ');
   end
if frame > 2
   delete(HOGnum) ;
   delete(FrameNum) ;
end   
      str = strcat([' Frame ', ' ',num2str(frame)]);
      FrameNum = text(printFramePosX,printFramePosY,str);
      FrameNum.Color='red';
         txt  = sprintf(' %.2f ', max_response_number);    
         str1 = strcat([' HOG Response ', ' ',txt]);
         HOGnum = text(printHogPosX,printHogPosY,str1);
         HOGnum.Color='red';     

end

