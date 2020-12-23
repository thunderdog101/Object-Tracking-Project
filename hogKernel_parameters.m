function [features,kernel,kernel_type,feature_type,interp_factor, ...
    padding,lambda,output_sigma_factor,cell_size] = hogKernel_parameters
% Sets parameters for HOG, kernel features, etc
%  
   kernel_type = 'gaussian';  %#ok<NASGU>
   feature_type = 'hog'; 
   features.gray = false;
   padding =  1.5;                     % extra area surrounding the target
   lambda = 1e-4;                      % regularization
   output_sigma_factor = 0.1;  % spatial bandwidth (proportional to target)
	
%  hog  is used as the feature_type
        kernel.type = 'gaussian' ;
		interp_factor = 0.02;		
		kernel.sigma = 0.5;		
		kernel.poly_a = 1;
		kernel.poly_b = 9;		
		features.hog = true;
		features.hog_orientations = 9;
		cell_size = 4;
        kernel_type = kernel.type ;
end

