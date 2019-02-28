%% functionname: function description

% input = matchpoints1 nx2 matrix consisting of [x y] coordinate pairs
%		  matchpoints2 nx2 matrix consisting of [x y] coordinate pairs

% output = homography matrix 4x4

function [h] = my_ransacc(matchpoints1,matchpoints2, I1,I2, max_iterations, sigma_threshold, percentage_inliers, Visualize_RANCAC_plot)
	%Ransac parameters
	switch nargin
		case num2cell(1:3)
			sprintf('Enter at least four parameter, i.e.matchpoints1,matchpoints2, I1,I2');	
			% max_iterations = 200;			%number of iteration
			% sigma_threshold = 3; 			%threshold
			% percentage_inliers = 60;		%percentage(converted to ratio) of inliers to fit the model
			% Visualize_RANCAC_plot = false;	%param to visually see RANSAC plot model at every iteration		
		case 4
			max_iterations = 20;			%number of iteration
			sigma_threshold = 3; 			%threshold
			percentage_inliers = 60;		%percentage(converted to ratio) of inliers to fit the model
			Visualize_RANCAC_plot = false;	%param to visually see RANSAC plot model at every iteration
		case 5
			sigma_threshold = 3; 			%threshold
			percentage_inliers = 60;		%percentage(converted to ratio) of inliers to fit the model
			Visualize_RANCAC_plot = false;	%param to visually see RANSAC plot model at every iteration
		case 6
			percentage_inliers = 60;		%percentage(converted to ratio) of inliers to fit the model
			Visualize_RANCAC_plot = false;	%param to visually see RANSAC plot model at every iteration
			
		case 7
			Visualize_RANCAC_plot = false;	%param to visually see RANSAC plot model at every iteration
		case 8
			sprintf('received all 6 params');	
		case num2cell(9:13)
			fprintf(['Kuch Zyada nhi ho gya?']);
		
		otherwise
			fprintf(['No parameters are provied\n', ...
					 'function call is as follows:\n', ...
					 'my_ransac(data, max_iterations, sigma_threshold, percentage_inliers)']);
			return;
	end
	percentage_inliers = percentage_inliers/100;
	% max_iterations = 200;		%number of iteration
	% sigma_threshold = 3; 		%threshold
	% percentage_inliers = 0.8;			%ratio of inliers required to assert that a model fits well to data
	n_samples = length(matchpoints1);		%number of input points
	window_open = false;

	%iterative model
	ratio = 0;
	model_slope = 0;
	model_intercept = 0;

	%perform RANSAC iteration
	for i = 1:max_iterations
		%pick any four random points
		n = 4;
		p = randperm(n_samples,1);
		indice_1_1 = p(1);
		indice_1_2 = p(2);
		indice_1_3 = p(3);
		indice_1_4 = p(4);

		p = randperm(n_samples,1);
		indice_2_1 = p(1);
		indice_2_2 = p(2);
		indice_2_3 = p(3);
		indice_2_4 = p(4]);
		% k = data.pts(:,indice_1)
		% l = data.pts(:,indice_2)
		x_source = [matchpoints1(indice_1_1,1);matchpoints1(indice_1_2,1);matchpoints1(indice_1_3,1);matchpoints1(indice_1_4,1)];
		y_source = [matchpoints1(indice_1_1,2);matchpoints1(indice_1_2,2);matchpoints1(indice_1_3,2);matchpoints1(indice_1_4,2)];

		x_destination = [matchpoints2(indice_2_1,1);matchpoints2(indice_2_2,1);matchpoints2(indice_2_3,1);matchpoints2(indice_2_4,1)];
		y_destination = [matchpoints2(indice_2_1,2);matchpoints2(indice_2_2,2);matchpoints2(indice_2_3,2);matchpoints2(indice_2_4,2)];

		h_model = est_homography(x_destination, y_destination, x_source, y_source);

		% for j = 1:n_samples
			[x_s_tranform, y_s_tranform] = apply_homography(h_model,x_source,y_source);

			
		% end

		maybe_points = [data.pts(:,indice_1) data.pts(:,indice_2)];
		% calling a function to model the line using the randomly selected points
		[slope, c] = model_of_line(maybe_points);
		x_y_inliers = [];
		number_of_inliers = 0;

		%find orthogonal lines to the model for all testing points
		for j = 1:n_samples
			if (j ~=  indice_1 ||  j ~=  indice_2)
				x0 = data.pts(1,j);
				y0 = data.pts(2,j);

				%find an intercept point of the model with a normal from point (x0,y0)
				[x1 y1] = find_intercept_point(slope, c, x0, y0);

				% distance from point to the model
	        	perpendicular_dist = sqrt((x1 - x0)^2 + (y1 - y0)^2);

	        	%check whether it's an inlier or not
	        	if perpendicular_dist < sigma_threshold
	        	    ponits_to_vector = [x0;y0];
	            	x_y_inliers = [x_y_inliers ponits_to_vector];
	            	number_of_inliers = number_of_inliers + 1;
	        	end
	        end
	    end

	    % in case a new model is better - save parameters
	    if number_of_inliers/n_samples > ratio
	    	ratio = number_of_inliers/n_samples;
	    	model_slope = slope;
	    	model_intercept = c;
	    	inliers = number_of_inliers;
		end
		%X = sprintf('%s will be %d this year.',name,age);
		%disp(X)
		sprintf('Inlier ratio = %d',number_of_inliers/n_samples);
		sprintf('model_slope = %d',model_slope);
		sprintf('model_intercept = %d',model_intercept);

		if (Visualize_RANCAC_plot == true)
			if window_open ==true
				close 'Visualize RANSAC happening';
				window_open = false;
			end
			%plot the current step
			figure('Name','Visualize RANSAC happening');
			plot_ransac(model_slope,model_intercept,data);
			pause(0.05);
			window_open = true;
			% close 'Visualize RANSAC happening';

		end
		if number_of_inliers > n_samples*percentage_inliers
			sprintf('The model is found');		
			break;
		end
end

%% SSD: function description
function [] = SSD(X,Y,x,y,I1,I2)
	pad = 20;
	patch_size = 40;
	sigma_ = 2;
	need_to_pad_image_1 = 0;
	need_to_pad_image_2 = 0;
	if( (X>patch_size/2 & X<(size(I1,1)-patch_size/2) ) | ((Y>patch_size/2 & Y<(size(I1,2)-patch_size/2) )
		X_padded = X;
		Y_padded = Y;
	
	else
		X_padded = X+(patch_size/2);
		Y_padded = Y+(patch_size/2);
		need_to_pad_image_1 = 1;
	end
	if( (x>patch_size/2 & x<(size(I1,1)-patch_size/2) ) | ((y>patch_size/2 & y<(size(I1,2)-patch_size/2) )
		x_padded = x;
		y_padded = y;
	else
		x_padded = x+(patch_size/2);
		y_padded = y+(patch_size/2);
		need_to_pad_image_2 = 1;
	end

	if(need_to_pad_image_1 == true | need_to_pad_image_2 == true)
		padded_image_1 = padarray(I1,[pad pad],'both','symmetric');
		padded_image_2 = padarray(I2,[pad pad],'both','symmetric');
	else
		padded_image_1 = I1;
		padded_image_2 = I2;
	end

	% features_1 = zeros((patch_size/4)+1,(patch_size/4)+1);
	% features_2 = zeros((patch_size/4)+1,(patch_size/4)+1);
	% 	x_padded(:,image_) = x1(:,image_)+(patch_size/2);
	% y_padded(:,image_) = y1(:,image_)+(patch_size/2);
	% padded_image = padarray(image_tensor(:,:,image_),[pad pad],'both','symmetric');
	% for i = 1:length(x1(:,image_))
		% fprintf('\nx1-x2 = %d-%d',x_padded(i)-(patch_size/2),x_padded(i)+(patch_size/2));
		patch_1 = padded_image_1(Y_padded-(patch_size/2):Y_padded+(patch_size/2),X_padded-(patch_size/2):X_padded+(patch_size/2));
		patch_2 = padded_image_2(y_padded-(patch_size/2):y_padded+(patch_size/2),x_padded-(patch_size/2):x_padded+(patch_size/2));
		filtered_patch_1 = imgaussfilt(patch_1,sigma_);
		filtered_patch_2 = imgaussfilt(patch_2,sigma_);
		downsize_patch_1 = imresize(filtered_patch_1,0.18,'nearest');
		downsize_patch_2 = imresize(filtered_patch_2,0.18,'nearest');
		features_1 = imresize(downsize_patch_1,[numel(downsize_patch_1),1]);
		features_2 = imresize(downsize_patch_2,[numel(downsize_patch_1),1]);

	mu_1 = mean(features_1);
	sd_1 = std(features_1);

	mu_2 = mean(features_2);
	sd_2 = std(features_2);
	norm_features_1 = (features_1 -mu_1)./sd_1;
	norm_features_2 = (features_1 -mu_2)./sd_2;

end
