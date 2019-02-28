%% About

% CMSC-426 : Computer Vision 
% Project - 2 : Panorama Image Stitching 
% Author : Kartik Madhira, Prateek Arora, Gireesh Suresh 
% email  : kartikmadhira1@gmail.com , Prateekarorav2@gmail.com, gireesh@umd.edu
% School : University of Maryland, College Park


%% References:

% [1] https://cmsc426.github.io/2018/proj/p2/
% [2] https://www.mathworks.com/help/vision/examples/feature-based-panoramic-image-stitching.html

%% Clean Slate

close all; warning off;
clear all; clc;

%% Switch to the current directory of mfile.

if(~isdeployed)
  cd(fileparts(which(mfilename)));
end

%% Load the Data Files

directory   = '/home/abhinav/CMSC-733/YourDirectoryID_p1/Phase1/Data/Train/Set3/';

sourceFiles = dir(fullfile(directory,'*.jpg'));
%sourceFiles = natsortfiles({sourceFiles.name});
panScene=imageDatastore(directory);
%montage(panScene.Files);
fileCount   = length(sourceFiles) ;  % To calculate the total number of files in the directory.
disp('Reading the Image files from Directory...');


tforms(fileCount) = projective2d(eye(3));
imageSize = zeros(fileCount,2);
I = readimage(panScene, 1);
gray1=rgb2gray(I);

[iBest1,jBest1,c1] = anms(I,300);
[features1,g1] = featDesc(iBest1,jBest1,gray1);

for inputFile = 2:(fileCount)
    fprintf('Input_image - %s \n',int2str(inputFile));
    %filename = char(sourceFiles(inputFile));
    I=readimage(panScene,inputFile);
   
% Convert the Input Image to Grayscale
    gray2 = rgb2gray(I);
    imageSize(inputFile,:) = size(gray2);

    %% 1) ANMS
    previBest=iBest1;
    prevjBest=jBest1;
    %prevC=[iBest1,jBest1,c1];
    prevFeatures=features1;
    [iBest2,jBest2,c2] = anms(I,300);

    %% 2) Feature Descriptor 

    [features2,g2] = featDesc(iBest2,jBest2,gray2);

    %% 3) Feature Matching

    [matchedPoints1,matchedPoints2] = getMatches(prevFeatures,features2,previBest,iBest2,prevjBest,jBest2);
   
   % im=showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage');

   % H = est_homography(matchedPoints2(1:5,1),matchedPoints2(1:5,2),matchedPoints1(1:5,1),matchedPoints1(1:5,2));
    tforms(inputFile)=estimateGeometricTransform(matchedPoints2,matchedPoints1, 'projective','Confidence', 99.9, 'MaxNumTrials', 2000);
    tforms(inputFile).T = tforms(inputFile).T * tforms(inputFile-1).T;
    iBest1=iBest2;
    jBest1=jBest2;
    features1=features2;
end

%% Panaroma stiching

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = tforms(i).T * Tinv.T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama1 = zeros([height width 3], 'like', I);



blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:fileCount

    I = readimage(panScene, i);

    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    % Generate a binary mask.
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    % Overlay the warpedImage onto the panorama.
    panorama1 = step(blender, panorama1, warpedImage, mask);
end
figure
imshow(panorama1)


%% ANMS 

function [xBest,yBest,RGB] = anms(image,nBestPoints)

%ANMS algorithm

%convert the thing to grayscale image
gray=rgb2gray(image);
cornerScore=cornermetric(gray);

%get the local maximas from the images.
locMaximas=imregionalmax(cornerScore);

%get the binary values 
cornerLoc=find(locMaximas==true);

%get the count of strong corners
nStrong=length(cornerLoc);

%get the locations of pixels of strong corners
[x,y]=find(locMaximas==true);
Rpoints=inf([nStrong,1]);

xLoc=zeros([nStrong,1]);
yLoc=zeros([nStrong,1]);

Ed=0;
%x=j and y=i
%get every pixel location and compare metric w.r.t to all other metrics.
for i=1:nStrong;
    for j=1:nStrong;
        if(cornerScore(x(j),y(j))>cornerScore(x(i),y(i)));
            Ed=((x(j)-x(i))^2)+((y(j)-y(i))^2);
        end
        if(Ed<Rpoints(i));
            Rpoints(i)=Ed;
            xLoc(i)=x(i);
            yLoc(i)=y(i);
        end
    end
end

%sort the rpoints and get the x,y values
[sort1,ind]=sort(Rpoints,'descend');
nBest=sort1(1:nBestPoints);
%plot the x and y points

[rr gg bb] = deal(gray);
xBest=zeros([length(nBest),1]);
yBest=zeros([length(nBest),1]);
for i=1:length(nBest)
    xBest(i)=xLoc(ind(i));
    yBest(i)=yLoc(ind(i));
    rr(xLoc(ind(i)),yLoc(ind(i))) = 255;
    gg(xLoc(ind(i)),yLoc(ind(i))) = 0;
    bb(xLoc(ind(i)),yLoc(ind(i))) = 0; 
end
RGB = cat(3,rr,gg,bb);
end



%% Feature Descriptor

function [descVector,patch] = featDesc(iBest,jBest,gray)
descVector=zeros([1,64]);
descVector(:)=[];
for i=1:length(iBest)
    % Initially do a Zero-Padding to the image.
    grayPadded=padarray(gray,[40 40],'both');
    
    % Choose a keypoint and get its corresponding Patch
    patch=grayPadded(iBest(i)+20:iBest(i)+60,jBest(i)+20:jBest(i)+60);
    
    
    % Apply Gaussian Blur to this patch
    blurPatch=imgaussfilt(patch,2);

    % Subsampling the Image patch (41x41 here) to 8*8
    sampPatch=imresize(blurPatch,0.25,'nearest');
    size(sampPatch);
    
    % Reshape the Patch Size to a Vector
    sampPatch=reshape(sampPatch,[1,121]);
    sampPatch=cast(sampPatch,'single');
    
    
    % Zero Mean
    sampPatch = sampPatch - mean(sampPatch(:));

    % Unit Variance
    sampPatch = sampPatch / std(sampPatch(:));
    mean(sampPatch);
    var(sampPatch);
    descVector=vertcat(descVector,sampPatch);
    
end
%normalize the dataset

end

function ssdOut = ssd(vecA,vecB)
    ssdOut=sum((vecA(:)-vecB(:)).^2);
end

%% Feature match

function [matchedPoints1,matchedPoints2] = getMatches(features1,features2,iBest1,iBest2,jBest1,jBest2)
matchedPoints1=zeros(1,2);
matchedPoints1(:)=[];
matchedPoints2=zeros(1,2);
matchedPoints2(:)=[];
for i=1:length(iBest1)
    S=0;
    I=0;
    for j=1:length(iBest1)
        ssdO(j)=ssd(features1(i,:),features2(j,:));
        size(features2(j));
        %sort the ssd and get the lowest two values
    end
    [S,I]=sort(ssdO,'ascend');
    firstMatch=S(1);
    secondMatch=S(2);
    if(((firstMatch/secondMatch)<0.5))
        matchedPoints1=vertcat(matchedPoints1,[jBest1(i) iBest1(i)]);
        matchedPoints2=vertcat(matchedPoints2,[jBest2(I(1)) iBest2(I(1))]);
        end
    end
end

function H = est_homography(X,Y,x,y)
% H = est_homography(X,Y,x,y)
% Compute the homography matrix from source(x,y) to destination(X,Y)
%
%    X,Y are coordinates of destination points
%    x,y are coordinates of source points
%    X/Y/x/y , each is a vector of n*1, n>= 4
%
%    H is the homography output 3x3
%   (X,Y, 1)^T ~ H (x, y, 1)^T

A = zeros(length(x(:))*2,9);

for i = 1:length(x(:)),
 a = [x(i),y(i),1];
 b = [0 0 0];
 c = [X(i);Y(i)];
 d = -c*a;
 A((i-1)*2+1:(i-1)*2+2,1:9) = [[a b;b a] d];
end

[U S V] = svd(A);
h = V(:,9);
H = reshape(h,3,3)';
end

function [X, Y] = apply_homography(H, x, y)
% [X, Y] = apply_homography(H, x, y)
% Use homogrphay matrix H to compute position (x,y) in the source image to
% the position (X,Y) in the destination image
%
% Input
%   H : 3*3 homography matrix, refer to setup_homography
%   x : the column coords vector, n*1, in the source image
%   y : the column coords vector, n*1, in the source image
% Output
%   X : the column coords vector, n*1, in the destination image
%   Y : the column coords vector, n*1, in the destination image

p1 = [x'; y'; ones(1, length(x))];
q1 = H*p1;
q1 = q1./[q1(3, :); q1(3,:); q1(3, :)];

X = q1(1,:)';
Y = q1(2, :)';
end