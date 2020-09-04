clear all 
close all

sensor=camvidMonoCameraSensor();
withDataset=0;


images= imageDatastore('imgseq');
first=0;
dnnseg=0;

if dnnseg==1
% Download the pretrained network.
pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/segnetVGG16CamVid.mat';
pretrainedFolder = fullfile(tempdir,'pretrainedSegNet');
pretrainedSegNet = fullfile(pretrainedFolder,'segnetVGG16CamVid.mat');
if ~exist(pretrainedFolder,'dir')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained SegNet (107 MB)...');
    websave(pretrainedSegNet,pretrainedURL);
    disp('Download complete.');
end
% Load the network.
data = load(pretrainedSegNet);
net = data.net;
end



cells_per_meter=80;
map_side=4;
map_origin=[1 1];
occgrid=occupancyMap(map_side,map_side,cells_per_meter);        


% Define bird's-eye-view transformation parameters.
distAheadOfSensor = 2; % in meters, as previously specified in monoCamera height input
spaceToOneSide    = 1;  % look 3 meters to the right and left
bottomOffset      = 0;  
outView = [bottomOffset, distAheadOfSensor, -spaceToOneSide, spaceToOneSide];

outImageSize = [NaN, 256]; % output image width in pixels; height is chosen automatically to preserve units per pixel ratio

birdsEyeConfig = birdsEyeView(sensor,outView,outImageSize);




if withDataset==1
    calibrationData = load('camera_params_camvid.mat');
    images= imageDatastore('imgseq');
else
    cam = ipcam('http://192.168.29.34:8080/video');
    calibrationData = load('camera_params_camvid_g5.mat');
end
vSet=imageviewset;

addpath('C:\Users\HP\Documents\MATLAB\Examples\R2020a\vision\MonocularVisualSimultaneousLocalizationAndMappingExample');
% Inspect the first image
currFrameIdx = 1;
vSet = addView(vSet, currFrameIdx, rigid3d(eye(3), [0 0 0]));

if withDataset==1
    currI=readimage(images,currFrameIdx);
else
    currI = snapshot(cam);
end
himage = imshow(currI);

% Set random seed for reproducibility
rng(0);

% Create a cameraIntrinsics object to store the camera intrinsic parameters.
% The intrinsics for the dataset can be found at the following page:
% https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Note that the images in the dataset are already undistorted, hence there
% is no need to specify the distortion coefficients.


% Describe camera configuration.
focalLength    = calibrationData.cameraParams.FocalLength;
principalPoint = calibrationData.cameraParams.PrincipalPoint;
imageSize      = calibrationData.cameraParams.ImageSize;
intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);



% Detect and extract ORB features
[preFeatures, prePoints] = helperDetectAndExtractFeatures(currI); 

currFrameIdx = currFrameIdx + 1;
firstI       = currI; % Preserve the first frame 

isMapInitialized  = false;

% Map initialization loop
while ~isMapInitialized
    %input('enter');
    if withDataset==1
        currI=readimage(images,currFrameIdx);
    else
        currI = snapshot(cam);
    end
    
    [currFeatures, currPoints] = helperDetectAndExtractFeatures(currI); 
    
    currFrameIdx = currFrameIdx + 1;
    
    % Find putative feature matches
    indexPairs = matchFeatures(preFeatures, currFeatures, 'Unique', true, ...
        'MaxRatio', 0.7, 'MatchThreshold', 70);
    
    % If not enough matches are found, check the next frame
    minMatches = 100;
    if size(indexPairs, 1) < minMatches
        continue
    end
    
    preMatchedPoints  = prePoints(indexPairs(:,1),:);
    currMatchedPoints = currPoints(indexPairs(:,2),:);
    
    % Compute homography and evaluate reconstruction
    [tformH, scoreH, inliersIdxH] = helperComputeHomography(preMatchedPoints, currMatchedPoints);

    % Compute fundamental matrix and evaluate reconstruction
    [tformF, scoreF, inliersIdxF] = helperComputeFundamentalMatrix(preMatchedPoints, currMatchedPoints);
    
    % Select the model based on a heuristic
    ratio = scoreH/(scoreH + scoreF);
    ratioThreshold = 0.45;
    if ratio > ratioThreshold
        inlierTformIdx = inliersIdxH;
        tform          = tformH;
    else
        inlierTformIdx = inliersIdxF;
        tform          = tformF;
    end

    % Computes the camera location up to scale. Use half of the 
    % points to reduce computation
    inlierPrePoints  = preMatchedPoints(inlierTformIdx);
    inlierCurrPoints = currMatchedPoints(inlierTformIdx);
    [relOrient, relLoc, validFraction] = relativeCameraPose(tform, intrinsics, ...
        inlierPrePoints(1:2:end), inlierCurrPoints(1:2:end));
    
    % If not enough inliers are found, move to the next frame
    if validFraction < 0.3 || numel(size(relOrient))==3
        continue
    end
    
    % Triangulate two views to obtain 3-D map points
    relPose = rigid3d(relOrient, relLoc);
    [isValid, xyzWorldPoints, inlierTriangulationIdx] = helperTriangulateTwoFrames(...
        rigid3d, relPose, inlierPrePoints, inlierCurrPoints, intrinsics);
    
    if ~isValid
        continue
    end
    
    % Get the original index of features in the two key frames
    indexPairs = indexPairs(inlierTformIdx(inlierTriangulationIdx),:);
    
    isMapInitialized = true;
    
    disp(['Map initialized with frame 1 and frame ', num2str(currFrameIdx-1)])
end % End of map initialization loop

if isMapInitialized
    close(himage.Parent.Parent); % Close the previous figure
    % Show matched features
    hfeature = showMatchedFeatures(firstI, currI, prePoints(indexPairs(:,1)), ...
        currPoints(indexPairs(:, 2)), 'Montage');
else
    error('Unable to initialize map.')
end


% Create an empty imageviewset object to store key frames
vSetKeyFrames = imageviewset;

% Create an empty helperMapPointSet object to store 3D map points
mapPointSet   = helperMapPointSet;

% Add the first key frame. Place the camera associated with the first 
% key frame at the origin, oriented along the Z-axis
preViewId     = 1;
vSetKeyFrames = addView(vSetKeyFrames, preViewId, rigid3d, 'Points', prePoints,...
    'Features', preFeatures.Features);

% Add the second key frame
currViewId    = 2;
vSetKeyFrames = addView(vSetKeyFrames, currViewId, relPose, 'Points', currPoints,...
    'Features', currFeatures.Features);

% Add connection between the first and the second key frame
vSetKeyFrames = addConnection(vSetKeyFrames, preViewId, currViewId, relPose, 'Matches', indexPairs);

% Add 3-D map points
[mapPointSet, newPointIdx] = addMapPoint(mapPointSet, xyzWorldPoints);

% Add observations of the map points
preLocations   = prePoints.Location;
currLocations  = currPoints.Location;
preScales      = prePoints.Scale;
currScales     = currPoints.Scale;

% Add image points corresponding to the map points in the first key frame
mapPointSet   = addObservation(mapPointSet, newPointIdx, preViewId, indexPairs(:,1), ....
    preLocations(indexPairs(:,1),:), preScales(indexPairs(:,1)));

% Add image points corresponding to the map points in the second key frame
mapPointSet   = addObservation(mapPointSet, newPointIdx, currViewId, indexPairs(:,2), ...
    currLocations(indexPairs(:,2),:), currScales(indexPairs(:,2)));




% Run full bundle adjustment on the first two key frames
tracks       = findTracks(vSetKeyFrames);
cameraPoses  = poses(vSetKeyFrames);

[refinedPoints, refinedAbsPoses] = bundleAdjustment(xyzWorldPoints, tracks, ...
    cameraPoses, intrinsics, 'FixedViewIDs', 1, ...
    'PointsUndistorted', true, 'AbsoluteTolerance', 1e-7,...
    'RelativeTolerance', 1e-15, 'MaxIteration', 50);

% Scale the map and the camera pose using the median depth of map points
medianDepth   = median(vecnorm(refinedPoints.'));
refinedPoints = refinedPoints / medianDepth;

refinedAbsPoses.AbsolutePose(currViewId).Translation = ...
    refinedAbsPoses.AbsolutePose(currViewId).Translation / medianDepth;
relPose.Translation = relPose.Translation/medianDepth;


display('Relative Pose is: ');
relPose
realDist=input(strcat('Enter real world distance bw frame1 and frame ',num2str(currFrameIdx-1)));

realScale=norm(norm(realDist)/norm(relPose.Translation));

vSet = addView(vSet, currFrameIdx-1, relPose);
%vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);


% Update key frames with the refined poses
vSetKeyFrames = updateView(vSetKeyFrames, refinedAbsPoses);
vSetKeyFrames = updateConnection(vSetKeyFrames, preViewId, currViewId, relPose);

% Update map points with the refined positions
mapPointSet = updateLocation(mapPointSet, refinedPoints);

% Update view direction and depth 
mapPointSet = updateViewAndRange(mapPointSet, vSetKeyFrames.Views, newPointIdx);

% Visualize matched features in the current frame
%close(hfeature.Parent.Parent);
featurePlot = helperVisualizeMatchedFeatures(currI, currPoints(indexPairs(:,2)));

% Visualize initial map points and camera trajectory
mapPlot     = helperVisualizeMotionAndStructure(vSetKeyFrames, mapPointSet);

% Show legend
showLegend(mapPlot);


% ViewId of the current key frame
currKeyFrameId    = currViewId;

% ViewId of the last key frame
lastKeyFrameId    = currViewId;

% ViewId of the reference key frame that has the most co-visible 
% map points with the current key frame
refKeyFrameId     = currViewId;

% Index of the last key frame in the input image sequence
lastKeyFrameIdx   = currFrameIdx - 1; 

% Indices of all the key frames in the input image sequence
addedFramesIdx    = [1; lastKeyFrameIdx];

isLoopClosed      = false;

readings=[];
snaps=[];

% Main loop
while ~isLoopClosed
    if withDataset==1
        currI=readimage(images,currFrameIdx);
    else
        currI = snapshot(cam);
    end

    [currFeatures, currPoints] = helperDetectAndExtractFeatures(currI);

    % Track the last key frame
    % mapPointsIdx:   Indices of the map points observed in the current frame
    % featureIdx:     Indices of the corresponding feature points in the 
    %                 current frame
    [currPose, mapPointsIdx, featureIdx] = helperTrackLastKeyFrame(mapPointSet, ...
        vSetKeyFrames.Views, currFeatures, currPoints, lastKeyFrameId, intrinsics);
    
    
    
    
    
    % Track the local map
    % refKeyFrameId:      ViewId of the reference key frame that has the most 
    %                     co-visible map points with the current frame
    % localKeyFrameIds:   ViewId of the connected key frames of the current frame
    [refKeyFrameId, localKeyFrameIds, currPose, mapPointsIdx, featureIdx] = ...
        helperTrackLocalMap(mapPointSet, vSetKeyFrames, mapPointsIdx, ...
        featureIdx, currPose, currFeatures, currPoints, intrinsics);
    
    
    realScale=norm(norm(realDist)/norm(relPose.Translation));
    realPose=currPose;
    realPose.Translation=realScale*currPose.Translation;
    realPose.Translation
    readings=[readings; realPose];
    snaps=[snaps; currI];
    norm(realPose.Translation)
    vSet = addView(vSet, currFrameIdx, realPose);
    
    
    % Check if the current frame is a key frame. 
    % A frame is a key frame if both of the following conditions are satisfied:
    %
    % 1. At least 20 frames have passed since the last key frame or the 
    %    current frame tracks fewer than 80 map points
    % 2. The map points tracked by the current frame are fewer than 90% of 
    %    points tracked by the reference key frame
    isKeyFrame = helperIsKeyFrame(mapPointSet, refKeyFrameId, lastKeyFrameIdx, ...
        currFrameIdx, mapPointsIdx);
    
    % Visualize matched features
    updatePlot(featurePlot, currI, currPoints(featureIdx));
    
    if ~isKeyFrame
        currFrameIdx = currFrameIdx + 1;
        continue
    end
    
    % Update current key frame ID
    currKeyFrameId  = currKeyFrameId + 1;
    
    
    % Add the new key frame 
    [mapPointSet, vSetKeyFrames] = helperAddNewKeyFrame(mapPointSet, vSetKeyFrames, ...
        currPose, currFeatures, currPoints, mapPointsIdx, featureIdx, localKeyFrameIds);
    
    % Update view direction and depth
    mapPointSet = updateViewAndRange(mapPointSet, vSetKeyFrames.Views, mapPointsIdx);
    
    % Remove outlier map points that are observed in fewer than 3 key frames
    mapPointSet = helperCullRecentMapPoints(mapPointSet, vSetKeyFrames, newPointIdx);
    
    % Create new map points by triangulation
    [mapPointSet, vSetKeyFrames, newPointIdx] = helperCreateNewMapPoints(mapPointSet, vSetKeyFrames, ...
        currKeyFrameId, intrinsics);

    % Local bundle adjustment
    [mapPointSet, vSetKeyFrames] = helperLocalBundleAdjustment(mapPointSet, vSetKeyFrames, ...
        currKeyFrameId, intrinsics); 
    
    % Visualize 3D world points and camera trajectory
    updatePlot(mapPlot, vSetKeyFrames, mapPointSet);
    
    %helperUpdateCameraPlots(currKeyFrameId, camEstimated, poses(vSet));
    %helperUpdateCameraTrajectories(currKeyFrameId, trajectoryEstimated, poses(vSet));

    % Initialize the loop closure database
    if currKeyFrameId == -1
        % Load the bag of features data created offline
        bofData         = load('bagOfFeaturesData.mat');
        loopDatabase    = invertedImageIndex(bofData.bof);
        loopCandidates  = [1; 2];
        
    % Check loop closure after some key frames have been created    
    elseif currKeyFrameId < -1
        
        % Detect possible loop closure key frame candidates
        [isDetected, validLoopCandidates] = helperCheckLoopClosure(vSetKeyFrames, currKeyFrameId, ...
            loopDatabase, currI, loopCandidates);
        
        if isDetected 
            % Add loop closure connections
            [isLoopClosed, mapPointSet, vSetKeyFrames] = helperAddLoopConnections(...
                mapPointSet, vSetKeyFrames, validLoopCandidates, ...
                currKeyFrameId, currFeatures, currPoints, intrinsics);
        end
    end
    
    % If no loop closure is detected, add the image into the database
    if isLoopClosed
        flname=strcat(string(currFrameIdx),'.png');
        imwrite(currI,flname);
        currds = imageDatastore(flname);
        addImages(loopDatabase, currds, 'Verbose', false);
        loopCandidates= [loopCandidates; currKeyFrameId]; %#ok<AGROW>
    end
    
    % Update IDs and indices
    lastKeyFrameId  = currKeyFrameId;
    lastKeyFrameIdx = currFrameIdx;
    addedFramesIdx  = [addedFramesIdx; currFrameIdx]; %#ok<AGROW>
    currFrameIdx  = currFrameIdx + 1;
    
    
    
    
    if dnnseg==1
        % Segment the image.
        [C,scores,allScores] = semanticseg(currI,net);

        % Overlay free space onto the image.
        B = labeloverlay(currI,C,'IncludedLabels',"Road");

        % Display free space and image.
        %figure(1)
        %imshow(B)

        % Use the network's output score for Road as the free space confidence.
        roadClassIdx = 4;
        freeSpaceConfidence = allScores(:,:,roadClassIdx);

        % Display the free space confidence.
        %figure(2)
        %imagesc(freeSpaceConfidence)
        %title('Free Space Confidence Scores')
        %colorbar
    else
        GI=rgb2gray(currI);
        mask = false(size(GI)); 
        mask(480-20,740/2) = true;
        W = graydiffweight(GI, mask, 'GrayDifferenceCutoff', 5);
        thresh = 0.0001;
        [BW, D] = imsegfmm(W, mask, thresh);
        B=labeloverlay(currI,BW);
        figure(20), imshow(imresize(B,0.35));
        freeSpaceConfidence=BW;
    end
    
    
    
    
    % Resize image and free space estimate to size of CamVid sensor. 
    imageSize = sensor.Intrinsics.ImageSize;
    I = imresize(currI,imageSize);
    freeSpaceConfidence = imresize(freeSpaceConfidence,imageSize);

    % Transform image and free space confidence scores into bird's-eye view.
    imageBEV = transformImage(birdsEyeConfig,I);
    freeSpaceBEV = transformImage(birdsEyeConfig,freeSpaceConfidence); 

    % Display image frame in bird's-eye view.
    %figure(30), imshow(imageBEV);
    
    %figure(4), imagesc(freeSpaceBEV);
    %title('Free Space Confidence');
    
    
    % Define dimensions and resolution of the occupancy grid.
    gridX = distAheadOfSensor;
    gridY = 2 * spaceToOneSide;
    cellSize = 1/cells_per_meter; % in meters to match units used by CamVid sensor

    % Create the occupancy grid from the free space estimate.
    occupancyGrid = createOccupancyGridFromFreeSpaceEstimate(...
        freeSpaceBEV, birdsEyeConfig, gridX, gridY, cellSize);
    
    if first==0
        first=1;
        Xinit = size(occupancyGrid(:,1));
        Yinit = size(occupancyGrid(1,:));
        [Xq,Yq] = meshgrid(1:Xinit(1),1:Yinit(2));
        Xq = reshape(Xq, 1, []);
        Yq = reshape(Yq, 1, []);
        probb=reshape(occupancyGrid',1,[]);
        Xqm=Xq/cells_per_meter;
        Yqm=Yq/cells_per_meter;
        updateOccupancy(occgrid,[(map_origin(1)+Xqm)' (map_origin(2)+Yqm)'],probb)
        figure(5)
        show(occgrid)
    end
    %{
    % Create bird's-eye plot.
    bep = birdsEyePlot('XLimits',[0 distAheadOfSensor],'YLimits', [-5 5]);

    % Add occupancy grid to bird's-eye plot.
    hold on
    [numCellsY,numCellsX] = size(occupancyGrid);
    X = linspace(0, gridX, numCellsX);
    Y = linspace(-gridY/2, gridY/2, numCellsY);
    h = pcolor(X,Y,occupancyGrid);
    title('Occupancy Grid (probability)')
    colorbar
    delete(legend)

    % Make the occupancy grid visualization transparent and remove grid lines.
    h.FaceAlpha = 0.5;
    h.LineStyle = 'none';
    
    % Add coverage area to plot.
    caPlotter = coverageAreaPlotter(bep, 'DisplayName', 'Coverage Area');

    % Update it with a field of view of 35 degrees and a range of 60 meters
    mountPosition = [0 0];
    range = 15;
    orientation = 0;
    fieldOfView = 35;
    plotCoverageArea(caPlotter, mountPosition, range, orientation, fieldOfView);
    hold off
    %}
    
    del=realPose.Translation;
    sensorloc=[(map_origin(1)+spaceToOneSide+del(1)/100) (map_origin(2)+del(3)/100)];
     
    R=realPose.Rotation; 
    eulZYX = rotm2eul(R);
    theta=rad2deg(eulZYX(2)); %TO ROTATE CLOCKWISE BY X DEGREES
    R=[cosd(theta) -sind(theta); sind(theta) cosd(theta)]; %CREATE THE MATRIX
    RX=Xqm-spaceToOneSide; %Rotate map about sensor location
    RY=Yqm-0;
    rotXY=[RX(:) RY(:)]*R'; %MULTIPLY VECTORS BY THE ROT MATRIX 
    Xq = reshape(rotXY(:,1), 1, []);
    Yq = reshape(rotXY(:,2), 1, []);
    
    %SHIFTING
    
    Xqt = map_origin(1)+Xq+spaceToOneSide+(del(1)/100);
    Yqt = map_origin(2)+Yq+0+(del(3)/100);
    
    prev_probs=getOccupancy(occgrid, [Xqt' Yqt']);
    probb=reshape(occupancyGrid',1,[]);
    probb=min(prev_probs',probb);
    updateOccupancy(occgrid,[Xqt' Yqt'],probb)
    figure(5);
    show(occgrid);
    hold on 
    plot(sensorloc(1), sensorloc(2),'ro')
    hold off
    
    
    
    
end % End of main loop


function [features, validPoints] = helperDetectAndExtractFeatures(Irgb, varargin)

scaleFactor = 1.01;
numLevels   = 3;
numPoints   = 1000;

% In this example, the images are already undistorted. In a general
% workflow, uncomment the following code to undistort the images.
%
% if nargin > 1
%     intrinsics = varargin{1};
% end
% Irgb  = undistortImage(Irgb, intrinsics);

% Detect ORB features
Igray  = rgb2gray(Irgb);

points = detectORBFeatures(Igray, 'ScaleFactor', scaleFactor, 'NumLevels', numLevels);

% Select a subset of features, uniformly distributed throughout the image
points = selectUniform(points, numPoints, size(Igray, 1:2));

% Extract features
[features, validPoints] = extractFeatures(Igray, points);
end


function [H, score, inliersIndex] = helperComputeHomography(matchedPoints1, matchedPoints2)

[H, inlierPoints1, inlierPoints2] = estimateGeometricTransform( ...
    matchedPoints1, matchedPoints2, 'projective', ...
    'MaxNumTrials', 1e3, 'MaxDistance', 4, 'Confidence', 90);

[~, inliersIndex] = intersect(matchedPoints1.Location, ...
    inlierPoints1.Location, 'row', 'stable');

locations1 = inlierPoints1.Location;
locations2 = inlierPoints2.Location;
xy1In2     = transformPointsForward(H, locations1);
xy2In1     = transformPointsInverse(H, locations2);
error1in2  = sum((locations2 - xy1In2).^2, 2);
error2in1  = sum((locations1 - xy2In1).^2, 2);

outlierThreshold = 6;

score = sum(max(outlierThreshold-error1in2, 0)) + ...
    sum(max(outlierThreshold-error2in1, 0));
end


function [F, score, inliersIndex] = helperComputeFundamentalMatrix(matchedPoints1, matchedPoints2)

[F, inliersLogicalIndex]   = estimateFundamentalMatrix( ...
    matchedPoints1, matchedPoints2, 'Method','RANSAC',...
    'NumTrials', 1e3, 'DistanceThreshold', 0.01);

inlierPoints1 = matchedPoints1(inliersLogicalIndex);
inlierPoints2 = matchedPoints2(inliersLogicalIndex);

inliersIndex  = find(inliersLogicalIndex);

locations1    = inlierPoints1.Location;
locations2    = inlierPoints2.Location;

% Distance from points to epipolar line
lineIn1   = epipolarLine(F', locations2);
error2in1 = (sum([locations1, ones(size(locations1, 1),1)].* lineIn1, 2)).^2 ...
    ./ sum(lineIn1(:,1:2).^2, 2);
lineIn2   = epipolarLine(F, locations1);
error1in2 = (sum([locations2, ones(size(locations2, 1),1)].* lineIn2, 2)).^2 ...
    ./ sum(lineIn2(:,1:2).^2, 2);

outlierThreshold = 4;

score = sum(max(outlierThreshold-error1in2, 0)) + ...
    sum(max(outlierThreshold-error2in1, 0));

end


function [isValid, xyzPoints, inlierIdx] = helperTriangulateTwoFrames(...
    pose1, pose2, matchedPoints1, matchedPoints2, intrinsics)

[R1, t1]   = cameraPoseToExtrinsics(pose1.Rotation, pose1.Translation);
camMatrix1 = cameraMatrix(intrinsics, R1, t1);

[R2, t2]   = cameraPoseToExtrinsics(pose2.Rotation, pose2.Translation);
camMatrix2 = cameraMatrix(intrinsics, R2, t2);

[xyzPoints, reprojectionErrors] = triangulate(matchedPoints1, ...
    matchedPoints2, camMatrix1, camMatrix2);

% Filter points by view direction and reprojection error
minReprojError = 1;
inlierIdx  = xyzPoints(:,3) > 0 & reprojectionErrors < minReprojError;
xyzPoints  = xyzPoints(inlierIdx ,:);

% A good two-view with significant parallax
ray1       = xyzPoints - pose1.Translation;
ray2       = xyzPoints - pose2.Translation;
cosAngle   = sum(ray1 .* ray2, 2) ./ (vecnorm(ray1, 2, 2) .* vecnorm(ray2, 2, 2));

% Check parallax
minParallax = 3; % in degrees
isValid = all(cosAngle < cosd(minParallax) & cosAngle>0);
end


function isKeyFrame = helperIsKeyFrame(mapPoints, ...
    refKeyFrameId, lastKeyFrameIndex, currFrameIndex, mapPointsIndices)

numPointsRefKeyFrame = numel(getMapPointIndex(mapPoints, refKeyFrameId));

% More than 20 frames have passed from last key frame insertion
tooManyNonKeyFrames = currFrameIndex >= lastKeyFrameIndex + 20;

% Track less than 80 map points
tooFewMapPoints     = numel(mapPointsIndices) < 80;

% Tracked map points are fewer than 90% of points tracked by
% the reference key frame
tooFewTrackedPoints = numel(mapPointsIndices) < 0.9 * numPointsRefKeyFrame;

isKeyFrame = (tooManyNonKeyFrames || tooFewMapPoints) && tooFewTrackedPoints;
end


function mapPoints = helperCullRecentMapPoints(mapPoints, keyFrames, newPointIdx)

for i = 1: numel(newPointIdx)
    idx =  newPointIdx(i);
    % If a map point is observed in less than 3 key frames, drop it
    if numel(mapPoints.Observations{idx, 1})< 3 &&...
            max(mapPoints.Observations{idx, 1}) < keyFrames.Views.ViewId(end)
        mapPoints = updateValidity(mapPoints, idx, false);
    end
end
end

function rmse = helperEstimateTrajectoryError(gTruth, cameraPoses)
locations       = vertcat(cameraPoses.AbsolutePose.Translation);
gLocations      = vertcat(gTruth.Translation);
scale           = median(vecnorm(gLocations, 2, 2))/ median(vecnorm(locations, 2, 2));
scaledLocations = locations * scale;

rmse = sqrt(mean( sum((scaledLocations - gLocations).^2, 2) ));
disp(['Absolute RMSE for key frame trajectory (m): ', num2str(rmse)]);
end




function sensor = camvidMonoCameraSensor()

calibrationData = load('camera_params_camvid_g5.mat');

% Describe camera configuration.
focalLength    = calibrationData.cameraParams.FocalLength;
principalPoint = calibrationData.cameraParams.PrincipalPoint;
imageSize      = calibrationData.cameraParams.ImageSize;

% Camera height estimated based on camera setup pictured in [1]:
% http://mi.eng.cam.ac.uk/~gjb47/tmp/prl08.pdf
height = 0.07;  % height in meters from the ground

% Camera pitch was computed using camera extrinsics provided in data set.
pitch = 0;  % pitch of the camera, towards the ground, in degrees

camIntrinsics = cameraIntrinsics(focalLength,principalPoint,imageSize);
sensor = monoCamera(camIntrinsics,height,'Pitch',pitch);
end


function occupancyGrid = createOccupancyGridFromFreeSpaceEstimate(...
    freeSpaceBEV,birdsEyeConfig,gridX,gridY,cellSize)
% Return an occupancy grid that contains the occupancy probability over
% a uniform 2-D grid.

% Number of cells in occupancy grid.
numCellsX = ceil(gridX / cellSize);
numCellsY = ceil(gridY / cellSize);

% Generate a set of (X,Y) points for each grid cell. These points are in
% the vehicle's coordinate system. Start by defining the edges of each grid
% cell.

% Define the edges of each grid cell in vehicle coordinates.
XEdges = linspace(0,gridX,numCellsX);
YEdges = linspace(-gridY/2,gridY/2,numCellsY);

% Next, specify the number of sample points to generate along each
% dimension within a grid cell. Use these to compute the step size in the
% X and Y direction. The step size will be used to shift the edge values of
% each grid to produce points that cover the entire area of a grid cell at
% the desired resolution.

% Sample 20 points from each grid cell. Sampling more points may produce
% smoother estimates at the cost of additional computation.
numSamplePoints = 20;

% Step size needed to sample number of desired points.
XStep = (XEdges(2)-XEdges(1)) / (numSamplePoints-1);
YStep = (YEdges(2)-YEdges(1)) / (numSamplePoints-1);

% Finally, slide the set of points across both dimensions of the grid
% cells. Sample the occupancy probability along the way using
% griddedInterpolant.

% Create griddedInterpolant for sampling occupancy probability. Use 1
% minus the free space confidence to represent the probability of occupancy.
occupancyProb = 1 - freeSpaceBEV;
sz = size(occupancyProb);
[y,x] = ndgrid(1:sz(1),1:sz(2));
F = griddedInterpolant(y,x,occupancyProb);

% Initialize the occupancy grid to zero.
occupancyGrid = zeros(numCellsY*numCellsX,1);

% Slide the set of points XEdges and YEdges across both dimensions of the
% grid cell. 
for j = 1:numSamplePoints
    
    % Increment sample points in the X-direction
    X = XEdges + (j-1)*XStep;
   
    for i = 1:numSamplePoints
        
        % Increment sample points in the Y-direction
        Y = YEdges + (i-1)*YStep;
        
        % Generate a grid of sample points in bird's-eye-view vehicle coordinates
        [XGrid,YGrid] = meshgrid(X,Y);
        
        % Transform grid of sample points to image coordinates
        xy = vehicleToImage(birdsEyeConfig,[XGrid(:) YGrid(:)]);
        
        % Clip sample points to lie within image boundaries
        xy = max(xy,1);
        xq = min(xy(:,1),sz(2));        
        yq = min(xy(:,2),sz(1));
        
        % Sample occupancy probabilities using griddedInterpolant and keep
        % a running sum.
        occupancyGrid = occupancyGrid + F(yq,xq);  
    end
    
end

% Determine mean occupancy probability.
occupancyGrid = occupancyGrid / numSamplePoints^2;
occupancyGrid = reshape(occupancyGrid,numCellsY,numCellsX);
end