clc
close all

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

sensor=camvidMonoCameraSensor();

images= imageDatastore('imgseq');
first=0;
dnnseg=0;

cells_per_meter=80;
map_side=4;
map_origin=[1 1];
occgrid=occupancyMap(map_side,map_side,cells_per_meter);        



load('groundTruthPoses.mat');
% Create an empty imageviewset object to manage the data associated with each view.
vSet = imageviewset;

% Read and display the first image.

I=readimage(images,1);

calibrationData = load('camera_params_camvid.mat');

% Describe camera configuration.
focalLength    = calibrationData.cameraParams.FocalLength;
principalPoint = calibrationData.cameraParams.PrincipalPoint;
imageSize      = calibrationData.cameraParams.ImageSize;
intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

prevI = undistortImage(rgb2gray(I), intrinsics); 

% Detect features. 
prevPoints = detectSURFFeatures(prevI, 'MetricThreshold', 500);

% Select a subset of features, uniformly distributed throughout the image.
numPoints = 200;
prevPoints = selectUniform(prevPoints, numPoints, size(prevI));

% Extract features. Using 'Upright' features improves matching quality if 
% the camera motion involves little or no in-plane rotation.
prevFeatures = extractFeatures(prevI, prevPoints, 'Upright', true);

% Add the first view. Place the camera associated with the first view
% at the origin, oriented along the Z-axis.
viewId = 1;


vSet = addView(vSet, viewId, rigid3d(eye(3), [0 0 0]), 'Points', prevPoints);

% Setup axes.
figure
axis([-220, 50, -140, 20, -50, 300]);

% Set Y-axis to be vertical pointing down.
view(gca, 3);
set(gca, 'CameraUpVector', [0, -1, 0]);
camorbit(gca, -120, 0, 'data', [0, 1, 0]);

grid on
xlabel('X (cm)');
ylabel('Y (cm)');
zlabel('Z (cm)');
hold on

% Plot estimated camera pose. 
cameraSize = 7;
camPose = poses(vSet);
camEstimated = plotCamera(camPose, 'Size', cameraSize,...
    'Color', 'g', 'Opacity', 0);


% Initialize camera trajectories.
trajectoryEstimated = plot3(0, 0, 0, 'g-');
trajectoryActual    = plot3(0, 0, 0, 'b-');

legend('Estimated Trajectory', 'Actual Trajectory');
title('Camera Trajectory');

% Read and display the image.
viewId = 2;
I = readimage(images,viewId);

% Convert to gray scale and undistort.
I = undistortImage(rgb2gray(I), intrinsics);

% Match features between the previous and the current image.
[currPoints, currFeatures, indexPairs] = helperDetectAndMatchFeatures(...
    prevFeatures, I);

% Estimate the pose of the current view relative to the previous view.
[orient, loc, inlierIdx] = helperEstimateRelativePose(...
    prevPoints(indexPairs(:,1)), currPoints(indexPairs(:,2)), intrinsics);

% Exclude epipolar outliers.
indexPairs = indexPairs(inlierIdx, :);
    
% Add the current view to the view set.
vSet = addView(vSet, viewId, rigid3d(orient, loc), 'Points', currPoints);

% Store the point matches between the previous and the current views.
vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);


vSet = helperNormalizeViewSet(vSet, groundTruthPoses);

helperUpdateCameraPlots(viewId, camEstimated, poses(vSet));
helperUpdateCameraTrajectories(viewId, trajectoryEstimated, poses(vSet));

prevI = I;
prevFeatures = currFeatures;
prevPoints   = currPoints;


for viewId=3:30
    I=readimage(images,viewId);
    
    if viewId<15    
        % Convert to gray scale and undistort.
        I = undistortImage(rgb2gray(I), intrinsics);

        % Match points between the previous and the current image.
        [currPoints, currFeatures, indexPairs] = helperDetectAndMatchFeatures(...
            prevFeatures, I);

        % Eliminate outliers from feature matches.
        inlierIdx = helperFindEpipolarInliers(prevPoints(indexPairs(:,1)),...
            currPoints(indexPairs(:, 2)), intrinsics);
        indexPairs = indexPairs(inlierIdx, :);

        % Triangulate points from the previous two views, and find the 
        % corresponding points in the current view.
        [worldPoints, imagePoints] = helperFind3Dto2DCorrespondences(vSet,...
            intrinsics, indexPairs, currPoints);

        % Since RANSAC involves a stochastic process, it may sometimes not
        % reach the desired confidence level and exceed maximum number of
        % trials. Disable the warning when that happens since the outcomes are
        % still valid.
        warningstate = warning('off','vision:ransac:maxTrialsReached');

        % Estimate the world camera pose for the current view.
        [orient, loc] = estimateWorldCameraPose(imagePoints, worldPoints, intrinsics);

        % Restore the original warning state
        warning(warningstate)

        % Add the current view to the view set.
        vSet = addView(vSet, viewId, rigid3d(orient, loc), 'Points', currPoints);

        % Store the point matches between the previous and the current views.
        vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);    

        tracks = findTracks(vSet); % Find point tracks spanning multiple views.

        camPoses = poses(vSet);    % Get camera poses for all views.

        % Triangulate initial locations for the 3-D world points.
        xyzPoints = triangulateMultiview(tracks, camPoses, intrinsics);

        % Refine camera poses using bundle adjustment.
        [~, camPoses] = bundleAdjustment(xyzPoints, tracks, camPoses, ...
            intrinsics, 'PointsUndistorted', true, 'AbsoluteTolerance', 1e-12,...
            'RelativeTolerance', 1e-12, 'MaxIterations', 200, 'FixedViewID', 1);

        vSet = updateView(vSet, camPoses); % Update view set.

        % Bundle adjustment can move the entire set of cameras. Normalize the
        % view set to place the first camera at the origin looking along the
        % Z-axes and adjust the scale to match that of the ground truth.
        vSet = helperNormalizeViewSet(vSet, groundTruthPoses);

        % Update camera trajectory plot.

        helperUpdateCameraPlots(viewId, camEstimated, poses(vSet));
        helperUpdateCameraTrajectories(viewId, trajectoryEstimated, poses(vSet));

        prevI = I;
        prevFeatures = currFeatures;
        prevPoints   = currPoints;  
    
    else
        
        % Convert to gray scale and undistort.
        I = undistortImage(rgb2gray(I), intrinsics);

        % Match points between the previous and the current image.
        [currPoints, currFeatures, indexPairs] = helperDetectAndMatchFeatures(...
            prevFeatures, I);    

        % Triangulate points from the previous two views, and find the 
        % corresponding points in the current view.
        [worldPoints, imagePoints] = helperFind3Dto2DCorrespondences(vSet, ...
            intrinsics, indexPairs, currPoints);

        % Since RANSAC involves a stochastic process, it may sometimes not
        % reach the desired confidence level and exceed maximum number of
        % trials. Disable the warning when that happens since the outcomes are
        % still valid.
        warningstate = warning('off','vision:ransac:maxTrialsReached');

        % Estimate the world camera pose for the current view.
        [orient, loc] = estimateWorldCameraPose(imagePoints, worldPoints, intrinsics);

        % Restore the original warning state
        warning(warningstate)

        % Add the current view and connection to the view set.
        vSet = addView(vSet, viewId, rigid3d(orient, loc), 'Points', currPoints);
        vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);

        % Refine estimated camera poses using windowed bundle adjustment. Run 
        % the optimization every 7th view.
        if mod(viewId, 3) == 0        
            % Find point tracks in the last 15 views and triangulate.
            windowSize = 40;
            startFrame = max(1, viewId - windowSize);
            tracks = findTracks(vSet, startFrame:viewId);
            camPoses = poses(vSet, startFrame:viewId);
            [xyzPoints, reprojErrors] = triangulateMultiview(tracks, camPoses, intrinsics);

            % Hold the first two poses fixed, to keep the same scale. 
            fixedIds = [startFrame, startFrame+1];

            % Exclude points and tracks with high reprojection errors.
            idx = reprojErrors < 2;

            [~, camPoses] = bundleAdjustment(xyzPoints(idx, :), tracks(idx), ...
                camPoses, intrinsics, 'FixedViewIDs', fixedIds, ...
                'PointsUndistorted', true, 'AbsoluteTolerance', 1e-12,...
                'RelativeTolerance', 1e-12, 'MaxIterations', 200);

            vSet = updateView(vSet, camPoses); % Update view set.
        end

        % Update camera trajectory plot.

        helperUpdateCameraPlots(viewId, camEstimated, poses(vSet));
        helperUpdateCameraTrajectories(viewId, trajectoryEstimated, poses(vSet));

        prevI = I;
        prevFeatures = currFeatures;
        prevPoints   = currPoints;  

    end

    
    
    I=readimage(images,viewId);
    if dnnseg==1
        % Segment the image.
        [C,scores,allScores] = semanticseg(I,net);

        % Overlay free space onto the image.
        B = labeloverlay(I,C,'IncludedLabels',"Road");

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
        GI=rgb2gray(I);
        mask = false(size(GI)); 
        mask(800/2,600-30) = true;
        W = graydiffweight(GI, mask, 'GrayDifferenceCutoff', 20);
        thresh = 0.001;
        [BW, D] = imsegfmm(W, mask, thresh);
        B=labeloverlay(I,BW);
        figure(2), imshow(B);
        freeSpaceConfidence=BW;
    end
    % Define bird's-eye-view transformation parameters.
    distAheadOfSensor = 2; % in meters, as previously specified in monoCamera height input
    spaceToOneSide    = 1;  % look 3 meters to the right and left
    bottomOffset      = 0;  
    outView = [bottomOffset, distAheadOfSensor, -spaceToOneSide, spaceToOneSide];

    outImageSize = [NaN, 256]; % output image width in pixels; height is chosen automatically to preserve units per pixel ratio

    birdsEyeConfig = birdsEyeView(sensor,outView,outImageSize);
    
    
    % Resize image and free space estimate to size of CamVid sensor. 
    imageSize = sensor.Intrinsics.ImageSize;
    I = imresize(I,imageSize);
    freeSpaceConfidence = imresize(freeSpaceConfidence,imageSize);

    % Transform image and free space confidence scores into bird's-eye view.
    imageBEV = transformImage(birdsEyeConfig,I);
    freeSpaceBEV = transformImage(birdsEyeConfig,freeSpaceConfidence); 

    % Display image frame in bird's-eye view.
    figure(30), imshow(imageBEV);
    
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
    
    del=camPose.AbsolutePose.Translation;
    sensorloc=[(map_origin(1)+spaceToOneSide+del(1)/100) (map_origin(2)+del(3)/100)];
     
    camPose = poses(vSet,viewId);
    R=camPose.AbsolutePose.Rotation; 
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

end





function sensor = camvidMonoCameraSensor()

calibrationData = load('camera_params_camvid.mat');

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