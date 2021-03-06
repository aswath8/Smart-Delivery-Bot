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

sensor=camvidMonoCameraSensor()

images= imageDatastore('imgseq');
I=readimage(images,1);
first=0;
dnnseg=0;


cells_per_meter=80;
map_side=3;
occgrid=occupancyMap(3,3,cells_per_meter);        



for i=1:21
    I=readimage(images,i);
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
        thresh = 0.01;
        [BW, D] = imsegfmm(W, mask, thresh);
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
    %figure(3)
    %imshow(imageBEV)
    
    %figure(4)
    %imagesc(freeSpaceBEV)
    %title('Free Space Confidence')
    
    
    % Define dimensions and resolution of the occupancy grid.
    gridX = distAheadOfSensor;
    gridY = 2 * spaceToOneSide;
    cellSize = 1/cells_per_meter; % in meters to match units used by CamVid sensor

    % Create the occupancy grid from the free space estimate.
    occupancyGrid = createOccupancyGridFromFreeSpaceEstimate(...
        freeSpaceBEV, birdsEyeConfig, gridX, gridY, cellSize);
    
    if first==0
        first=1
        Xinit = size(occupancyGrid(:,1));
        Yinit = size(occupancyGrid(1,:));
        [Xq,Yq] = meshgrid(1:Xinit(1),1:Yinit(2));
        Xq = reshape(Xq, 1, []);
        Yq = reshape(Yq, 1, []);
        Xq = Xq+50;
        Yq = Yq+50;
        probb=reshape(occupancyGrid',1,[]);
        updateOccupancy(occgrid,[(Xq/100)' (Yq/100)'],probb)
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
    
        
        
        
    theta=0; %TO ROTATE CLOCKWISE BY X DEGREES
    %R=[cosd(theta) -sind(theta); sind(theta) cosd(theta)]; %CREATE THE MATRIX
    %rotXY=XY*R'; %MULTIPLY VECTORS BY THE ROT MATRIX 
    %Xqr = reshape(rotXY(:,1), 1, []);
    %Yqr = reshape(rotXY(:,2), 1, []);
    %SHIFTING
    Xq = Xq+0;
    Yq = Yq+5;
    
    prev_probs=getOccupancy(occgrid, [Xq' Yq']);
    probb=reshape(occupancyGrid',1,[]);
    probb=min(prev_probs',probb);
    updateOccupancy(occgrid,[(Xq/100)' (Yq/100)'],probb)
    figure(5)
    show(occgrid)

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