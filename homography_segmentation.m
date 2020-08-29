close all;
Ixx = imread('k1.png');  % read the image into the matrix
imshow(Ixx);          % display the image

pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
pretrainedNetwork = fullfile(pretrainedFolder,'deeplabv3plusResnet18CamVid.mat'); 
if ~exist(pretrainedNetwork,'file')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetwork,pretrainedURL);
end

data = load(pretrainedNetwork); 
net = data.net;
Ix=imread('k1_large.png');
%Ix=imresize(Ix,[720 960]);
C = semanticseg(Ix, net);
figure;
cmap = camvidColorMap;
imshow(B);

B = labeloverlay(Ix,C,'Colormap',cmap,'Transparency',0.4);

classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];
pixelLabelColorbar(cmap, classes);

%{

load('Homography.mat')

T = projective2d(H);
J = imwarp(Ixx,T);
figure;
imshow(J);

T = projective2d(H);
J = imwarp(B,T);
figure;
imshow(J);

%}