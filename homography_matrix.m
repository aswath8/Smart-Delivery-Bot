function v = homography_matrix(u, theta, t, f, camMoves)
% HOMOGRAPHY_MATRIX returns homography matrix for view of points in a plane
%   V = HOMOGRAPHY_MATRIX(U, THETA, T, F) returns a 3*3 matrix that
%   transforms homogenous coordinates of points in a plane into homogeneous
%   coordinates of points in an image of it.
%
%   By default, the plane starts at Z=0 in conventional camera coordinates,
%   with X and Y plane and camera coordinates coinciding. The plane is
%   rotated by an angle THETA (positive clockwise) about an axis in the
%   direction U. The plane is then translated by a vector T given in camera
%   coordinates. F is the focal length of the camera.
%
%   V = HOMOGRAPHY_MATRIX(..., CAMMOVES) with CAMMOVES true specifies the
%   rotation and translation of the camera in the plane's frame of
%   reference.
%
%   V = HOMOGRAPHY_MATRIX([], R, T, F, CAMMOVES) uses the rotation matrix
%   R. CAMMOVES may be omitted, in which case it is taken as false.
%
%   Note that if the plane is the ground, and the camera is looking down on
%   it, then in picturing the initial alignment the Z axis will be pointing
%   down into the ground. If the camera motion is specified, the third
%   component of T is likely to be negative in this case. If U is [1 0 0]
%   and 0 < THETA < pi/2 the "normal" view of a plane is produced, with the
%   plane's Y and the image's y becoming more negative towards the top of
%   the image.

if nargin < 5; camMoves = false; end

if isempty(u)   % r is given
    r = theta;
else
    r = rotation_matrix(u, theta);
end

if camMoves
    r = r.';
    t = r * (-t);
end

v = [f*r(1:2, 1:2) f*t(1:2); r(3, 1:2) t(3)];

end