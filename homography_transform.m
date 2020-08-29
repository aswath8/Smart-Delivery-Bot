function y = homography_transform(x, h, dir)
% HOMOGRAPHY_TRANSFORM applies homographic transform to vectors
%   Y = HOMOGRAPHY_TRANSFORM(X, H) takes a 2xN matrix, each column of which
%   gives the position of a point in a plane. It returns a 2xN matrix whose
%   columns are the input vectors transformed according to the homography
%   H, represented as a 3x3 homogeneous matrix.
%   Y = HOMOGRAPHY_TRANSFORM(X, H, DIR) allows the direction of the
%   transformation to be specified. If DIR is true the result is as above.
%   If DIR is false the inverse of the transformation H is used.

if nargin < 3 || dir
    q = h * [x; ones(1, size(x,2))];
else
    q = h \ [x; ones(1, size(x,2))];
end

p = q(3,:);
y = [q(1,:)./p; q(2,:)./p];

end