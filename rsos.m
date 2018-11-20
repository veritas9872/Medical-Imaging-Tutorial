function image = rsos(data)
% RSOS Root Sum of Squares Function.
% 
% The root Sum of Squares function necessary for converting 3D multichannel 
% complex data into 2D real valued data.
assert(length(size(data)) == 2 || length(size(data)) == 3, 'Data must be either 2D or 3D.');
image = abs(data) .^ 2;
image = sum(image, 3);
image = sqrt(image);
assert(length(size(image)) == 2);
end