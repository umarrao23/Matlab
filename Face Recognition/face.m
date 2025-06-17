clc; clear; close all;

%% Images are loaded
imgSingle = imread('single.jpg'); %load the single person image using imread
imgGroup = imread('group.jpg');   %load the group image 

%conversion to grey scale both images
graySingle = rgb2gray(imgSingle);     %single image converted to grey scale
grayGroup = rgb2gray(imgGroup);       %double image converted to grey scale

%Load haar cascade classifier
faceDetector = vision.CascadeObjectDetector('haarcascade_frontalface_default.xml');%haar Cascade loaded as face detector

%detect faces using haar cascade in both single and group
allBoxesSingle = step(faceDetector, graySingle);    %
bboxGroup = step(faceDetector, grayGroup);          %

%select the largest face from single  (background face if present removed)
[~, idx] = max(allBoxesSingle(:,3) .* allBoxesSingle(:,4)); % area = width × height
bboxSingle = allBoxesSingle(idx, :);

%extract LBP from single
singleFace = imresize(imcrop(graySingle, bboxSingle), [100 100]);
featuresSingle = extractLBP(singleFace);

% loop through eachh face in group extract LBP of each face
minDist = inf;
matchIndex = -1;

for i = 1:size(bboxGroup,1)
    groupFace = imresize(imcrop(grayGroup, bboxGroup(i,:)), [100 100]); %resize
    featuresGroup = extractLBP(groupFace); %extract LBP features

    dist = norm(featuresSingle - featuresGroup); %calculates eluciden lbp distance between single and group
    if dist < minDist   %keeps track of smallets distance
        minDist = dist;
        matchIndex = i;
    end
end

%Display Group Image with Match Box 
figure;
imshow(imgGroup);
hold on;

%match decision
threshold = 0.5;  %trhreshold for eluciden distance to match

if minDist < threshold %smallets distance below 0.5 it is a match 
    rectangle('Position', bboxGroup(matchIndex,:), 'EdgeColor', 'red', 'LineWidth', 5); %draw box if match
    title(['Match Found: Face #' num2str(matchIndex) ', Distance=' num2str(minDist)]);

end

hold off;



% LBP Feature Extraction Function
function lbpVec = extractLBP(img)  % Input grayscale image/ Output normalized 256-bin LBP histogram
    img = double(img);             % Convert image to double for numeric comparisons
    [r, c] = size(img);            % Get image dimensions (rows and columns)
    lbp = zeros(r-2, c-2);         % Initialize LBP matrix (excluding borders)

    % Loop through every pixel excluding border pixels
    for i = 2:r-1
        for j = 2:c-1
            center = img(i,j);     % The center pixel around which LBP is computed
            code = 0;              % Initialize LBP code (8-bit binary number)

            % Compare 8 surrounding pixels with center pixel
            % If neighbor >= center, set corresponding bit to 1 (weighted by 2^position)
            code = code + (img(i-1,j-1) >= center) * 2^7;  % Top-left
            code = code + (img(i-1,j  ) >= center) * 2^6;  % Top
            code = code + (img(i-1,j+1) >= center) * 2^5;  % Top-right
            code = code + (img(i  ,j+1) >= center) * 2^4;  % Right
            code = code + (img(i+1,j+1) >= center) * 2^3;  % Bottom-right
            code = code + (img(i+1,j  ) >= center) * 2^2;  % Bottom
            code = code + (img(i+1,j-1) >= center) * 2^1;  % Bottom-left
            code = code + (img(i  ,j-1) >= center) * 2^0;  % Left

            lbp(i-1,j-1) = code;   % Store the final LBP code at that pixel
        end
    end

    %create histogram of LBP codes (values range from 0 to 255 → 256 bins)
    lbpVec = histcounts(lbp(:), 0:256);  

    %normalize the histogram to sum to 1 (makes it independent of image size)
    lbpVec = lbpVec / sum(lbpVec);  

end





