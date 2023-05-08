function [lineOut, fillOut] = stdshade(amatrix,alpha,acolor,F,smth)
% STDSHADE -- Standard Deviation Shading
%
%    STDSHADE shades the plot around the mean that is bounded by the
%    standard deviation of the following data. Here, each row represents a
%    row of data coming from a matrix. It allows autonomous control of
%    choosing own colors, and the transparency desired for shading. The
%    code automatically plots the shade chosen. This function only plots
%    the shaded range and doesn't plots the datapoints.
%
%    This code is a publication specific script but can be adjusted as per
%    requirements. Please follow the accompanying comments.
%
%    # Inputs:
%    - 'acolor': defines the used color (default is red)
%    - 'F': assignes the used x axis (default is steps of 1).
%    - 'alpha': defines transparency of the shading (default is no shading
%               and black mean line)
%    - 'smth': defines the smoothing factor (default is no smooth)
%
%    # Outputs:
%    - 'lineOut': defines the border for the shading plot.
%                 For example: If the 'lineOut' is chosen as zero, the
%                 shades will not be accompained by any lines along the
%                 border of the shaded area in the plot.
%    - 'fillOut': defines the filled out plot region.
%
%    # Reference: Ghosh A., and Shaikh S. et al., Lightweight Reinforcement
%    Learning Decoders for Autonomous, Scalable, Neuromorphic
%    intra-cortical Brain Machine Interface; submitted Neuromorphic
%    Computing & Interface, 2023.
%
%    # Version: v1.1
%    # Context: This function is used to generate the publication specific
%               plots. The accompanying scripts generates the plots, this
%               function specifically generates the shades. <refer: Fig. 4,
%               5, 7, 8, and 9 of the main text; and S1, S4, S5 of the
%               supplementary text.
%
% License: Please see the accompanying file named "LICENSE"
% Author: Aayushman Ghosh, University of Illinois Urbana Champaign, May 2023.
%         <aghosh14@illinois.edu>

% - Check to see if the inputs are present
if exist('acolor','var')==0 || isempty(acolor)
    acolor='r'; % Default color is red.
end
if exist('F','var')==0 || isempty(F)
    F=1:size(amatrix,2); % To determine the dimension of the x-axis. Code won't run if the matrix is empty.
end
if exist('smth','var')
    if isempty(smth)
        smth=1;
    end
else
    smth = 1; % No smoothing by default.
end
if ne(size(F,1),1)
    F=F';
end

% - Calculate the necessary values for the plots.
amean = nanmean(amatrix,1); % Calculating mean over the first dimension.
if smth > 1
    amean = boxFilter(nanmean(amatrix,1),smth); % Use boxfilter to smooth data
end
astd = nanstd(amatrix,[],1); % to get std shading

% astd = nanstd(amatrix,[],1)/sqrt(size(amatrix,1)); % to get sem shading - Publication specific line removal.
if exist('alpha','var') == 0 || isempty(alpha)
    fillOut = fill([F fliplr(F)],[amean+astd fliplr(amean-astd)],acolor,'linestyle','none');
    acolor = 'k'; % Directly assigning the color. Can be changed.
else
    fillOut = fill([F fliplr(F)],[amean+astd fliplr(amean-astd)],acolor, 'FaceAlpha', alpha,'linestyle','none');
end
if ishold == 0
    check = true; else check = false;
end
hold on;
lineOut = 0;
% lineOut = plot(F,amean, 'color', acolor,'linewidth',1.5); % Change color or linewidth to adjust mean line
if check
    hold off;
end
end

function dataOut = boxFilter(dataIn, fWidth)
% apply 1-D boxcar filter for smoothing
fWidth = fWidth - 1 + mod(fWidth,2); %make sure filter length is odd
dataStart = cumsum(dataIn(1:fWidth-2),2);
dataStart = dataStart(1:2:end) ./ (1:2:(fWidth-2));
dataEnd = cumsum(dataIn(length(dataIn):-1:length(dataIn)-fWidth+3),2);
dataEnd = dataEnd(end:-2:1) ./ (fWidth-2:-2:1);
dataOut = conv(dataIn,ones(fWidth,1)/fWidth,'full');
dataOut = [dataStart,dataOut(fWidth:end-fWidth+1),dataEnd];
end


