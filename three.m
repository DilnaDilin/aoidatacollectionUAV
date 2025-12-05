%% UAV4, GBS4 - IoT: 20 to 100
clc; clear; close all;

% IoT devices
IoT = {'20','40','60','80','100'};

% Algorithm labels
labels = {'Heuristic','EIPGA','AOTPACO','SFW','Proposed'};
nAlg = length(labels);

% Colors and markers for professional styling
colors = lines(nAlg);
markers = {'o','s','d','^','v'};

%% Data: rows = IoT (20->100), columns = algorithms
% Original data is from 100 -> 20, so flip to 20->100
peak_aoi = flipud([
    206.9 282.6 206.7 270.3 201;    % 100 IoT
    213.8 238.9 205.5 239.5 195;    % 80 IoT
    165.8 204   175.5 199.7 167.3;  % 60 IoT
    142.6 170.8 145.5 147.1 141.7;  % 40 IoT
    102.1 92.4  94    101   85.9    % 20 IoT
]);

avg_aoi = flipud([
    179.246 116.845 116.235 116.22 110.557;   % 100 IoT
    177.668 96.99375 103.84375 112.045 97.5975; % 80 IoT
    136.13 79.67833333 92.81666667 104.2333333 81.78666667; % 60 IoT
    100.6425 65.83 82.0875 90.765 66.7075;     % 40 IoT
    76.17 49.555 65 67.285 45.32                % 20 IoT
]);

total_dist = flipud([
    9931.051077 9709.369659 7191.596594 9064.641225 6704.91571;
    8431.459947 8258.318727 6685.909706 8099.570477 6159.144931;
    7501.436884 6841.139682 5852.717431 6823.946253 5455.925473;
    5905.706509 5634.124399 5164.350909 5467.73881 4934.106034;
    4261.167538 3643.640555 3619.375263 3875.480286 3275.710324
]);

%% Helper function to plot line graph
plotLineGraph = @(data, yLabel, fileName) ...
    plotLineGraphFunc(data, labels, colors, markers, IoT, yLabel, fileName);

%% 1. Peak AoI
plotLineGraph(peak_aoi, 'Peak AoI', 'Peak_AoI_UAV4_GBS4.pdf');

%% 2. Average AoI
plotLineGraph(avg_aoi, 'Average AoI', 'Avg_AoI_UAV4_GBS4.pdf');

%% 3. Total Distance
plotLineGraph(total_dist, 'Average Total Distance', 'TotalDist_UAV4_GBS4.pdf');

%% --- Function Definition ---
function plotLineGraphFunc(data, labels, colors, markers, XtickLabels, yLabel, fileName)
    figure('Color','w'); hold on; grid on; box on;
    nAlg = length(labels);
    nPoints = size(data,1);
    for i = 1:nAlg
        plot(1:nPoints, data(:,i), 'LineWidth',2, ...
            'Color', colors(i,:), ...
            'Marker', markers{i}, ...
            'MarkerSize',8, ...
            'DisplayName', labels{i});
    end
    xticks(1:nPoints); xticklabels(XtickLabels);
    xlabel('Number of IoT Devices'); ylabel(yLabel);
    legend('Location','northwest');
    set(gca,'FontSize',12);
    saveas(gcf,fileName);
end
