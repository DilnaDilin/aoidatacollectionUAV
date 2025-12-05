%% UAV3, GBS4 - IoT: 20 to 100
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
    239.9 298.2 241.1 293.2 216;   % 100 IoT
    224.1 279.1 218.8 261.8 203;   % 80 IoT
    187.3 228.8 174.4 212.7 152.5; % 60 IoT
    149.1 181.3 143.3 166.6 120.1; % 40 IoT
    113.5 106.6 101.8 114.6 94.2   % 20 IoT
]);

avg_aoi = flipud([
    195.296 111.515 128.012 159.526 126.763;
    182.08125 122.25 123.7425 143.535 111.02375;
    152.13 91.01666667 101.1616667 120.47 96.25666667;
    116.5425 76.6175 85.3125 99.345 74.925;
    88.54 67.56 84.575 72.525 52.84
]);

total_dist = flipud([
    8639.433969 8299.028219 6248.510127 7634.062906 6078.172516;
    8511.04622 7538.648067 5837.087343 6628.529741 5498.452512;
    6246.196787 6065.818611 5048.215689 5944.723934 4948.215689;
    5005.233232 4842.083287 4303.989392 4897.933064 3997.162719;
    3875.888626 3228.506977 3394.335644 3367.113552 3216.900988
]);

%% Helper function to plot line graph
plotLineGraph = @(data, yLabel, fileName) ...
    plotLineGraphFunc(data, labels, colors, markers, IoT, yLabel, fileName);

%% 1. Peak AoI
plotLineGraph(peak_aoi, 'Peak AoI', 'Peak_AoI_UAV3_GBS4.pdf');

%% 2. Average AoI
plotLineGraph(avg_aoi, 'Average AoI', 'Avg_AoI_UAV3_GBS4.pdf');

%% 3. Total Distance
plotLineGraph(total_dist, 'Average Total Distance', 'TotalDist_UAV3_GBS4.pdf');

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
