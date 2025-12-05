%% UAV2, GBS4 - IoT: 20 to 100
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
% Original data is from 100 -> 20, so we flip it to 20->100
peak_aoi = flipud([
    322.3 398.3 320.1 339.2 300.1;  % 100 IoT
    279.8 340.1 275   301   259;    % 80 IoT
    239.6 287   229.7 243   213.5;  % 60 IoT
    197.6 217.4 197   184.4 170.7;  % 40 IoT
    132.7 146.3 136.5 125.4 116.5   % 20 IoT
]);

avg_aoi = flipud([
    244.482 159.673 166.261 180.033 158.261;
    213.185 138.04  150.3775 161.8375 130.3775;
    192     117.2583 135.3133 141.3483 115.4917;
    155.2775 94.6675 117.675  113.1875 96.585;
    103.925  69.455  86.925  81.88    56.925
]);

total_dist = flipud([
    6421.398615 6959.597385 5846.611345 6294.05603  5246.611345;
    5928.731685 6132.37138  5387.164074 5566.264746 5087.164074;
    5154.46434  5278.595112 4514.641064 4805.324035 4299.153169;
    4360.42036  4198.102055 3755.070387 3885.849617 3358.302984;
    3148.528601 2793.229219 2703.754024 2746.450749 2403.754024
]);

%% Helper function to plot line graph
plotLineGraph = @(data, yLabel, fileName) ...
    plotLineGraphFunc(data, labels, colors, markers, IoT, yLabel, fileName);

%% 1. Peak AoI
plotLineGraph(peak_aoi, 'Peak AoI', 'Peak_AoI_UAV2_GBS4.pdf');

%% 2. Average AoI
plotLineGraph(avg_aoi, 'Average AoI', 'Avg_AoI_UAV2_GBS4.pdf');

%% 3. Total Distance
plotLineGraph(total_dist, 'Average Total Distance', 'TotalDist_UAV2_GBS4.pdf');

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
