%% UAV3, IoT=50 - Professional Line Graphs without Titles

clear; clc;

%% Data
GBS = {'1','2','3','4'}; % X-axis labels
labels = {'Heuristic','EIPGA','AOTPACO','SFW','Proposed'}; % Algorithm names
colors = lines(5); % color palette
markers = {'o','s','d','^','v'}; % marker style for each algorithm

% Peak AoI
peak_aoi = [
    183.3 206.4 161.6 185.1 145.9;
    190.2 212.8 170.5 196.2 149.5;
    194   223.4 171.1 198.9 151;
    213.8 233.9 179.8 201 153
];

% Average AoI
avg_aoi = [
    139.904 87.766 87.864 108.862 84.952;
    154.64   98.128 105.03 122.256 87;
    167.28  111.46 116.592 143.378 102.416;
    187.668 124.64 126.544 157.33 114.4
];

% Total Distance
total_dist = [
    6431.46 5911.75 4765.16 5385.91 4633.25;
    6143.25 5539.29 4619.00 5255.97 4566.31;
    6143.63 5413.44 4557.21 5159.97 4459.28;
    5971.68 5380.18 4142.53 5069.92 4060.14
];

%% Graph Properties
lineWidth = 2.0;
markerSize = 8;
fontSize = 12;

%% Plot Function
plotLineGraph = @(Ydata, ylabelText, filename) ...
    plotProfessional(Ydata, GBS, labels, colors, markers, lineWidth, markerSize, fontSize, ylabelText, filename);

%% 1. Peak AoI
plotLineGraph(peak_aoi, 'Peak AoI', 'Peak_AoI_UAV3_IoT50.pdf');

%% 2. Average AoI
plotLineGraph(avg_aoi, 'Average AoI', 'Avg_AoI_UAV3_IoT50.pdf');

%% 3. Total Distance
plotLineGraph(total_dist, 'Average Total Distance', 'TotalDist_UAV3_IoT50.pdf');

%% --- Helper Function ---
function plotProfessional(Ydata, Xlabels, labels, colors, markers, lw, ms, fs, ylabelText, filename)
    figure('Color','w'); hold on; grid on; box on;
    nGBS = size(Ydata,1);
    nAlg = size(Ydata,2);

    lineStyles = {'-','--',':','-.','-'}; % distinguish lines

    for i = 1:nAlg
        plot(1:nGBS, Ydata(:,i), 'LineWidth', lw, 'Color', colors(i,:), ...
            'Marker', markers{i}, 'MarkerSize', ms, 'LineStyle', lineStyles{i}, 'DisplayName', labels{i});
    end

    xticks(1:nGBS); xticklabels(Xlabels);
    xlabel('Number of GBS','FontSize',fs,'FontWeight','bold');
    ylabel(ylabelText,'FontSize',fs,'FontWeight','bold');
    set(gca,'FontSize',fs,'LineWidth',1.2);
    legend('FontSize',fs-2,'Location','northwest');
    ylim([min(Ydata(:))*0.9 max(Ydata(:))*1.21]); % auto-adjust
    xlim([0.8 nGBS+0.2]);
    
    % Save as vector PDF
    exportgraphics(gcf, filename, 'ContentType','vector');
end
