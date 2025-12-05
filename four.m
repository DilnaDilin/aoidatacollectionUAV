clc; clear; close all;

%% UAV counts
UAVs = [2 3 4];

%% Algorithm labels
labels = {'Heuristic','EIPGA','AOTPACO','SFW','Proposed'};
nAlg = length(labels);

%% Colors and markers
colors = lines(nAlg);
markers = {'o','s','d','^','v'};

%% --- Data for 20 IoT devices (rows = UAVs, cols = algorithms) ---
% Peak AoI
peak_aoi = [
    132.7 146.3 136.5 125.4 116.5;  % UAV2
    113.5 106.6 101.8 114.6 94.2;   % UAV3
    102.1 92.4  94    101   85.9    % UAV4
];

% Average AoI
avg_aoi = [
    103.925  69.455  86.925  81.88   56.925;  % UAV2
    88.54    52.84   84.575  72.525  67.56;   % UAV3
    76.17    45.32   65      67.285  49.555   % UAV4
];

% Total Distance
total_dist = [
    3148.528601 2793.229219 2703.754024 2746.450749 2403.754024;  % UAV2
    3875.888626 3228.506977 3394.335644 3367.113552 3216.900988;  % UAV3
    4261.167538 3643.640555 3619.375263 3875.480286 3275.710324   % UAV4
];

%% --- Peak AoI Plot ---
figure('Color','w'); hold on; grid on; box on;
for i = 1:nAlg
    plot(UAVs, peak_aoi(:,i), '-o', 'LineWidth',2, 'Color', colors(i,:), ...
        'Marker', markers{i}, 'MarkerSize',8, 'DisplayName', labels{i});
end
xticks(UAVs);
xlabel('Number of UAVs'); ylabel('Peak AoI');
legend('Location','northeast');
set(gca,'FontSize',12);
ylim([0 max(peak_aoi(:))+20]);
saveas(gcf,'PeakAoI_20IoT.pdf');

%% --- Average AoI Plot ---
figure('Color','w'); hold on; grid on; box on;
for i = 1:nAlg
    plot(UAVs, avg_aoi(:,i), '-s', 'LineWidth',2, 'Color', colors(i,:), ...
        'Marker', markers{i}, 'MarkerSize',8, 'DisplayName', labels{i});
end
xticks(UAVs);
xlabel('Number of UAVs'); ylabel('Average AoI');
legend('Location','northeast');
set(gca,'FontSize',12);
ylim([0 max(avg_aoi(:))+20]);
saveas(gcf,'AvgAoI_20IoT.pdf');

%% --- Total Distance Plot ---
figure('Color','w'); hold on; grid on; box on;
for i = 1:nAlg
    plot(UAVs, total_dist(:,i), '-d', 'LineWidth',2, 'Color', colors(i,:), ...
        'Marker', markers{i}, 'MarkerSize',8, 'DisplayName', labels{i});
end
xticks(UAVs);
xlabel('Number of UAVs'); ylabel('Average Total Distance');
legend('Location','northeast');
set(gca,'FontSize',12);
ylim([0 max(total_dist(:))+500]);
saveas(gcf,'TotalDist_20IoT.pdf');
