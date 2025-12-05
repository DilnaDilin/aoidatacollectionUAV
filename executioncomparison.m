clc; clear; close all;

%% IoT devices
IoT = [10 20 30 40 50];  % Number of IoT devices

%% Algorithm labels
labels = {'Heuristic','EIPGA','AOTPACO','SFW','Proposed'};
nAlg = length(labels);

%% Colors and markers
colors = lines(nAlg);
markers = {'o','s','d','^','v'};

%% Execution Time Data (rows = IoT, cols = algorithms)
runtime = [
    2.277483582 29.10485661 32.28016188 10.7557657 10.85147769;  % 10 IoT
    4.635749316 51.64297006 90.83417058 27.31025982 29.82542255;  % 20 IoT
    8.062258148 76.57452452 208.865916 37.20178432 46.63093023;  % 30 IoT
    13.67818749 98.6393621 382.2691833 50.86180978 62.78387725;   % 40 IoT
    15.53997154 173.0769959 389.0311903 84.88706911 90.36680651    % 50 IoT
];

%% --- Execution Time Plot ---
figure('Color','w'); hold on; grid on; box on;
for i = 1:nAlg
    plot(IoT, runtime(:,i), '-o', 'LineWidth',2, 'Color', colors(i,:), ...
        'Marker', markers{i}, 'MarkerSize',8, 'DisplayName', labels{i});
end

xticks(IoT);
xlabel('Number of IoT Devices'); ylabel('Execution Time (s)');
set(gca,'YScale','log');  % Log scale for runtime comparison
legend('Location','northwest');
set(gca,'FontSize',12);
ylim([1 1000]);  % Adjust according to your data

% Save figure as PDF for paper
saveas(gcf,'ExecutionTime_Comparison.pdf');
