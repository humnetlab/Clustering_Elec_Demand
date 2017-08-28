function plotClusterUse(regularity_matrixA, data_file_A, regularity_matrixT, data_file_T, noClusters, fontsizeForSmall, fontsizeSmallGCA)

% get the mean daily use per user
meanDailyUseUser = zeros(length(regularity_matrixA),1);
for i = 1:length(regularity_matrixA)
    UoI = regularity_matrixA(i,1);
    Use = data_file_A(data_file_A(:,2)==UoI,4:99);
    meanDailyUseUser(i,1) = sum( mean(Use, 1) );
end
% now for each cluster
meanClusterUseA = zeros(12,1);
for i = 1:noClusters
    meanClusterUseA(i) = mean( meanDailyUseUser(regularity_matrixA(:,64)==i) );
end
% Torino
meanDailyUseUser = zeros(length(regularity_matrixT),1);
for i = 1:length(regularity_matrixT)
    UoI = regularity_matrixT(i,1);
    Use = data_file_T(data_file_T(:,2)==UoI,4:99);
    meanDailyUseUser(i,1) = sum( mean(Use, 1) );
end
meanClusterUseT = zeros(12,1);
for i = 1:noClusters
    meanClusterUseT(i) = mean( meanDailyUseUser(regularity_matrixT(:,64)==i) );
end

figure;
a = zeros(length(meanClusterUseA),2);
a(:,1) = meanClusterUseA;
a(:,2) = meanClusterUseT;
gw = 0.9 ; % almost touching, 1 is touching
x = 1:size(a,1);
fig(1) = bar(x-gw/4,a(:,1),gw/2) ; hold on ;
fig(2) = bar(x+gw/4,a(:,2),gw/2) ; hold off ;
set(fig(1),'FaceColor',[0.2081, 0.1663, 0.5292]) ;
set(fig(2),'FaceColor',[0.9, 0, 0]) ;
ylabel('Average Consumption [kWh]', 'FontSize', fontsizeForSmall)
xlabel('Cluster ID', 'FontSize', fontsizeForSmall)
xlim([0.5 12.5])
xticks([1:12])
set(gca, 'FontSize', fontsizeSmallGCA)
l = cell(1,2);
l{1}='Austin'; l{2}='Torino';
legend([fig(1), fig(2)],l);