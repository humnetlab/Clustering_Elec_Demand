clc; clear all;
fontsizeSmallGCA = 16;
fontsizeForSmall = 17;
% 
% 25_08_2017
% Signal processing toolbox dependency
% this file combines data for both torino and austin analysis
% the intention is for it to be a self-contained single script file that
% produces all the final figures required for the paper
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% importing both data for torino and austin
% unzip input data if not already unzipped
% csv files are in the fmt explicit id, user id, day index (1 is the 1st
% March 2015). Days with all readings below 0.025kWh or nans filtered
% ALL VALUES IN kWh
data_file_A = csvread('Austin_Weekdays.csv');
data_file_T = csvread('Torino_Weekdays.csv');
readingsA = data_file_A(:, 4:99); 
readingsT = data_file_T(:, 4:99); 
%% Magnitude distribution for datasets 
daily_totalsA = sum(readingsA, 2); 
daily_totalsT = sum(readingsT, 2);

figure; 
nbins = 16;
binsRangeA = 0:2:100; 
histogram(daily_totalsA, binsRangeA,'Normalization','probability')
hold on 
binsRangeT = 0:2:100;
histogram(daily_totalsT, binsRangeT,'Normalization','probability')
xlabel('Total Daily Consumption [kWh]')
ylabel('Number of Readings')

fprintf('The average daily consumption in Austin is %f kWh \n', mean(daily_totalsA));
fprintf('The average daily consumption in Torino is %f kWh \n', mean(daily_totalsT));

%% Determining the number of clusters 

n = 20;
% readingsP_A = zeros(size(readingsA)); 
% readingsP_T = zeros(size(readingsT)); 
% %austin
% for i = 1:size(readingsP_A,1)
%     readingsP_A(i,:) = peakFindAndInterp(readingsA(i,:)); 
% end
% readingsN_A = zscore(readingsP_A); 
% %torino 
% for i = 1:size(readingsP_T,1)
%    readingsP_T(i,:) = peakFindAndInterp(readingsT(i,:));  
% end
% readingsN_T = zscore(readingsP_T); 
% sumd_A = zeros(n, 1); 
% sumd_T = zeros(n, 1); 
% for i = 1:n
%     rng default
%     [~, ~, sumd] = kmeans(readingsN_A, i, 'Distance', 'correlation','MaxIter',1000); 
%     sumd_A(i) = sum(sumd); 
%     [~, ~, sumd] = kmeans(readingsN_T, i, 'Distance', 'correlation','MaxIter',1000);
%     sumd_T(i) = sum(sumd); 
% end 
% % save the cluster goodness so dont have to redo
% save('clusterGoodness.mat','sumd_A','sumd_T')
% load('clusterGoodness.mat')
% extraVar_A = (sumd_A(1)-sumd_A)/sumd_A(1);
% extraVar_T = (sumd_T(1)-sumd_T)/sumd_T(1);
% 
% figure; 
% plot(1:n, sumd_A,'-o', 1:n, sumd_T,'-o')
% xlabel('Number of Clusters')
% ylabel('Sum of Within-Cluster Sums of Point-to-Centroid Distances')
% legend('Austin', 'Turin')
% 
% figure;
% plot(1:19, diff(extraVar_A)*100, '-o', 'LineWidth', 1.5)
% hold on
% plot(1:19, diff(extraVar_T)*100, '-o', 'LineWidth', 1.5)
% hold on
% plot([1,19],[1,1],'--','color','k')
% xlim([1,18])
% ylabel('Variance with additional cluster [%]', 'fontsize', 14)
% xlabel('Number of clusters', 'fontSize', 14)
% legend('Austin', 'Turin')
% set(gca, 'FontSize', 14)
% xticks([2,4,6,8,10,12,14,16,18])
% 
% figure;
% plot(1:20, extraVar_A*100, '-o', 'LineWidth', 1.5)
% hold on
% plot(1:20, extraVar_T*100, '-o', 'LineWidth', 1.5)
% hold on
% xlim([1,20])
% ylabel('Total variance accounted [%]', 'fontsize', 16)
% xlabel('Number of clusters', 'fontSize', 16)
% legend('Austin', 'Turin')
% set(gca, 'FontSize', 14)
% xticks([2,4,6,8,10,12,14,16,18])
% grid on
% 
% fprintf('Austin: 12 clusters explain %.2f percent of variance \n', extraVar_A(12)*100)
% fprintf('Torino: 12 clusters explain %.2f percent of variance \n', extraVar_T(12)*100)

%% kmeans clustering and density plots Austin 
readingsP_A = zeros(size(readingsA)); 
% austin preprocess
for i = 1:size(readingsP_A,1)
    readingsP_A(i,:) = peakFindAndInterp(readingsA(i,:)); 
end
readingsN_A = zscore(readingsP_A); 
% clustering and plotting
noClusters = 12; 
rng default;
[idxA, ~] = kmeans(readingsN_A, noClusters, 'Distance', 'correlation','MaxIter',1000);
data_file_A(:,100) = idxA;
ylimits = [0 3.1]; 
windows = [3, 4]; 
Cent_A =  plotClusters(noClusters, data_file_A, ylimits, windows); 
clear idx

%% kmeans clustering and density plots Torino
readingsP_T = zeros(size(readingsT));  
% torino preprocess
for i = 1:size(readingsP_T,1)
   readingsP_T(i,:) = peakFindAndInterp(readingsT(i,:));  
end
readingsN_T = zscore(readingsP_T); 
% clustering and plotting
noClusters = 12; 
rng default;
[idxT, ~] = kmeans(readingsN_T, noClusters, 'Distance', 'correlation','MaxIter',1000);
data_file_T(:,100) = idxT;
ylimits = [0 1.2]; 
windows = [3, 4]; 
Cent_T =  plotClusters(noClusters, data_file_T, ylimits, windows); 

%% magnitude distribution for austin and lognormal fit

data_sum_A = sum(readingsA, 2); 
cluster_bins = zeros(length(idxA), noClusters); 
for i = 1:noClusters
    k_idx = find(idxA == i); 
    cluster_bins(k_idx, i) = data_sum_A(k_idx); 
end

figure;
binranges = 0.0001:5:105;  
[bincounts] = histc(cluster_bins, binranges); 
MCplot = bar((0:5:100)+2.5, bincounts/length(daily_totalsA), 'stacked');
ylabel('Relative Frequnecy', 'FontSize', fontsizeForSmall)
xlabel('Total Daily Consumption [kWh]', 'FontSize', fontsizeForSmall)
hold on
% fit lognormal
parmhat = lognfit(daily_totalsA);
xt = 0:1:100;
y_probability = lognpdf(xt,parmhat(1),parmhat(2));
y_count = y_probability*5; % 5 is the bin width
h = plot(xt,y_count,'k');
set(h,'LineWidth',2); set(gca, 'FontSize', fontsizeSmallGCA)

%% magnitude distribution and lognormal fit for torino

data_sum_T = sum(readingsT, 2); 
cluster_bins = zeros(length(idxT), noClusters); 
for i = 1:noClusters
    k_idx = find(idxT == i); 
    cluster_bins(k_idx, i) = data_sum_T(k_idx); 
end

figure;
binranges = 0.001:2:40; 
[bincounts] = histc(cluster_bins, binranges); 
MCplot = bar((0:2:38)+1, bincounts/length(daily_totalsT), 'stacked');
ylabel('Relative Frequency')
xlabel('Total Daily Consumption [kWh]')
hold on
% fit lognormal
parmhat = lognfit(daily_totalsT);
xt = 0:0.5:41;
y_probability = lognpdf(xt,parmhat(1),parmhat(2));
y_count = y_probability*2;
h = plot(xt,y_count,'k');
set(h,'LineWidth',2)
set(gca, 'FontSize', fontsizeSmallGCA); hold off

%% Regularity matrix, plotting use per cluster

regularity_matrixA = getRegularityMatrix(data_file_A); 
regularity_matrixT = getRegularityMatrix(data_file_T); 

[clusterFq1A, clusterFq2A, clusterFq3A, clusterFq4A] = getClusterFq(data_file_A, regularity_matrixA);
[clusterFq1T, clusterFq2T, clusterFq3T, clusterFq4T] = getClusterFq(data_file_T, regularity_matrixT); 

% ClusterFq1 is the number of visits to that cluster
% ClusterFq2 is the fraction of total visits
% ClusterFq3 is the number of users classified by that cluster
% ClusterFq4 is the number of users who visit that many clusters
plotClusterUse(regularity_matrixA, data_file_A, regularity_matrixT, data_file_T, noClusters, fontsizeForSmall, fontsizeSmallGCA);

%% entropy 
S_max = -log2(1/12); % because there are 12 potential states
[entropyA, ranEntropyA] = getEntropy(regularity_matrixA, noClusters); 
[entropyT, ranEntropyT] = getEntropy(regularity_matrixT, noClusters); 

figure;
binedges = [0:0.1:3.8];
[N,~] = histcounts(entropyA(:,2),binedges);
h1 = bar(binedges(1:end-1)+((binedges(2)-binedges(1))/2),N/sum(N),'barwidth',1,'faceColor',[0.2081, 0.1663, 0.5292]);
set(h1,'FaceAlpha', 0.4)
xlabel('S, Shannon Entropy', 'fontsize', fontsizeForSmall)
ylabel('Relative Frequency', 'fontsize', fontsizeForSmall)
hold on
[N,edges] = histcounts(entropyT(:,2),binedges);
h2 = bar(binedges(1:end-1)+((binedges(2)-binedges(1))/2),N/sum(N),'barwidth',1,'faceColor',[0.9, 0, 0]);
set(h2,'FaceAlpha', 0.4)
legend('Austin','Torino')
xlim([0,S_max])
xticks(0:0.5:3.5)
set(gca, 'FontSize', fontsizeSmallGCA)
hold off

%% predictability 
predictabilityA = getPredictability(entropyA, regularity_matrixA);
predictabilityT = getPredictability(entropyT, regularity_matrixT); 

figure;
binedges = 0:0.025:1;
[N,~] = histcounts(predictabilityA,binedges);
h1 = bar(binedges(1:end-1)+((binedges(2)-binedges(1))/2),N/sum(N),'barwidth',1,'faceColor',[0.2081, 0.1663, 0.5292]);
set(h1,'FaceAlpha', 0.4)
xlabel('\Pi, Predictability', 'fontsize', fontsizeForSmall)
ylabel('Relative Frequency', 'fontsize', fontsizeForSmall)
hold on
[N,~] = histcounts(predictabilityT,binedges);
h2 = bar(binedges(1:end-1)+((binedges(2)-binedges(1))/2),N/sum(N),'barwidth',1,'faceColor',[0.9, 0, 0]);
set(h2,'FaceAlpha', 0.4)
legend('Austin','Torino')
xlim([0,1])
set(gca, 'FontSize', fontsizeSmallGCA)
hold off

%% assortativity heat map for austin

cluster_changes_map_A = zeros(12,12); 
cf = 1:12; 
for i = 1:size(regularity_matrixA,1)
    for j = 1:63
        if(ismember(regularity_matrixA(i,j), cf))
            current_cluster = regularity_matrixA(i,j);
        end    
        if(ismember(regularity_matrixA(i,j+1), cf))
            next_cluster = regularity_matrixA(i,j+1);
            cluster_changes_map_A(current_cluster,next_cluster) = cluster_changes_map_A(current_cluster,next_cluster) +1;
        end
    end
end

transition_matrixNA = cluster_changes_map_A/sum(sum(cluster_changes_map_A));
a_coeffs = sum(transition_matrixNA,1);
b_coeffs = sum(transition_matrixNA,2);
new_matrixA = zeros(12,12);
for i = 1:12
    for j = 1:12
        new_matrixA(i,j) = ( transition_matrixNA(i,j)-a_coeffs(i)*b_coeffs(j) )/(1-a_coeffs(i)*b_coeffs(j));
    end
end

figure 
% colormap RedBlue
imagesc(new_matrixA)
xticks(cf)
yticks(cf)
set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
xlabel('"From" Clusters', 'fontSize',16)
ylabel('"To" Clusters', 'fontSize',16)
c = colorbar;
set(gca,'FontSize',14,'FontName','Sans Open')
caxis([-0.02 0.02])

matrix_elements = 0;
for i = 1:12
    matrix_elements = matrix_elements + transition_matrixNA(i,i);
end
ab_products = 0;
for i = 1:12
    ab_products = ab_products + a_coeffs(i)*b_coeffs(i);
end
assortativityA = (matrix_elements - ab_products)/(1-ab_products);
asqb_products = 0;
for i = 1:12
    asqb_products = asqb_products + (a_coeffs(i)^2)*b_coeffs(i);
end
absq_products = 0;
for i = 1:12
    absq_products = absq_products + a_coeffs(i)*(b_coeffs(i)^2);
end
sd_errorA = sqrt(((ab_products+(ab_products)^2-asqb_products-absq_products)/(1-ab_products))/144);

%% assortativity heat map for torino

cluster_changes_map_T = zeros(12,12); 

for i = 1:size(regularity_matrixT,1)
    for j = 1:63
        if(ismember(regularity_matrixT(i,j), 1:12))
            current_cluster = regularity_matrixT(i,j);
        end    
        if(ismember(regularity_matrixT(i,j+1), 1:12))
            next_cluster = regularity_matrixT(i,j+1);
            cluster_changes_map_T(current_cluster,next_cluster) = cluster_changes_map_T(current_cluster,next_cluster) +1;
        end
    end
end


transition_matrixNT = cluster_changes_map_T/sum(sum(cluster_changes_map_T));
a_coeffs = sum(transition_matrixNT,1);
b_coeffs = sum(transition_matrixNT,2);
new_matrixT = zeros(12,12);
for i = 1:12
    for j = 1:12
        new_matrixT(i,j) = ( transition_matrixNT(i,j)-a_coeffs(i)*b_coeffs(j) )/(1-a_coeffs(i)*b_coeffs(j));
    end
end

figure 
% colormap redBlue
imagesc(new_matrixT)
xticks(cf)
yticks(cf)
set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
xlabel('"From" Clusters', 'fontSize',16)
ylabel('"To" Clusters', 'fontSize',16)
colorbar
set(gca,'FontSize',14,'FontName','Sans Open')
caxis([-0.02 0.02])

matrix_elements = 0;
for i = 1:12
    matrix_elements = matrix_elements + transition_matrixNT(i,i);
end
ab_products = 0;
for i = 1:12
    ab_products = ab_products + a_coeffs(i)*b_coeffs(i);
end
assortativityT = (matrix_elements - ab_products)/(1-ab_products);

asqb_products = 0;
for i = 1:12
    asqb_products = asqb_products + (a_coeffs(i)^2)*b_coeffs(i);
end
absq_products = 0;
for i = 1:12
    absq_products = absq_products + a_coeffs(i)*(b_coeffs(i)^2);
end
sd_errorT = sqrt( ((ab_products+(ab_products)^2-asqb_products-absq_products)/(1-ab_products))/144 );