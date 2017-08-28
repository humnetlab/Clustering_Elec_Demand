function [clusterFq, clusterFqDays, clusterFq3, clusterFq4]  = getClusterFq(data_file, regularity_matrix)

noClusters = length(unique(data_file(:,100))); % find the list of residential clusters 
clusterFq = zeros(noClusters,1); 
clusterFqDays = zeros(noClusters,1); % frequency of cluster as fraction of number of days 
 
for i = 1:noClusters
    visits = data_file(:,end) == i;
    clusterFq(i) = sum(visits); 
    clusterFqDays(i) = sum(visits)/length(data_file); 
end

users = unique(regularity_matrix(:,1)); 
clusterFq3 = zeros(noClusters, 1); 

for i = 1:noClusters
    clusterFq3(i) = length(find(regularity_matrix(:,64) == i))./length(users);
end

% ClusterFq is the number of visits to that cluster
% ClusterFqDays is the fraction of total visits
% ClusterFq3 is the number of users classified by that cluster

clusterFq4 = zeros(noClusters,1); 
for i = 1:noClusters
    clusterFq4(i) = length(find(regularity_matrix(:,63) == i))./length(users);
end

end 