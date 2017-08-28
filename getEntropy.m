function [entropy, ranEntropy] = getEntropy(regularity_matrix, noClusters)

entropy = zeros(size(regularity_matrix,1), 2); 
ranEntropy = zeros(size(regularity_matrix,1), 2); 

for i = 1:size(regularity_matrix,1)
    entropy(i,1) = regularity_matrix(i,1); 
    ranEntropy(i,1) = regularity_matrix(i,1); 
    row = regularity_matrix(i, 2:62); 
    row(isnan(row)) = 0; % change NaN values to 0's
    row(row == 0) = []; 
    Si = zeros(noClusters, 1); 
    Sran = zeros(noClusters,1); 
    for k = 1:noClusters
        if isempty(find(row==k, 1)) 
            Si(k) = 0; 
            Sran(k) = 0; 
        else 
            p_ci = length(find(row == k))/length(row); 
            Si(k) = -p_ci*log2(p_ci); 
            Sran(k) = -log2(p_ci); 
        end
    end
    entropy(i,2) = sum(Si); 
    ranEntropy(i,2) = sum(Sran); 
end

end 