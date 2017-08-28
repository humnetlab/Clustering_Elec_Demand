function regularity_matrix = getRegularityMatrix(data_file)

dates = unique(data_file(:,3)); % list of all dates 
usersR = unique(data_file(:,2)); %all unique users 
% i already have dates vector 
regularity_matrix = zeros(length(usersR), 66); %each row is a user; first column user; each day (2-62), and then col 63 has the mode 

for i = 1:length(usersR)
    regularity_matrix(i,1) = usersR(i);
    for j = 1:length(dates)
        date = dates(j); 
        k_idx = find(data_file(:,2) == usersR(i) & data_file(:,3) == date); 
        if isempty(k_idx)
            k = nan; 
        else
            k = data_file(k_idx, end); % cluster for i'th user on j'th date 
        end 
        %day = getDayNumber(date); 
        regularity_matrix(i,date) = k; 
    end
    row = regularity_matrix(i,2:61); 
    row = row(row~=0 & isnan(row)==0);
    T = length(row); 
    N = length(unique(row)); 
    regularity_matrix(i,63) = N; % number of unique clusters visited by user
    [M, F] = mode(row); 
    regularity_matrix(i,64) = M; % the most frequent cluster 
    regularity_matrix(i,65) = F; % the number of times M appears 
    regularity_matrix(i,66) = T; % total number of clusters visited (mostly if there is data available)
end


end