function predictability = getPredictability(entropy, regularity_matrix)
% simple function for estimating predictability, must be between 0 and 1

entropies = entropy(:,2);

N = regularity_matrix(:,63); 
users = length(entropies); 

iter = 993;
predictability = zeros(size(entropy));
predictability(:,1) = entropy(:,1); 

for j = 1:users
    if entropies(j)>0
        errors = zeros(iter,1);
        Si_true = entropies(j);
        PIi_guess = linspace(0.08,1,iter); 
        for i = 1:iter
            Hi = -PIi_guess(i)*log2(PIi_guess(i)) - (1-PIi_guess(i))*log2(1-PIi_guess(i));
            Si_guess = Hi + (1-PIi_guess(i))*log2(N(j)-1);
            errors(i) = abs(Si_guess - Si_true);
        end
        % find where the minumum error is
        index1 = find(errors==min(errors), 1);
        predictability(j,2) = PIi_guess(index1); 
    else
        predictability(j,2) = 1;
    end
end

end