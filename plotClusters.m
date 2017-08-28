function Cent = plotClusters(noClusters, data_file, ylimits, windows)

idx = data_file(:, 100); 
Cent = zeros(noClusters, 96); 
no_xlabel_figs = [1,2,3,4,5,6,7,8]; 
no_ylabel_figs = [2,3,4,6,7,8,10,11,12]; 

figure('units','normalized','position',[0.5 0.5 0.35 0.4]);
for i = 1:noClusters
    profiles = data_file(idx == i, 4:99);
    subplot(windows(1),windows(2),i)
    x = repmat(1:96, size(profiles,1),1);
    X = reshape(x, [size(x, 1)*size(x,2),1]);
    K = reshape(profiles, [size(x, 1)*size(x,2),1]);
    dscatter(X./4, K*4)
    hold on  
    Cent(i,:) = mean(profiles,1);
    plot((1:96)./4, 4*Cent(i,:), 'r', 'LineWidth', 2)
    ylim(ylimits)
    xlim([0 24])
    if ismember(i,no_xlabel_figs)
        set(gca,'XTickLabel',[]);
    else
        xlabel('Time [hr]','FontSize',16)
    end
    if ismember(i,no_ylabel_figs)
        set(gca,'YTickLabel',[]);
    else
        ylabel('Demand [kW]','FontSize',16)
    end
    set(gca,'FontSize',16,'FontName','Sans Open')
end
hold off

end 