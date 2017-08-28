function yi = peakFindAndInterp(profile)

[peaks, locs] = findpeaks(profile, 'MinPeakProminence', 0.01); 
locs = [1, locs, 96]; 
peaks = [profile(1), peaks, profile(96)]; 
yi = pchip(locs, peaks, 1:96); % interpolating based on curve shape (method called PCHIP)

end 