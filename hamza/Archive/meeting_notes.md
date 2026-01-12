12.18.2025 meeting notes

1. Validation (resolution)
    1. Check boarding test —> create a checkboard across the world 
        1. Checkboard of 10m x 10m squares: model learns really well
            1. Given inputs, what is optimal point for this
    2. Interesting: vary the lajang polynomial
2. Train it to learn a masked representation
3. Not asking spatial inform action

EXPT: checkerboard at different resolutions, see SatCLIP performance


First initial expt: 
1. Replicate SatCLIP
    1. Get sense of effective resolution —> 
        1. What is actual effective resolution 
            1. Ecoregions: level 1: aligned with actual task, Lebel 1-3 eco regions
            2. Different levels —> validate performance
            3. Find something that’s higher resolution than maybe ecolevels
        2. Still need to show data at a higher granularity within non sparse datasets
            1. Ex 1km x 1km 
2. Census block group — population density in the US


￼

Look for areas where the scaling shows higher resolution tasks and performance in scaling with that

Do inference on L=10, L=40, what is effective resolution of these (checkerboard examples), ‘out 

Exploratory stuff:
	Fourier features, learn over layers, try other 

L =10 , L=40, how much more efficient —>  

Multi scale RFF becomes large, this is more MLP
	
More of an efficiency improvement and training perspective: 
	multi scale, RFFs 



-----
SatCLIP benchmarking table

Test Continent (S2-100K) (S2-100K) (iNat) (tag) (Planet) (MP-16) (y ∼ g(c))
Asia
Air Temp.∗ R
2 ↑ 0.75 ± 0.05 0.63 ± 0.04 −0.50 ± 1.32 −3.95 ± 4.89 −2.13 ± 3.50 0.77 ± 0.28 0.20 ± 1.64
Elevation∗ 0.46 ± 0.09 0.48 ± 0.07 −0.26 ± 0.03 −0.29 ± 0.01 −0.07 ± 0.06 0.50 ± 0.03 −0.16 ± 0.06
Pop. Density∗ 0.42 ± 0.08 0.45 ± 0.04 −1.02 ± 0.32 −0.37 ± 0.04 0.05 ± 0.12 0.38 ± 0.04 0.03 ± 0.07
Countries† % Acc. ↑ 36.90 ± 4.32 19.17 ± 2.82 1.28 ± 0.01 1.12 ± 0.00 1.56 ± 0.47 23.12 ± 2.50 1.24 ± 0.12
iNaturalist∗ 19.60 ± 0.78 20.91 ± 0.77 21.49 ± 0.85 17.52 ± 0.38 16.14 ± 0.42 20.94 ± 0.38 21.08 ± 0.69
Biome∗ 25.89 ± 2.79 16.44 ± 1.21 3.00 ± 2.60 1.76 ± 0.04 37.81 ± 4.47 31.67 ± 1.91 6.24 ± 2.71
Ecoregions† 21.02 ± 1.09 10.86 ± 1.19 1.41 ± 0.14 1.49 ± 0.03 1.36 ± 0.10 6.65 ± 1.03 1.52 ± 0.47
Africa
Air Temp.∗ R
2 ↑ −4.71 ± 2.29 −1.48 ± 0.70 −2.67 ± 5.80 −7.91 ± 0.04 −17.43 ± 18.37 −9.91 ± 28.82 −27.36 ± 39.46
Elevation∗ −1.80 ± 1.74 −0.21 ± 0.09 −1.20 ± 0.55 −0.13 ± 0.06 −0.79 ± 0.43 −0.34 ± 0.10 −2.43 ± 2.67
Pop. Density∗ 0.17 ± 0.12 0.18 ± 0.09 −0.31 ± 0.16 −0.34 ± 0.02 0.15 ± 0.05 0.32 ± 0.03 −0.50 ± 0.34
Countries† % Acc. ↑ 30.65 ± 4.23 10.22 ± 1.62 0.45 ± 0.04 0.47 ± 0.01 0.48 ± 0.00 10.32 ± 2.75 2.74 ± 2.52
iNaturalist∗ 9.53 ± 0.57 6.23 ± 0.47 8.65 ± 0.52 7.47 ± 0.53 5.18 ± 0.38 7.69 ± 0.30 9.96 ± 0.33
Biome∗ 35.72 ± 5.48 12.34 ± 1.75 1.09 ± 0.48 1.29 ± 0.04 49.86 ± 1.57 28.28 ± 3.06 1.46 ± 0.67
Ecoregions† 32.03 ± 1.19 12.91 ± 1.63 0.94 ± 0.04 0.88 ± 0.01 0.92 ± 0.12 12.41 ± 2.20 7.72 ± 3.93
# of wins 8 5 1 0 2 4 2