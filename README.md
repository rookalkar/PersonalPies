# Personal Pies
This repository contains analysis done for the 'Personal Pies' project.

# Models
We build 5 models using the cyclical power model. These models are saved in the folder 'models' and are as follows:
1. single_cycle_model - this is the basic cyclical power model
2. weighted_cycles_model - this is a mixture model, built from multiple single cycle models
3. pooled_wts_indv_power - this model describes that the population chooses the same strategy when estimating a proportion, however each person has a different beta value (power coefficient)
4. indv_wts_pooled_power_model - this model describes that each person chooses a different strategy when estimating proportions, however the population as a whole has the same beta value (power coefficient)
5. indv_wts_and_power_model - this model describes that each person chooses a differnt strategy and has a different beta value

# Data
We use data from two previous studies:
1. Simplevis:
Link to study: https://eagereyes.org/blog/2016/a-reanalysis-of-a-study-about-square-pie-charts-from-2009
Link to data: https://github.com/eagereyes/pie-summaries/tree/master/data
2. Arcs Angles Areas
Link to study: https://kosara.net/papers/2016/Skau-EuroVis-2016.pdf
Link to data: https://github.com/eagereyes/pie-summaries/blob/master/data/arcs-angles-areas-merged.csv

# Analysis
There are two R scripts in this repository:
1. SimpleVisAnalysis.Rmd: this file contains the model fits of all 5 models to both the datasets 
2. Model Comparison.Rmd: this file compares the 5 models using kfold distributions
