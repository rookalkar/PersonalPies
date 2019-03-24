# Personal Pies
This repository contains analysis done for the 'Personal Pies' project.

# Models
We build 5 models using the cyclical power model. These models are saved in the folder 'models' and are as follows:
1. single_cycle_model - this is the basic cyclical power model
2. weighted_cycles_model - this is a mixture model, built from multiple single cycle models
3. pooled_wts_indv_power -(PWIP) this model describes that the population chooses the same strategy when estimating a proportion, however each person has a different beta value (power coefficient)
4. indv_wts_pooled_power_model - (IWPP) this model describes that each person chooses a different strategy when estimating proportions, however the population as a whole has the same beta value (power coefficient)
5. indv_wts_and_power_model - (IWIP) this model describes that each person chooses a differnt strategy and has a different beta value

# Data
We use data from the following previous study:
1. Arcs Angles Areas
Link to study: https://kosara.net/papers/2016/Skau-EuroVis-2016.pdf
Link to data: https://github.com/eagereyes/pie-summaries/blob/master/data/arcs-angles-areas-merged.csv

The data is cleaned and some answers are flipped. This clean data can be found in the file 'Arcs Angles Areas Results Data.xlsx' in the "flipped" sheet.

# Analysis
There is one R script in this repository:
1. AAA_analyisis.Rmd

#Model Fits
Since the models are large and take time to fit, we have included the model fit objects. You can uncomment pieces of code and use these fit objects for your analysis.
