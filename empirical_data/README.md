# Introduction
This is a folder containing empirical data of vegetated ecosystems in Africa.

The data were laboriously extracted from the original global dataset, which follows the details in the paper ([_Early Predictor_](https://journals.aps.org/prx/accepted/e2075Kb9Zde1860517e53a2509870f0dbc868ad39)). 
If you use these data in a publication, project, etc., then please cite:

1. [Early Predictor](https://journals.aps.org/prx/accepted/e2075Kb9Zde1860517e53a2509870f0dbc868ad39): Zijia Liu, Xiaozhu Zhang, Xiaolei Ru, Ting-Ting Gao, Jack Murdoch Moore, and Gang Yan, _Early Predictor for the Onset of Critical Transitions in Networked Dynamical Systems_, Physical Review X, in press

2. [Tree coverage](https://doi.org/10.5067/MODIS/MOD44B.061): DiMiceli, C., R. Sohlberg, J. Townshend. MODIS/Terra Vegetation Continuous Fields Yearly L3 Global 250m SIN Grid V061. 2022, distributed by NASA EOSDIS Land Processes DAAC. Accessed YYYY-MM-DD

3. [Rainfall](https://doi.org/10.5067/TRMM/TMPA/MONTH/7): Tropical Rainfall Measuring Mission (TRMM) (2011), TRMM (TMPA/3B43) Rainfall Estimate L3 1 month 0.25 degree x 0.25 degree V7, Greenbelt, MD, Goddard Earth Sciences Data and Information Services Center (GES DISC), Accessed: YYYY-MM-DD

## Folders
+ `fine_tuning` - Data used to fine-tune on pretrained model.
+ `test` - Data of a new district.

## File
+ `linear_inter.py` - Code to linearly interpolate the raw data (in the folders) for early predicting.
