# Introduction
This repository contains the codes to early predictor for the onset of critical transitions in networked dynamical systems.

If you use anything in this repository, then please cite:

Zijia Liu, Xiaozhu Zhang, Xiaolei Ru, Ting-Ting Gao, Jack Murdoch Moore, and Gang Yan, [_Early Predictor for the Onset of Critical Transitions in Networked Dynamical Systems_](https://journals.aps.org/prx/accepted/e2075Kb9Zde1860517e53a2509870f0dbc868ad39), Physical Review X, 2024

This paper has received prominent recognition, having been featured in _Nature Physics_ through Dr. Karen Mudryk’s article titled [_"Precise Precognition"_](https://www.nature.com/articles/s41567-024-02623-9). Additionally, the American Physical Society (APS) has selected it for highlighting, with Prof. Naoki Masuda penning a captivating viewpoint entitled [_"Predicting Tipping Points in Complex Systems"_](https://physics.aps.org/articles/v17/110).

## Files
+ `biomass_data_generating.py` - File to generate data for resource biomass systems.
+ `biomass_prediction.py` - File for predicting the critical time of sharp decline of resource biomass.
+ `GNN_RNNmodel.py` - File of our GIN-GRU deep learning neural network architecture.
+ `NPRT_HN.py` - File to test the robustness of our approach against different fractions of incomplete data.
+ `NPRT_SNR.py` - File to test the robustness of our approach against different SNR (dB) of observational noise.
+ `transient_conti_generating.py` - File to generate data for the circumstance in which the control parameter continuously increases.
+ `transient_prediction.py` - File to test the robustness of our approach against transient data.
+ `CTPiCS.py` - File to pre-train joint model on massive synthetic data of three systems.
+ `CTPiCS2real.py` - File to fine-tune joint model and then predict a new empirical system.

## Folder
+ `empirical_data` - Data of a new empirical system.
