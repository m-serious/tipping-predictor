# Introduction
This repository contains the codes to early predictor for the onset of critical transitions in networked dynamical systems.
+ `biomass_data_generating.py` - File to generate data for resource biomass systems.
+ `biomass_prediction.py` - File for predicting the critical time of sharp decline of resource biomass.
+ `GNN_RNNmodel.py` - File of our GIN-GRU deep learning neural network architecture.
+ `NPRT_HN.py` - File to test the robustness of our approach against different fractions of incomplete data.
+ `NPRT_SNR.py` - File to test the robustness of our approach against different SNR (dB) of observational noise.
+ `transient_conti_generating.py` - File to generate data for the circumstance in which the control parameter continuously increases.
+ `transient_prediction.py` - File to test the robustness of our approach against transient data.
+ `CTPiCS.py` - File to pre-train joint model on massive synthetic data of three systems.
+ `CTPiCS2real.py` - File to fine-tune joint model and then predict a new empirical system.
