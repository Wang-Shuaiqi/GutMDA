# GutMDA
**GutMDA: a multiple graph convolutional networks model for predicting gut microbiota-drug associations.**

# Dataset
1)MDAD: Microbe-drug association database including 2470 microbe-drug interaction pairs, between 1373 drugs and 173 microbes;

2)MASI: Microbiota-active substance interaction database including 8760 microbe-drug interaction pairs, between 1041 drugs and 613 microbes;

# Data description
* data_sources: Names and Pubchemid for drugs; Names and Taxonomyid for microbes.
* adj_drug2microbe: interaction pairs between drugs and microbes.
* adj_drug2dis: interaction pairs between drugs and diseases.
* adj_microbe2dis: interaction pairs between microbes and diseases.
* drugsimilarity: integrated drug similarity matrix.
* microbesimilarity: integrated microbe similarity matrix.
* diseasesimilarity: integrated disease similarity matrix.

# Requirements
* Python 3.7
* PyTorch
* PyTorch Geometric
* numpy
* scipy
