# Metis-Project3-InstacartKaggle

Project 3: Instacart Kaggle competition - classification task on predicting which products instacart users will reorder, 
with the goal of optimizing the mean F1-score of the basket of reordered product predictions for each user.

This repo includes 3 files:

**InstaC_PreProc.py**: data preprocessing and feature extraction  
**InstaC_Modeling.py**: GBM model training, F1-optimization and prediction output  
**Mcnulty_Presentation_Jeddy**: pdf of project presentation slides   

Note that I did more feature extraction and another pass at modeling after the presentation to improve my competition score,
so not all of my final features are reflected in the presentation.

The code I used for F1 optimization was kindly provided by Faron as a kaggle kernel:
https://www.kaggle.com/mmueller/f1-score-expectation-maximization-in-o-n
