# Human Activity Recognition

The UCI HAR dataset consists of a training set and a test set. There are 7335 samples in each set and each sample is made up of 561 features. Both datasets were rescaled to 0 mean and unit variance. Outlier detection was performed on both sets which found 12102 outliers which were removed. This left 4541 samples in our training and test datasets.
A Principal component analysis was performed on the dataset which showed that 98% of the variance in the activity label was explained by 70 principal components. Each classifier was trained with 70 principal components and predictions were obtained. A confusion matrix was obtained from each classifier. A k-fold cross validation was performed for each classifier on the training set for 10 folds. The Accuracy, Precision, Recall and F1 score was also calculated for each classification technique. 

To compare 3 different classifiers on our chosen dataset a script was developed in the programming language Python. The raw training and test data sets were in text file format. They were reformatted into an Excel workbook and read into a numpy dataframe. Scikit learn’s Principal Component Analysis algorithm was used to find the optimum number of principal components. Scikit learn’s SVM, KNeighboursClassifier and RandomForestClassifier algorithms was used to classify the data. Sklearn’s metrics function for k-fold cross validation (cross_val_score) was used to calculate each classifier’s mean squared error. Each classifier’s Accuracy, Precision, Recall and F1 score was also calculated. 

Principal Component Analysis (PCA) results are shown as follows:

![PCA](https://github.com/dmcglynn10/HumanActivity/blob/main/PCA.png?raw=true)

The confusion Matrix from the RandomForest classifier is as follows:

![confustionMatrixRandomForest](https://github.com/dmcglynn10/HumanActivity/blob/main/confustionMatrixRandomForest.png?raw=true)

The confusion Matrix from the k-nn classifier is as follows:

![confustionMatrixNearestNeighbours](https://github.com/dmcglynn10/HumanActivity/blob/main/confustionMatrixNearestNeighbours.png?raw=true)

The confusion Matrix from the Support Vector Machines classifier is as follows:

![confustionMatrixSVM](https://github.com/dmcglynn10/HumanActivity/blob/main/confustionMatrixSVM.png?raw=true)
