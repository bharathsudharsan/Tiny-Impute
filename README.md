# Tiny-Impute: On-device Hybrid Anomaly Detection and Data Imputation

## Imputation Algorithms:

Summary of the three hybrid anomaly detection and data imputation Algorithms:

### Moving Average with Simple Linear Regression (MA-SLR)

This algorithm is designed for MCUs and small CPU devices (like Arduino boards), considering their hardware limitations. In this algorithm, we developed and employed a hybrid system that seamlessly integrates moving averages with Z-score thresholding to accurately pinpoint and remove anomalous data points within a dataset. This is further augmented by utilizing a modified linear regression method for data imputation [[MA-SLR.ipynb]](Imputation_Algorithms/MA-SLR.ipynb)

### K-Nearest Neighbors with Expectation-Maximization (KNN-EM)

This algorithm is designed for edge devices (like gateways, AIoT boards, and SBCs) with processing and memory capabilities higher than MCUs. The design of this algorithm combines our highly-optimized unsupervised K-Nearest Neighbors (KNN) and Expectation-Maximization (EM) for anomaly detection and data imputation respectively [[KNN-EM.ipynb]](Imputation_Algorithms/KNN-EM.ipynb)

### Optimized Laplacian Convolutional Representation (LCR-Opt)

Here, we deeply modified and optimized a top-performing and high-resource consuming (LCR) method, that imputes missing data using a low-rank approximation model complemented by regularization techniques. To check this algorithm for MCUs [[LCR-Opt-for-IoT-Boards.ipynb]](Imputation_Algorithms/LCR-Opt-for-IoT-Boards.ipynb). For the same code working on CPU devices [[LCR-Opt_for_RPi_and_Laptop.ipynb]](Imputation_Algorithms/LCR-Opt_for_RPi_and_Laptop.ipynb)

## Datasets:

Datasets used to test Tiny-Impute algorithms MA-SLR, KNN-EM, LCR-Opt:

- Gesture Phase Segmentation: The dataset is composed by features extracted from 7 videos with people gesticulating. It contains 50 attributes divided into two files for each video [[Original Dataset]](https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation) [[Test Samples]](Datasets_Sample/Gesture_Phase_Segmentation_Sample.csv)

- Iris Flowers: A small classic dataset. Very popular datasets used for evaluating classification methods [[Original Dataset]](https://archive.ics.uci.edu/dataset/53/iris) [[Test Samples]](Datasets_Sample/Iris_Flowers_Sample.csv)

- Mammographic Mass: Discrimination of benign and malignant mammographic masses based on BI-RADS attributes and the patient's age. To access Original Dataset [[Original Dataset]](https://archive.ics.uci.edu/dataset/161/mammographic+mass) [[Test Samples]](Datasets_Sample/Mammographic_Mass_Sample.csv)

- Daily and Sports Activities: The dataset comprises motion sensor data of 19 daily and sports activities each performed by 8 subjects in their own style for 5 minutes [[Original Dataset]](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) [[Test Samples]](Datasets_Sample/Daily_Sports_Activities_Sample.csv)

- Urban Observatory - CO: Carbon Monoxide (CO) data taken from the Urban Observatory, Newcastle University [[Original Dataset]](https://data.ncl.ac.uk/collections/Urban_Observatory_Data_Newcastle/5059913)

