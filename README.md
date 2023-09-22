##Tiny-Impute: On-device Hybrid Anomaly Detection and Data Imputation


###Datasets:

Datasets used to test Tiny-Impute Algorithms MA-SLR, KNN-EM, LCR-Opt:

1. Gesture Phase Segmentation: The dataset is composed by features extracted from 7 videos with people gesticulating. It contains 50 attributes divided into two files for each video [[Original dataset]](https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation) [Test Samples](Datasets_Sample/Gesture_Phase_Segmentation_Sample.csv)

2. Iris Flowers: A small classic dataset. Very popular datasets used for evaluating classification methods [Original dataset](https://archive.ics.uci.edu/dataset/53/iris) [Test Samples](Datasets_Sample/Iris_Flowers_Sample.csv)

3. Mammographic Mass: Discrimination of benign and malignant mammographic masses based on BI-RADS attributes and the patient's age. To access original dataset [Original dataset](https://archive.ics.uci.edu/dataset/161/mammographic+mass) [Test Samples](Datasets_Sample/Mammographic_Mass_Sample.csv).

4. Daily and Sports Activities: The dataset comprises motion sensor data of 19 daily and sports activities each performed by 8 subjects in their own style for 5 minutes. To access original dataset [Original dataset](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities). To access sample to be used immediately [Test Samples](Datasets_Sample/Daily_Sports_Activities_Sample.csv).

5. Urban Observatory - CO: CO data taken from the Urban Observatory, Newcastle University. [Original dataset](https://data.ncl.ac.uk/collections/Urban_Observatory_Data_Newcastle/5059913)
## Imputation Algorithms:
This work presented three algorithm:
1. Moving Average with Simple Linear Regression (MA-SLR): This algorithm is designed for MCUs and small CPU devices (like Arduino boards), considering their hardware limitations. In this algorithm, we developed and employed a hybrid system that seamlessly integrates moving averages with Z-score thresholding to accurately pinpoint and remove anomalous data points within a dataset. This is further augmented by utilizing a modified linear regression method for data imputation. To check this algorithm [Click Here](Imputation_Algorithms/MA-SLR.ipynb).
2. K-Nearest Neighbors with Expectation-Maximization (KNN-EM): This algorithm is designed for edge devices (like gateways,AIoT boards, and SBCs) with processing and memory capabilities higher than MCUs. The design of this algorithm combines our highly-optimized unsupervised K-Nearest Neighbors (KNN) and Expectation-Maximization (EM) for anomaly detection and data imputation respectively. To check this algorithm [Click Here](Imputation_Algorithms/KNN-EM.ipynb).
3. Optimized Laplacian Convolutional Representation (LCR-Opt): Here, we deeply modify and optimize a top-performing and high-resource consuming (LCR) method, that imputes missing data
using a low-rank approximation model complemented by regularization techniques. To check this algorithm for MCUs [Click Here](Imputation_Algorithms/LCR-Opt-for-IoT-Boards.ipynb). For the same code working on CPU devices [Click Here](Imputation_Algorithms/LCR-Opt_for_RPi_and_Laptop.ipynb).