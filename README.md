# Tiny-Impute: On-device Hybrid Anomaly Detection and Data Imputation



## Imputation Algorithms

Summary of the three hybrid anomaly detection and data imputation Algorithms:

### Moving Average with Simple Linear Regression (MA-SLR)

This algorithm is designed for MCUs and small CPU devices (like Arduino boards), considering their hardware limitations. In this algorithm, we developed and employed a hybrid system that seamlessly integrates moving averages with Z-score thresholding to accurately pinpoint and remove anomalous data points within a dataset. This is further augmented by utilizing a modified linear regression method for data imputation [[MA-SLR.py/IoT_Boards]](Imputation_Algorithms_IoT_Boards/MA-SLR.py)[[MA-SLR.ipynb/PC_and_RPi]](Imputation_Algorithms_PC/MA-SLR.ipynb).

### K-Nearest Neighbors with Expectation-Maximization (KNN-EM)

This algorithm is designed for edge devices (like gateways, AIoT boards, and SBCs) with processing and memory capabilities higher than MCUs. The design of this algorithm combines our highly-optimized unsupervised K-Nearest Neighbors (KNN) and Expectation-Maximization (EM) for anomaly detection and data imputation respectively [[KNN-EM.py/IoT_Boards]](Imputation_Algorithms_IoT_Boards/KNN-EM.py)[[KNN-EM.ipynb/PC_and_RPi]](Imputation_Algorithms_PC/KNN-EM.ipynb).
### Optimized Laplacian Convolutional Representation (LCR-Opt)

Here, we deeply modified and optimized a top-performing and high-resource consuming (LCR) method, that imputes missing data using a low-rank approximation model complemented by regularization techniques [[LCR-Opt.py/IoT_Boards]](Imputation_Algorithms_IoT_Boards/LCR-Opt-for-IoT-Boards.py)[[LCR-Opt.ipynb/PC_and_RPi]](Imputation_Algorithms_PC/LCR-Opt_for_RPi_and_Laptop.ipynb).

## Test Datasets

Datasets used to test Tiny-Impute algorithms MA-SLR, KNN-EM, LCR-Opt:

- Gesture Phase Segmentation: The dataset is composed by features extracted from 7 videos with people gesticulating. It contains 50 attributes divided into two files for each video [[Original Dataset]](https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation) [[Test Samples]](Datasets_Sample/Gesture_Phase_Segmentation_Sample.csv)

- Iris Flowers: A small classic dataset. Very popular datasets used for evaluating classification methods [[Original Dataset]](https://archive.ics.uci.edu/dataset/53/iris) [[Test Samples]](Datasets_Sample/Iris_Flowers_Sample.csv)

- Mammographic Mass: Discrimination of benign and malignant mammographic masses based on BI-RADS attributes and the patient's age. To access Original Dataset [[Original Dataset]](https://archive.ics.uci.edu/dataset/161/mammographic+mass) [[Test Samples]](Datasets_Sample/Mammographic_Mass_Sample.csv)

- Daily and Sports Activities: The dataset comprises motion sensor data of 19 daily and sports activities each performed by 8 subjects in their own style for 5 minutes [[Original Dataset]](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) [[Test Samples]](Datasets_Sample/Daily_Sports_Activities_Sample.csv)

- Urban Observatory - CO: Carbon Monoxide (CO) data taken from the Urban Observatory, Newcastle University [[Original Dataset]](https://data.ncl.ac.uk/collections/Urban_Observatory_Data_Newcastle/5059913)

## IoT Boards

The IoT boards used to test the three imputation algorithms over five test datasets:

- Arduino MKR1000: [CPU] SAMD21 Cortex-M0+ 48MHz. [Memory] Flash 256KB, SRAM 32KB [[Board]](https://docs.arduino.cc/hardware/mkr-1000-wifi)

- ESP 32 Dev Kit: [CPU] Xtensa LX6 240 MHz. [Memory] Flash 4MB, SRAM 520KB [[Board]](https://www.espressif.com/en/products/socs/esp32)

- Raspberry Pi 4 Model B: [CPU] Cortex-A72 1.8GHz. [Memory] M-SD 16GB, SDRAM 4GB  [[Board]](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)

## Imputation Experiments

### CircuitPython & MicroPython - IoT Boards

Set up the IoT board by installing the appropriate Python implementation by following [[CircuitPython]](https://circuitpython.org/board/doit_esp32_devkit_v1/) or [[MicroPython]](https://youtu.be/fmgQ8Dcg9uM)
To have an easier experience with coding and running the repo on MCUs, intall and use Thonny IDE.

To run the expirements on IoT Board, clone this repo, copy the dataset sample (.csv files) to the board's memory, call the same name in the code, then run the (.py file) on the board.

### Jupyter Notebooks - PC / Collab

To run the expirements on local PC, clone this repo, open the algorithm of choice (.ipynb files), run all cells in sequence.  