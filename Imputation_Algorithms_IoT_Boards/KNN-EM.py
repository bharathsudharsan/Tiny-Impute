import math
import random
import time

def read_csv(file_name: str, num_rows_to_read: [int] = None) -> list[float]:
    """
    Read data from a CSV file and return it as a list of floats.

    Args:
        file_name (str): The name of the CSV file to read.
        num_rows_to_read (int, optional): The number of rows to read from the CSV file. If not specified, reads all rows.

    Returns:
        List[float]: The data from the CSV file as a list of floats.
    """
    data = []

    with open(file_name, 'r') as f:
        lines = f.readlines()[1:]  # Skip the first line (header) and read the rest
        for i, line in enumerate(lines):
            if num_rows_to_read is not None and i >= num_rows_to_read:
                break  # Stop reading after the specified number of rows

            row = float(line.strip())
            data.append(row)

    return data

def euclidean_distance(x1: float, x2: float) -> float:
    """
    Compute the Euclidean distance between two numbers.

    Args:
        x1 (float): The first number.
        x2 (float): The second number.

    Returns:
        float: The Euclidean distance between x1 and x2.
    """
    return abs(x1 - x2)

def knn(data: list[float], k: int, threshold: float) -> list[float]:
    """
    Detect outliers using the k-nearest neighbors (KNN) algorithm.

    Args:
        data (List[float]): List of data points.
        k (int): Number of nearest neighbors to consider.
        threshold (float): Threshold for outlier detection.

    Returns:
        List[float]: List of detected outliers.
    """
    data.sort()
    outliers = []
    for i in range(len(data)):
        distances = [(euclidean_distance(data[i], data[j]), j) for j in range(len(data)) if j != i]
        distances.sort(key=lambda x: x[0])

        # If distance to kth nearest neighbor is larger than threshold, consider it as an outlier
        if distances[k-1][0] > threshold:
            outliers.append(data[i])

    return outliers

def remove_outliers(dataset: list[float], outliers: list[float]) -> None:
    """
    Remove outliers from a dataset.

    Args:
        dataset (List[Optional[float]]): List of data points where outliers should be removed.
        outliers (List[Optional[float]]): List of outliers to be removed from the dataset.
    """
    for number in outliers:
        while number in dataset:
            index = dataset.index(number)
            dataset[index] = None

def introduce_missingness(data: list[float], missingness_percentage: float) -> list[Optional[float]]:
    """
    Introduce missingness to a dataset by setting random values to None.

    Args:
        data (List[Optional[float]]): The dataset to introduce missingness to.
        missingness_percentage (float): The percentage of missing values to introduce.

    Returns:
        List[Optional[float]]: The dataset with missing values (None).
    """
    num_missing = int(len(data) * missingness_percentage / 100)
    missing_indices = []
    while len(missing_indices) < num_missing:
        r = random.randint(0, len(data) - 1)
        if r not in missing_indices:
            missing_indices.append(r)
    for i in missing_indices:
        data[i] = None
    return data

def calculate_mean_variance(data: list[float], weights: list[float]) -> tuple[float, float]:
    """
    Calculate the weighted mean and variance of a dataset.

    Args:
        data (List[float]): The dataset.
        weights (List[float]): The weights associated with each data point.

    Returns:
        Tuple[float, float]: The mean and variance of the dataset.
    """
    n = len(data)
    mean = sum(w * x for x, w in zip(data, weights)) / n
    variance = sum(w * (x - mean) ** 2 for x, w in zip(data, weights)) / n
    return mean, variance

def em_imputation(data: list[float], num_iterations: int) -> list[float]:
    """
    Perform Expectation-Maximization (EM) imputation on a dataset with missing values.

    Args:
        data (List[float]): The dataset with missing values (use None for missing values).
        num_iterations (int): The number of EM iterations.

    Returns:
        List[float]: The imputed dataset with missing values filled in.
    """
    non_missing_data = [x for x in data if x is not None]
    mean, variance = calculate_mean_variance(non_missing_data, [1] * len(non_missing_data))

    for _ in range(num_iterations):
        estimated_data = [x if x is not None else mean for x in data]
        mean, variance = calculate_mean_variance(estimated_data, [1] * len(estimated_data))

    imputed_data = [x if x is not None else mean for x in data]
    return imputed_data

def calculate_rmse(original_data: list[float], imputed_data: list[float]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between original and imputed data.

    Args:
        original_data (List[float]): The original data.
        imputed_data (List[float]): The imputed data to be compared to the original.

    Returns:
        float: The RMSE between the original and imputed data.
    """
    n = len(original_data)
    squared_errors = [(original - imputed) ** 2 for original, imputed in zip(original_data, imputed_data)]
    mean_squared_error = sum(squared_errors) / n
    rmse = math.sqrt(mean_squared_error)
    return rmse

def calculate_mae(original_data: list[float], imputed_data: list[float]) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between original and imputed data.

    Args:
        original_data (List[float]): The original data.
        imputed_data (List[float]): The imputed data to be compared to the original.

    Returns:
        float: The MAE between the original and imputed data.
    """
    n = len(original_data)
    absolute_errors = [abs(original - imputed) for original, imputed in zip(original_data, imputed_data)]
    mae = sum(absolute_errors) / n
    return mae

# Batch size for data processing
batch_size=20

# Percentage of missingness in the data
missingness_percentage = 20

# Set number of iterations 
num_iterations = 50

# Define the number of neighbors and threshold
k = 3
threshold = 2.0

# Read dataset Sample
# file_name='Gesture_Phase_Segmentation_Sample.csv'
# original_data = read_csv(file_name,batch_size)
original_data = read_csv('dataset.csv',batch_size)
raw_data=original_data

# Measure the start time for execution time calculation
start_time = time.monotonic()

# Find and remove outliers
outliers = knn(raw_data, k, threshold)
remove_outliers(raw_data,outliers)

# Introduce missingness into the data with specified missingness percentage
data_with_missingness = introduce_missingness(raw_data[:], missingness_percentage)

# Perform EM imputation
imputed_data = em_imputation(data_with_missingness, num_iterations)

# Measure the end time for execution time calculation
end_time = time.monotonic()

# Calculate elapsed time in milliseconds
elapsed_time_seconds = end_time - start_time
elapsed_time_ms = elapsed_time_seconds * 1000

# Calculate RMSE and MAE between original and imputed data
rmse = calculate_rmse(original_data, imputed_data)
mae = calculate_mae(original_data, imputed_data)

# Print the results for the current dataset
print(f'Execution time: {elapsed_time_ms}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')