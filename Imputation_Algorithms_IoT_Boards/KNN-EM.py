import random
import os
import time

def read_csv(file_name: str, num_rows_to_read: Union[int, None] = None) -> List[float]:
    """
    Read data from a CSV file and return it as a list of floats.

    Args:
        file_name (str): The path to the CSV file to read.
        num_rows_to_read (int, optional): The number of rows to read from the CSV file. If None, all rows are read.

    Returns:
        List[float]: A list containing the parsed floating-point values from the CSV file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    data = []
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()[1:]
            for i, line in enumerate(lines):
                if num_rows_to_read is not None and i >= num_rows_to_read:
                    break
                row = float(line.strip())
                data.append(row)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_name}") from e

    return data

def euclidean_distance(x1: float, x2: float) -> float:
    """
    Calculate the Euclidean distance between two numbers.

    Args:
        x1 (float): The first number.
        x2 (float): The second number.

    Returns:
        float: The Euclidean distance between x1 and x2.
    """
    return abs(x1 - x2)

def knn(data: List[float], k: int, threshold: float) -> List[float]:
    """
    Find outliers in a list of data using the k-nearest neighbors (KNN) algorithm.

    Args:
        data (List[float]): The list of data points.
        k (int): The number of nearest neighbors to consider.
        threshold (float): The threshold value to determine outliers.
        
    Returns:
        List[float]: A list of outlier values.
    """
    data.sort()
    outliers = []
    for i in range(len(data)):
        distances = [(euclidean_distance(data[i], data[j]), j) for j in range(len(data)) if j != i]
        distances.sort(key=lambda x: x[0])
        if distances[k-1][0] > threshold:
            outliers.append(data[i])

    return outliers

def remove_outliers(dataset: List[Union[float, None]], outliers: List[float]) -> None:
    """
    Remove outliers from a dataset by replacing them with None.

    Args:
        dataset (List[Union[float, None]]): The dataset containing data points.
        outliers (List[float]): A list of outlier values to be removed.
    """
    for number in outliers:
        while number in dataset:
            index = dataset.index(number)
            dataset[index] = None
            
def introduce_missingness(data: List[Union[float, None]], missingness_percentage: float) -> List[Union[float, None]]:
    """
    Introduce missingness into a list of data points.

    Args:
        data (List[Union[float, None]]): The list of data points.
        missingness_percentage (float): The percentage of data points to set as missing (None).

    Returns:
        List[Union[float, None]]: A list of data points with missing values (None).
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

def calculate_mean_variance(data: List[Union[float, None]], weights: List[float]) -> Tuple[float, float]:
    """
    Calculate the mean and variance of data points with weights.

    Args:
        data (List[Union[float, None]]): The list of data points (including missing values as None).
        weights (List[float]): The list of weights corresponding to data points.

    Returns:
        Tuple[float, float]: A tuple containing the mean and variance of the data.
    """
    n = len(data)
    mean = sum(w * x for x, w in zip(data, weights)) / n
    variance = sum(w * (x - mean)**2 for x, w in zip(data, weights)) / n
    return mean, variance

def em_imputation(data: List[Union[float, None]], num_iterations: int) -> List[Union[float, None]]:
    """
    Perform Expectation-Maximization (EM) imputation to fill in missing values in a list of data points.

    Args:
        data (List[Union[float, None]]): The list of data points with missing values represented as None.
        num_iterations (int): The number of EM iterations to perform.

    Returns:
        List[Union[float, None]]: A list of data points with missing values imputed.
    """
    non_missing_data = [x for x in data if x is not None]
    mean, variance = calculate_mean_variance(non_missing_data, [1] * len(non_missing_data))

    for _ in range(num_iterations):
        estimated_data = [x if x is not None else mean for x in data]
        mean, variance = calculate_mean_variance(estimated_data, [1] * len(estimated_data))

    imputed_data = [x if x is not None else mean for x in data]
    return imputed_data

def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between actual and predicted values.

    Args:
        actual (List[float]): The list of actual values.
        predicted (List[float]): The list of predicted values.

    Returns:
        float: The RMSE value.
    """
    n = len(actual)
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    mean_squared_error = sum(squared_errors) / n
    rmse = mean_squared_error ** 0.5
    return rmse

def calculate_mae(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between actual and predicted values.

    Args:
        actual (List[float]): The list of actual values.
        predicted (List[float]): The list of predicted values.

    Returns:
        float: The MAE value.
    """
    n = len(actual)
    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    mae = sum(absolute_errors) / n
    return mae

# Batch size for data processing
batch_size=20

# Percentage of missingness in the data
missingness_percentage = 20

# Number of iterations for EM imputation
num_iterations = 50

# Value of k for k-nearest neighbors
k = 3

# Threshold value for outlier detection
threshold = 2.0

# Read dataset Sample
file_name='Gesture_Phase_Segmentation_Sample.csv'
original_data = read_csv(file_name,batch_size)
raw_data=original_data

# Measure the start time for execution time calculation
start_time = time.monotonic()

# Detect outliers using k-nearest neighbors (KNN) with specified k and threshold
outliers = knn(raw_data, k, threshold)
remove_outliers(raw_data, outliers)

# Introduce missingness into the data with specified missingness percentage
data_with_missingness = introduce_missingness(raw_data.copy(), missingness_percentage)
 
# Perform EM imputation with the specified number of iterations
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
print(f'Execution time: {elapsed_time_ms} ms')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')