import math
import random
import time

def read_csv(file_name: str, num_rows_to_read: Optional[int] = None) -> List[float]:
    """
    Reads a CSV file and returns a list of float values.

    Args:
        file_name (str): The name of the CSV file to read.
        num_rows_to_read (int, optional): The number of rows to read from the CSV file. If not specified, reads all rows.

    Returns:
        List[float]: A list of float values from the CSV file.
    """
    data = []
    with open(file_name, 'r') as f:
        lines = f.readlines()[1:]
        for i, line in enumerate(lines):
            if num_rows_to_read is not None and i >= num_rows_to_read:
                break
            row = float(line.strip())
            data.append(row)
    return data

def calculate_rmse(original_data: List[float], imputed_data: List[float]) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) between two lists of float values.

    Args:
        original_data (List[float]): The list of original data values.
        imputed_data (List[float]): The list of imputed data values.

    Returns:
        float: The RMSE between the original and imputed data.
    """
    n = len(original_data)
    squared_errors = [(original - imputed) ** 2 for original, imputed in zip(original_data, imputed_data)]
    mean_squared_error = sum(squared_errors) / n
    rmse = math.sqrt(mean_squared_error)
    return rmse

def calculate_mae(original_data: List[float], imputed_data: List[float]) -> float:
    """
    Calculates the Mean Absolute Error (MAE) between two lists of float values.

    Args:
        original_data (List[float]): The list of original data values.
        imputed_data (List[float]): The list of imputed data values.

    Returns:
        float: The MAE between the original and imputed data.
    """
    n = len(original_data)
    absolute_errors = [abs(original - imputed) for original, imputed in zip(original_data, imputed_data)]
    mae = sum(absolute_errors) / n
    return mae

def introduce_missingness(data: List[float], missingness_percentage: float) -> List[float]:
    """
    Introduces missingness into a list of data.

    Args:
        data (List[float]): The list of data.
        missingness_percentage (float): The percentage of missing values to introduce.

    Returns:
        List[float]: The list of data with missing values introduced.
    """
    num_missing = int(len(data) * missingness_percentage / 100)
    missing_indices = []

    while len(missing_indices) < num_missing:
        r = random.randint(0, len(data) - 1)
        if r not in missing_indices:
            missing_indices.append(r)

    for i in missing_indices:
        data[i] = 0

    return data

def moving_average(data: List[float], window_size: int) -> List[float]:
    """
    Calculates the moving average of a list of data.

    Args:
        data (List[float]): The input data.
        window_size (int): The size of the moving window.

    Returns:
        List[float]: The moving average of the data.
    """
    return [sum([x for x in data[i:i+window_size] if x != 0]) / window_size for i in range(len(data) - window_size + 1)]

def standard_deviation(data: List[float], window_size: int) -> List[float]:
    """
    Calculates the standard deviation within a moving window of data.

    Args:
        data (List[float]): The input data.
        window_size (int): The size of the moving window.

    Returns:
        List[float]: The standard deviation within the moving window.
    """
    avg = moving_average(data, window_size)
    variance = [sum([(x - avg[i])**2 for x in data[i:i+window_size] if x != 0]) / window_size for i in range(len(data) - window_size + 1)]
    return [var**0.5 for var in variance]

def detect_outliers(data: List[float], window_size: int, z_thresh: float) -> List[int]:
    """
    Detects outliers within a moving window of data based on a z-score threshold.

    Args:
        data (List[float]): The input data.
        window_size (int): The size of the moving window.
        z_thresh (float): The z-score threshold for outlier detection.

    Returns:
        List[int]: The indices of detected outliers.
    """
    outliers = []
    avg = moving_average(data, window_size)
    std_dev = standard_deviation(data, window_size)
    
    for i in range(len(data) - window_size + 1):
        if data[i + window_size - 1] != 0 and abs(data[i + window_size - 1] - avg[i]) > z_thresh * std_dev[i]:
            outliers.append(i + window_size - 1)
            data[i + window_size - 1] = 0
    return outliers

def SLR_impute(data: List[Union[float, int]]) -> List[Union[float, int]]:
    """
    Performs Simple Linear Regression (SLR) imputation to fill in missing values in a list of data.

    Args:
        data (List[Union[float, int]]): The input data with some missing values (0).

    Returns:
        List[Union[float, int]]: The data with missing values imputed using SLR.
    """
    known_data = [(i, d) for i, d in enumerate(data) if d != 0]
    missing_indices = [i for i, d in enumerate(data) if d == 0]

    if not known_data or not missing_indices:
        return data

    x_known, y_known = zip(*known_data)

    # Compute coefficients for linear regression
    n = len(x_known)
    m_x, m_y = sum(x_known) / n, sum(y_known) / n
    ss_xy = sum(y_known[i] * x_known[i] for i in range(n)) - n * m_y * m_x
    ss_xx = sum(x_known[i] * x_known[i] for i in range(n)) - n * m_x * m_x
    b_1 = ss_xy / ss_xx
    b_0 = m_y - b_1 * m_x

    for i in missing_indices:
        data[i] = b_0 + b_1 * i

    return data

# Batch size for data processing
batch_size=20

# Percentage of missingness in the data
missingness_percentage = 20

# Read dataset Sample
file_name='Gesture_Phase_Segmentation_Sample.csv'
original_data = read_csv(file_name,batch_size)

# Measure the start time for execution time calculation
start_time = time.monotonic()

# Value of window_size and z_threshold for Moving Averages
window_size = 5
z_thresh = 2

# Detect outliers using Moving Averages and the specified variables
outliers = detect_outliers(original_data, window_size, z_thresh)

# Introduce missingness into the data with specified missingness percentage
raw_data = introduce_missingness(original_data[:], missingness_percentage)
data = [float(item) for item in raw_data]

# Perform SLR imputation
imputed_data = SLR_impute(data)

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