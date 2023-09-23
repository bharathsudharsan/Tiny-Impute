from ulab import numpy as np
import ulab as numpy
import random
import math
import time
from builtins import abs

def compute_mae(predictions, targets):
    """
    Computes the Mean Absolute Error (MAE) between two sets of predictions and targets.

    Args:
        predictions (numpy.ndarray): The predicted values.
        targets (numpy.ndarray): The target values.

    Returns:
        float: The MAE.
    """
    temp = abs(predictions - targets)
    mae = (0.01*np.mean(temp, axis=0)[0]) 
    return mae

def compute_rmse(predictions, targets):
    """
    Computes the Root Mean Square Error (RMSE) between two sets of predictions and targets.

    Args:
        predictions (numpy.ndarray): The predicted values.
        targets (numpy.ndarray): The target values.

    Returns:
        float: The RMSE.
    """
    squared_diff = (predictions - targets) ** 2
    mean_squared_diff = np.mean(squared_diff, axis=0)[0]
    rmse = (0.01*np.sqrt(mean_squared_diff))
    return rmse

def laplacian(n: int, tau: int) -> np.ndarray:
    """
    Compute the discrete Laplacian operator.

    Args:
        n (int): The size of the Laplacian operator.
        tau (int): The parameter for the Laplacian.

    Returns:
        np.ndarray: The Laplacian operator.
    """
    ell = np.zeros(n)
    ell[0] = 2 * tau
    for k in range(tau):
        ell[k + 1] = -1
        ell[-k - 1] = -1
    return ell

def prox(z: np.ndarray, w: np.ndarray, lmbda: float, denominator_real: np.ndarray, denominator_imag: np.ndarray) -> np.ndarray:
    """
    Compute the proximal operator.

    Args:
        z (np.ndarray): Input array.
        w (np.ndarray): Input array.
        lmbda (float): Regularization parameter.
        denominator_real (np.ndarray): Real part of the denominator.
        denominator_imag (np.ndarray): Imaginary part of the denominator.

    Returns:
        np.ndarray: Result of the proximal operator.
    """
    T = len(z)
    padded_input = pad_to_power_of_2(lmbda * z - w)
    real_numerator, imag_numerator = np.fft.fft(padded_input)
    real_temp = np.zeros(T)
    imag_temp = np.zeros(T)
    for i in range(T):
        real_temp[i] = real_numerator[i] / denominator_real[i] if denominator_real[i] != 0 else 0
        imag_temp[i] = imag_numerator[i] / denominator_imag[i] if denominator_imag[i] != 0 else 0
    temp1 = 1 - T / (lmbda * np.sqrt(real_temp**2 + imag_temp**2))
    temp1[temp1 <= 0] = 0
    padded_real_temp = pad_to_power_of_2(real_temp * temp1)
    padded_imag_temp = pad_to_power_of_2(imag_temp * temp1)
    real_result, imag_result = np.fft.ifft(padded_real_temp, padded_imag_temp)
    return real_result[:T]

def update_z(y_train: np.ndarray, pos_train: np.ndarray, x: np.ndarray, w: np.ndarray, lmbda: float, eta: float) -> np.ndarray:
    """
    Update the variable z.

    Args:
        y_train (np.ndarray): The training data.
        pos_train (np.ndarray): Boolean array indicating positive training examples.
        x (np.ndarray): Input variable.
        w (np.ndarray): Weight variable.
        lmbda (float): Regularization parameter.
        eta (float): Learning rate parameter.

    Returns:
        np.ndarray: Updated z.
    """
    z = x + w / lmbda
    z = [(lmbda / (lmbda + eta) * z[i] + eta / (lmbda + eta) * y_train[i]) if pos_train[i] else z[i] for i in range(len(z))]
    return np.array(z)

def update_w(x: np.ndarray, z: np.ndarray, w: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Update the variable w.

    Args:
        x (np.ndarray): Input variable.
        z (np.ndarray): Input variable.
        w (np.ndarray): Weight variable.
        lmbda (float): Regularization parameter.

    Returns:
        np.ndarray: Updated w.
    """
    return w + lmbda * (x - z)

def where(condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Implement the numpy `where` function.

    Args:
        condition (np.ndarray): Boolean array indicating the condition.
        x (np.ndarray): Values to choose from when condition is True.
        y (np.ndarray): Values to choose from when condition is False.

    Returns:
        np.ndarray: Resulting array with values selected from x or y based on the condition.
    """
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if condition[i]:
            result[i] = x[i]
        else:
            result[i] = y[i]
    return result

def random_indices(T: int, num_zeros: int):
    """
    Generate random indices.

    Args:
        T (int): Total number of indices.
        num_zeros (int): Number of zeros to generate.

    Yields:
        int: Random indices.
    """
    indices = list(range(T))
    for i in range(num_zeros):
        random_index = random.randint(0, len(indices) - 1)
        yield indices.pop(random_index)
        
def pad_to_power_of_2(array: np.ndarray) -> np.ndarray:
    """
    Pad an array with zeros to the next power of 2 length.

    Args:
        array (np.ndarray): The input array.

    Returns:
        np.ndarray: The padded array.
    """
    target_len = 1
    while target_len < len(array):
        target_len *= 2
    padded_array = np.zeros(target_len)
    padded_array[:len(array)] = array
    return padded_array

def LCR(y_true: np.ndarray, y: np.ndarray, lmbda: float, gamma: float, tau: int, maxiter: int = 50) -> np.ndarray:
    """
    LCR (Laplacian Constrained Regression) algorithm.

    Args:
        y_true (np.ndarray): The true target values.
        y (np.ndarray): The input values.
        lmbda (float): Regularization parameter.
        gamma (float): Gamma parameter.
        tau (int): Tau parameter.
        maxiter (int, optional): Maximum number of iterations. Default is 50.

    Returns:
        np.ndarray: The result of the LCR algorithm.
    """
    eta = 100 * lmbda
    T = y.shape[0]
    pos_train = y != 0
    y_train = np.where(pos_train, y, np.zeros(y.shape))
    pos_test = [(y_true[i] != 0) and (y[i] == 0) for i in range(T)]
    y_test = np.where(pos_test, y_true, np.zeros(y_true.shape))
    z = y.copy()
    w = y.copy()
    laplacian_padded = pad_to_power_of_2(laplacian(T, tau))
    real_denom, imag_denom = np.fft.fft(laplacian_padded**2)
    denominator_real = lmbda + gamma * real_denom
    denominator_imag = gamma * imag_denom
    T = y_true.shape[0]
    del y_true, y
    show_iter = 10
    for it in range(maxiter):
        x = prox(z, w, lmbda, denominator_real, denominator_imag)
        z = update_z(y_train, pos_train, x, w, lmbda, eta)
        w = update_w(x, z, w, lmbda)
    print()
    return x

def read_csv(file_name: str, num_rows_to_read: int = None) -> np.ndarray:
    """
    Read data from a CSV file and return it as a NumPy array.

    Args:
        file_name (str): The name of the CSV file to read.
        num_rows_to_read (int, optional): The number of rows to read from the CSV file. If not specified, reads all rows.

    Returns:
        np.ndarray: The data from the CSV file as a NumPy array.
    """
    data = []

    with open(file_name) as f:
        header_skipped = False

        for line in f:
            if not header_skipped:
                header_skipped = True
                continue

            if num_rows_to_read is not None and len(data) >= num_rows_to_read:
                break

            row = [float(x) for x in line.strip().split(',')]
            data.append(row)

    rows, cols = len(data), len(data[0])
    result = np.zeros((rows, cols))

    for row in range(rows):
        for col in range(cols):
            result[row][col] = data[row][col]

    return result

# Batch size for data processing
batch_size=20

# Percentage of missingness in the data
missing_rate = 0.20

# Read dataset Sample
file_name='Gesture_Phase_Segmentation_Sample.csv'
original_data = read_csv(file_name,batch_size)

# Measure the start time for execution time calculation
start_time = time.monotonic()

# Find length of input data
L = len(original_data)
missing_vector = original_data.copy()

# Zero out a copy to introduce missingness
num_zeros = int(L * missing_rate)
zero_indices = list(random_indices(L, num_zeros))
for index in zero_indices:
    missing_vector[index] = 0
    
# Set LCR variable values
lmbda = 5e-3 * L
gamma = 2 * lmbda
tau = 3
maxiter = 100

# Perform LCR imputation
imputed_data = LCR(original_data, missing_vector, lmbda, gamma, tau, maxiter)

# Measure the end time for execution time calculation
end_time = time.monotonic()

# Calculate elapsed time in milliseconds
elapsed_time_seconds = end_time - start_time
elapsed_time_ms = elapsed_time_seconds * 1000

# Calculate RMSE and MAE between original and imputed data
mae=compute_mae(original_data,imputed_data)
rmse=compute_rmse(original_data, imputed_data)

# Print the results for the current dataset
print(f'Execution time: {elapsed_time_ms} ms')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')