#----------------------------------------------------------
# Módulo 2: Módulo 2 Implementación de una técnica de 
#           aprendizaje máquina sin el uso de un framework.
#
# Author:
#          Mariluz Daniela Sánchez Morales
#----------------------------------------------------------

import numpy as np

def sigmoid(z):
    """ Función sigmoide """
    return 1 / (1 + np.exp(-z))

def h(params, samples):
    """ Función que evalua la hipótesis de la regresión logística """
    return sigmoid(np.dot(samples, params))

def compute_cost(params, samples, y):
    """ Función que calcula el costo de la regresión logística con entropía cruzada """
    hyp = h(params, samples)

    # Condiciones para evitar un error en log(0)
    epsilon = 1e-10
    hyp = np.where(hyp == 0, epsilon, hyp)
    hyp = np.where(hyp == 1, 1 - epsilon, hyp)

    cost = -np.mean(y * np.log(hyp) + (1 - y) * np.log(1 - hyp))
    return cost

def gradient_descent(params, samples, y, alpha):
    """ Cálculo del gradiente para usarlo dentro de la función logistic_regression """
    m = len(y)
    hyp = h(params, samples)
    gradient = np.dot(samples.T, (hyp - y)) / m
    params -= alpha * gradient
    return params

def scale_features(samples):
    """ Normalización de datos con el promedio """
    samples = np.array(samples)
    mean = np.mean(samples[:, 1:], axis=0)
    max_val = np.max(samples[:, 1:], axis=0)
    samples[:, 1:] = (samples[:, 1:] - mean) / max_val
    return samples

def logistic_regression(samples, y, alpha, max_epochs=50000, tolerance=1e-6):
    """ Entrenamiento del modelo con gradiente """
    samples = np.array(samples)
    y = np.array(y)
    params = np.zeros(samples.shape[1])
    errors = []

    for epoch in range(max_epochs):
        old_params = params.copy()
        params = gradient_descent(params, samples, y, alpha)
        error = compute_cost(params, samples, y)
        errors.append(error)
        # print(f"Epoch {epoch + 1}: Error = {error}")

        if np.allclose(params, old_params, atol=tolerance):
            print('Final error: ', error)
            print('Convergence reached.')
            break
        
        if epoch == max_epochs:
            print('Final error: ', error)

    return params

def calculate_accuracy(tp, tn, fp, fn):
    """ Cálculo de precisión """
    total_samples = tp + tn + fp + fn
    accuracy = (tp + tn) / total_samples
    return accuracy

""" DATOS """
samples = [3365.37, 2920.58, 3471.23, 4156.72, 1793.93, 
     2685.21, 3274.92, 3154.33, 4097.64, 4517.13, 
     2473.25, 3610.48, 2097.63, 3142.97, 2862.79, 
     3424.85, 2756.37, 3595.43, 2201.85, 2839.53, 
     2813.75, 3385.71, 2249.18, 3158.54, 2285.14, 
     3257.91, 3614.73, 3067.21, 2938.59, 2950.10, 
     3187.35, 3494.57, 3999.56, 3068.33, 2802.18, 
     3260.22, 4531.79, 3702.54, 2453.62, 3158.44, 
     3558.63, 3458.32, 2742.75, 3202.84, 3559.71,
     3777.47, 3009.20, 2000.78, 3965.82, 1523.86]

y = [1, 0, 1, 1, 0, 
     0, 1, 1, 1, 1, 
     0, 1, 0, 1, 0, 
     1, 0, 1, 0, 0, 
     0, 1, 0, 1, 0, 
     1, 1, 1, 0, 0, 
     1, 1, 1, 0, 0, 
     1, 1, 1, 0, 1, 
     1, 1, 0, 1, 1, 
     1, 1, 0, 1, 0]

""" DIVISION DE DATOS """

# Manejo de datos
samples = np.c_[np.ones(len(samples)), samples]
samples = scale_features(samples)

data_size = len(samples)

# Los datos se revuelven para posteriormente dividirlos equitativamente
indices = np.arange(data_size)
np.random.shuffle(indices)

# Rangos de partición de los datos
train_size = int(0.6 * data_size)
validation_size = int(0.2 * data_size)
test_size = data_size - train_size - validation_size

# División de datos en train, validación y test
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + validation_size]
test_indices = indices[train_size + validation_size:]

# División de datos por set de datos
X_train, Y_train = [samples[i] for i in train_indices], [y[i] for i in train_indices]
X_val, Y_val = [samples[i] for i in val_indices], [y[i] for i in val_indices]
X_test, Y_test = [samples[i] for i in test_indices], [y[i] for i in test_indices]

# Entrenar modelo
alpha = 0.3 
params = logistic_regression(X_train, Y_train, alpha)
print('Final parameters: ', params)


"""MÉTRICAS DE EVALUACIÓN"""

def confusion_matrix(y_true, y_pred):
    TP = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    return TP, TN, FP, FN

def precision_score(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall_score(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

""" CON EL SET DE VALIDACIÓN """

# Calcular preciciones con set de validación
y_pred_val = [1 if h(params, x) >= 0.5 else 0 for x in X_val]

# Calcular métricas con la validación
precision_val = precision_score(Y_val, y_pred_val)
recall_val = recall_score(Y_val, y_pred_val)
f1_val = f1_score(Y_val, y_pred_val)
conf_matrix_val = confusion_matrix(Y_val, y_pred_val)

print(f'\nValidation Precision: {precision_val}')
print(f'Validation Recall: {recall_val}')
print(f'Validation F1 Score: {f1_val}')
print('Validation Confusion Matrix:', conf_matrix_val)

"""CON EL SET DE TEST"""

# Calcular preciciones con set de testeo
y_pred_test = [1 if h(params, x) >= 0.5 else 0 for x in X_test]

# Calcular metricas con el test
precision_test = precision_score(Y_test, y_pred_test)
recall_test = recall_score(Y_test, y_pred_test)
f1_test = f1_score(Y_test, y_pred_test)
conf_matrix_test = confusion_matrix(Y_test, y_pred_test)

print(f'\nTest Precision: {precision_test}')
print(f'Test Recall: {recall_test}')
print(f'Test F1 Score: {f1_test}')
print('Test Confusion Matrix:', conf_matrix_test)

# Msotrar prediciones del Test
print("\nTest Set Predictions:")
for i, sample in enumerate(X_test):
    prediction = 1 if h(params, sample) >= 0.5 else 0
    print(f"Sample: {sample[0]}, Actual: {Y_test[i]}, Predicted: {prediction}")