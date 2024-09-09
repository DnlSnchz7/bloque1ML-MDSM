#----------------------------------------------------------
# Módulo 2: Uso de framework o biblioteca de aprendizaje 
#           máquina para la implementación de una solución
#
# Author:
#          Mariluz Daniela Sánchez Morales
#----------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

""" ENTRENAMIENTO: Carga de datos """

# Listas de datos donde se evaluará si un cliente realizará una compra en función de su ingreso mensual
X = [3365.37, 2920.58, 3471.23, 4156.72, 1793.93, 
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

# Convertir listas en DataFrame
df = pd.DataFrame({
    'ingreso_mensual': X,
    'compra': y
})

# Dividir el dataset en entrenamiento y testeo
X = df[['ingreso_mensual']]
y = df['compra']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Inicializar el modelo de árbol de decisión
tree_model = DecisionTreeClassifier()

# Entrenar el modelo en el conjunto de entrenamiento
tree_model.fit(X_train, y_train)

""" ENTRENAMIENTO: Set de validación """

# Predicciones en el conjunto de validación
y_val_pred = tree_model.predict(X_val)

# Métricas en el conjunto de validación
accuracy_val = accuracy_score(y_val, y_val_pred)
precision_val = precision_score(y_val, y_val_pred)
recall_val = recall_score(y_val, y_val_pred)
f1_val = f1_score(y_val, y_val_pred)
cm_val = confusion_matrix(y_val, y_val_pred)

# Mostrar métricas de evaluación en la validación
print("Validation Set Metrics:")
print(f'Accuracy en Validación: {accuracy_val:.2f}')
print(f"Precision: {precision_val:.2f}")
print(f"Recall: {recall_val:.2f}")
print(f"F1-Score: {f1_val:.2f}")
print(f"Confusion Matrix: {cm_val}")

""" ENTRENAMIENTO: Set de testeo """

# Predicciones en el conjunto de test
y_test_pred = tree_model.predict(X_test)

# Métricas en el conjunto de validación
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Mostrar métricas de evaluación en el test
print("Test Set Metrics:")
print(f'Accuracy en Validación: {accuracy_val:.2f}')
print(f"Precision: {precision_val:.2f}")
print(f"Recall: {recall_val:.2f}")
print(f"F1-Score: {f1_val:.2f}")
print(f"Confusion Matrix: {cm_val}")


""" PRUEBA: Nuevo set de datos """

# Nuevos datos para testeo
X_test_nuevos = [1500.45, 3200.67, 4500.12, 2800.40, 3700.89,
                 5000.00, 1900.55, 3300.75, 2100.20, 2600.30,
                 3400.80, 2200.90, 2800.10, 3600.70, 4000.50,
                 2900.60, 3400.20, 2400.85, 3100.30, 3300.40,
                 2200.45, 3100.50, 2700.70, 3500.60, 4300.40,
                 2500.20, 3300.10, 1900.75, 3000.80, 3100.60,
                 2000.65, 3700.10, 3100.70, 3200.20, 2500.90,
                 2900.30, 4100.70, 3000.60, 2300.55, 2500.30]

y_test_nuevos = [0, 1, 1, 0, 1,
                 1, 0, 1, 0, 0,
                 1, 0, 0, 1, 1,
                 0, 1, 0, 1, 1,
                 0, 1, 0, 1, 1,
                 0, 1, 0, 1, 1,
                 0, 1, 1, 1, 0,
                 0, 1, 1, 0, 0]

# Convertir nuevos datos en DataFrame
df_test_nuevos = pd.DataFrame({
    'ingreso_mensual': X_test_nuevos
})

# Predicciones en los nuevos datos de testeo
y_test_nuevos_pred = tree_model.predict(df_test_nuevos)

# Métricas en el conjunto de test
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

# Mostrar métricas de evaluación
print("New Test Set Metrics:")
print(f'Accuracy en Validación: {accuracy:.2f}')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Confusion Matrix: {cm}")

# Mostrar las predicciones y los valores verdaderos
for ingreso, prediccion, verdadero in zip(X_test_nuevos, y_test_nuevos_pred, y_test_nuevos):
    print(f'Ingreso: {ingreso:.2f} - Predicción: {prediccion} - Real: {verdadero}')