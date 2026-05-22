# 1. Importamos las librerías necesarias
from sklearn.datasets import load_iris
from sklearn.model_split import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Cargamos los datos
# Este dataset ya viene integrado en scikit-learn
iris = load_iris()
X = iris.data  # Características: largo y ancho de pétalos/sépalos
y = iris.target  # Lo que queremos predecir: el tipo de flor (0, 1 o 2)

# 3. Dividimos los datos: Entrenamiento y Prueba
# Dejamos el 80% para entrenar al modelo y el 20% para evaluar si aprendió bien
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Seleccionamos el algoritmo y lo entrenamos
# Usaremos "Random Forest" (Bosque Aleatorio), que es muy potente y robusto
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)  # Aquí es donde ocurre la "magia" del aprendizaje

# 5. Ponemos a prueba el modelo
# Le pedimos que prediga las flores del grupo de prueba (que nunca ha visto)
predicciones = modelo.predict(X_test)

# 6. Evaluamos los resultados
precision = accuracy_score(y_test, predicciones)
print(f"¡Precisión del modelo!: {precision * 100:.2f}%\n")

# Mostramos un reporte detallado de cómo le fue con cada tipo de flor
print("Reporte de Clasificación:")
print(classification_report(y_test, predicciones, target_names=iris.target_names))

# 7. ¡Hagamos una predicción con una flor nueva!
# Imaginemos que encontramos una flor con estas medidas: [largo sépalo, ancho sépalo, largo pétalo, ancho pétalo]
nueva_flor = [[5.1, 3.5, 1.4, 0.2]]
prediccion_nueva = modelo.predict(nueva_flor)
nombre_flor = iris.target_names[prediccion_nueva[0]]

print(f"Para las medidas {nueva_flor[0]}, el modelo predice que es una: {nombre_flor.upper()}")
