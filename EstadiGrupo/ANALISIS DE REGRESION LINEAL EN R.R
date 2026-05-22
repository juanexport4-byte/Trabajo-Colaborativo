# ------------------------------------------
# ANALISIS DE REGRESION LINEAL EN R
# Archivo: 3_analisis_r.R
# ------------------------------------------

# 1. Simular datos
set.seed(123)

# Variable independiente (X)
x <- c(1,2,3,4,5,6,7,8,9,10)

# Variable dependiente (Y)
# Simulamos una relación lineal con algo de error
y <- 5 + 2*x + rnorm(10, mean = 0, sd = 2)

# Ver datos
datos <- data.frame(x, y)
print(datos)

# ------------------------------------------
# 2. Crear el modelo de regresión lineal
# ------------------------------------------

modelo <- lm(y ~ x)

# Ver resumen estadístico del modelo
summary(modelo)

# ------------------------------------------
# 3. Graficar puntos y línea de regresión
# ------------------------------------------

plot(x, y,
     main = "Regresión Lineal",
     xlab = "Variable X",
     ylab = "Variable Y",
     pch = 19)

# Agregar línea de regresión
abline(modelo, col = "red", lwd = 2)

# ------------------------------------------
# 4. Evaluar residuos (errores del modelo)
# ------------------------------------------

# Obtener residuos
residuos <- residuals(modelo)

print(residuos)

# Gráfico de residuos
plot(residuos,
     main = "Gráfico de Residuos",
     ylab = "Error",
     xlab = "Observación",
     pch = 19)

# Línea horizontal en 0
abline(h = 0, col = "blue", lwd = 2)

# ------------------------------------------
# 5. Diagnóstico del modelo
# ------------------------------------------

par(mfrow = c(2,2))
plot(modelo)
