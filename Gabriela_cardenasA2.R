# Cargar bibliotecas necesarias
install.packages(c("caret", "MASS", "glmnet", "boot"))
library(tidyverse)
library(caret)
library(MASS)
library(glmnet)
library(boot)

# Parte 0: Configuración del Repositorio de GitHub (debe hacerse manualmente)
# Parte 1: Exploración y Manipulación de Datos
datos <- read.csv("datos_salud.csv")

# Tomar el 1% de los datos aleatoriamente para cada tarea
set.seed(123)
datos_muestra <- datos %>% sample_frac(0.01)
datos_muestra$DiabetesBinaria <- make.names(factor(ifelse(datos_muestra$Diabetes > 0, 1, 0)))
datos_muestra$DiabetesBinaria <- factor(datos_muestra$DiabetesBinaria, levels = c("X0", "X1"))

# Subconjunto de datos para cada tarea de regresión
datos_bmi <- datos_muestra
datos_salud_mental <- datos_muestra
datos_salud_fisica <- datos_muestra

# Parte 2: KNN
# Dividir los datos de muestra en conjuntos de entrenamiento y prueba (80% - 20%)
set.seed(189)  # Establecer semilla para reproducibilidad
indices_entrenamiento <- createDataPartition(datos_muestra$DiabetesBinaria, p = 0.8, list = FALSE, times = 1)
datos_entrenamiento <- datos_muestra[indices_entrenamiento, ]
datos_prueba <- datos_muestra[-indices_entrenamiento, ]

# Definir control de entrenamiento usando validación cruzada de 10 veces y probabilidades de clase
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# Entrenar el modelo KNN
set.seed(40)  # Establecer semilla para reproducibilidad
modelo_knn <- train(DiabetesBinaria ~ ., data = datos_entrenamiento, method = "knn", trControl = ctrl, tuneLength = 10)  # Probar valores de 'k' de 1 a 10

# Mostrar resultados de ajuste para seleccionar el mejor valor de 'k'
print(modelo_knn)

# Hacer predicciones en el conjunto de prueba
predicciones_knn <- predict(modelo_knn, newdata = datos_prueba)

# Evaluar el modelo KNN usando las probabilidades de clase
confusionMatrix(predicciones_knn, datos_prueba$DiabetesBinaria)

# Parte 3: Regresión Lineal y Multilineal (como se muestra en la respuesta anterior)

# Regresión BMI
modelo_bmi <- lm(BMI ~ ., data = datos_bmi)
resultados_cv_bmi <- cv.glm(data = datos_bmi, glmfit = modelo_bmi, K = 10)
print(resultados_cv_bmi)

# Regresión Salud Mental
modelo_salud_mental <- lm(SaludMental ~ ., data = datos_salud_mental)
resultados_cv_salud_mental <- cv.glm(data = datos_salud_mental, glmfit = modelo_salud_mental, K = 10)
print(resultados_cv_salud_mental)

# Regresión Salud Física
modelo_salud_fisica <- lm(SaludFisica ~ ., data = datos_salud_fisica)
resultados_cv_salud_fisica <- cv.glm(data = datos_salud_fisica, glmfit = modelo_salud_fisica, K = 10)
print(resultados_cv_salud_fisica)
