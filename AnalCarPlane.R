library(EBImage)
library(keras)
library(caret)
library(pbapply)

car_dir <- "carros/"
planes_dir <- "avioes/"
teste_dir <- "teste/"

width <- 30
height <- 30

extrair_caracteristicas <- function(dir_path, width, height){
  img_size <- width * height
  image_name <- list.files(dir_path)
  print(paste("Iniciando Processo", length(image_name), "imagens"))
  
  lista_parametros <- pblapply(image_name, function(imgname){
    img <- readImage(file.path(dir_path, imgname))
    img_resized <- resize(img, w = width, h = height)
    img_matrix <- as.matrix(img_resized@.Data)
    img_vector <- as.vector(t(img_matrix))
    
    return(img_vector)
  })
  
  feature_matrix <- do.call(rbind, lista_parametros)
  feature_matrix <- as.data.frame(feature_matrix)
  
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  return(feature_matrix)
}

car_data <- extrair_caracteristicas(dir_path = car_dir, width = width, height = height)
planes_data <- extrair_caracteristicas(dir_path = planes_dir, width = width, height = height)

car_data$label <- 0
planes_data$label <- 1

allData <- rbind(car_data, planes_data)

indices <- createDataPartition(allData$label, p = 0.90, list = FALSE)
train <- allData[indices, ]
test <- allData[-indices,]

trainLabels <- to_categorical(train$label)
testLabels <- to_categorical(test$label)

x_train <- data.matrix(train[, -ncol(train)])
y_train <- data.matrix(train[, ncol(train)])

x_test <- data.matrix(test[, -ncol(test)])
y_test <- data.matrix(test[, ncol(test)])

model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(2700)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')

summary(model)

model %>%
  compile(loss = "binary_crossentropy", 
          optimizer = optimizer_adam(),
          metrics = c('accuracy'))

hitory <- model %>%
  fit(x_train,
      trainLabels,
      epochs = 10,
      batch_size = 32,
      validation_split = 0.2)

plot(hitory)

model %>% evaluate(x_test, testLabels, verbose = 1)

pred <- model %>% predict_classes(x_test)
table(Predicted = pred, Reais = y_test)

test <- extrair_caracteristicas(teste_dir, width = width, height = height)
pred_test <- model %>% predict_classes(as.matrix(test))
pred_test
