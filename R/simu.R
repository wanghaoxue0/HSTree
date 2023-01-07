# R code for hierarchical shrinkage based on original python code
# simulation code
# author: Haoxue Wang (hw613@cam.ac.uk)
# University of Cambridge
# date: 6 January, 2022

# simulation function

simu <- function(simu=TRUE){


library(rpart)
library(randomForest)
library(gbm)
source("fit.R")
source("fitCV.R")
# for classification problem
# data_cls <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"), header = FALSE)
# fit4 <- rpart(V2~.,data=data_cls,control = rpart.control(maxdepth = 5))
# for regression problem
# library(care)
# data(efron2004)
# X <- efron2004$x # 442 10
# y <- efron2004$y

# for regression problem
set.seed(2023)
X=read.csv("X.csv",header = FALSE)
y=read.csv("y.csv",header = FALSE)
colnames(y) <-"y"

# split the data into 3:1
smp_size <- floor(0.75 * nrow(X))
train_ind <- sample(seq_len(nrow(X)), size = smp_size)
X_train <- X[train_ind, ]
y_train <- data.frame(y[train_ind, ])
colnames(y_train) <-"y"
X_test <- X[-train_ind, ]
y_test <- data.frame(y[-train_ind, ])
colnames(y_test) <-"y"

# compare the original CART and hierarchical shrinkage tree
# original decision tree model
fit <- rpart(y~., data=data.frame(X_train,y_train), control = rpart.control(maxdepth = 5))
fit1 <- HSTreeRegressor(X_train, y_train, shrinkage="constant") # the default estimator is CART
fit2 <- HSTreeRegressor(X_train, y_train, estimator="CART") # the default shrinkage method is node_based
msep = sum((predict(fit, X_test)-y_test[[1]])^2)/nrow(X)
msep1 = sum((predict(fit1, X_test)-y_test[[1]])^2)/nrow(X)
msep2 = sum((predict(fit2, X_test)-y_test[[1]])^2)/nrow(X)
plot(fit1)
text(fit1, use.n = TRUE)

# compare original random forest and with hierarchical shrinkage
fit <- randomForest(X_train, y_train[[1]], ntree=50, maxnodes=5)
fit1 <- HSTreeRegressor(X_train, y_train, estimator="RandomForest")  # the default shrinkage method is node_based
fit2 <- HSTreeRegressor(X_train, y_train, estimator="RandomForest", shrinkage="constant")
fit3 <- HSTreeRegressor(X_train, y_train, estimator="RandomForest", shrinkage="leaf_based")

msep = sum((predict(fit, X_test)-y_test[[1]])^2)/nrow(X)
msep1 = sum((predict(fit1, X_test)-y_test[[1]])^2)/nrow(X)
msep2 = sum((predict(fit2, X_test)-y_test[[1]])^2)/nrow(X)
msep3 = sum((predict(fit3, X_test)-y_test[[1]])^2)/nrow(X)
# compare original gradient boosting model and with hierarchical shrinkage
fit <- gbm(y~., data = data.frame(X_train,y_train), n.trees=100, interaction.depth=2)
fit1 <- HSTreeRegressor(X_train, y_train, interaction.depth=2, estimator="GradientBoosting")  # the default shrinkage method is node_based
fit2 <- HSTreeRegressor(X_train, y_train, interaction.depth=2, estimator="GradientBoosting", shrinkage="constant")
msep = sum((predict(fit, X_test)-y_test[[1]])^2)/nrow(X)
msep1 = sum((predict(fit1, X_test)-y_test[[1]])^2)/nrow(X)
msep2 = sum((predict(fit2, X_test)-y_test[[1]])^2)/nrow(X)

# Cross Validation
fit <- HSTreeRegressorCV(X, y, reg_param=c(0.1, 1, 10, 20, 50, 100, 500), cv=4, verbose=TRUE, shrinkage="constant") # the default estimator is CART

# to print the structure for each tree
# single= pretty.gbm.tree(fit, i.tree = 1)
}
