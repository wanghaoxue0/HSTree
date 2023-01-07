# R package for HSTree

**Paper Author : Abhineet Agarwal, Yan Shuo Tan, Omer Ronen, Chandan Singh, Bin Yu**

**R Package Author:  Haoxue Wang (hw613@cam.ac.uk)---University of Cambridge**

This package is R version for the Hierarchical Shinkage algorithm based on python, there is another R package for [FIGS](https://github.com/wanghaoxue0/figs). algorithm. Hopefully more R version for [imodels](https://github.com/csinva/imodels) will be developed in the furture. The introduction manual of the package is in [Manual](https://github.com/wanghaoxue0/HSTree/blob/main/HSTree_0.8.0.pdf). 

#### Introduction

Hierarchical shinkage is an extremely fast post-hoc regularization method which works on any decision tree (or tree-based ensemble, such as Random Forest). It does not modify the tree structure, and instead regularizes the tree by shrinking the prediction over each node towards the sample means of its ancestors (using a single regularization parameter). Experiments over a wide variety of datasets show that hierarchical shrinkage substantially increases the predictive performance of individual decision trees and decision-tree ensembles.  https://arxiv.org/abs/2202.00858

<img src="https://github.com/wanghaoxue0/HSTree/blob/main/image.png" style="zoom:20%" />

##### Install all the packages 

```R
install_github("wanghaoxue0/HSTree")
library(rpart)
library(randomForest)
library(gbm)
#library(HSTree) 
source("fit.R")
source("fitCV.R")
```

##### Use the cross validation

```R
set.seed(2023)
X=read.csv("X.csv",header = FALSE)
y=read.csv("y.csv",header = FALSE)
colnames(y) <-"y"
fit <- HSTreeRegressorCV(X, y, reg_param=c(0.1, 1, 10, 20, 50, 100, 500), cv=4, verbose=TRUE, shrinkage="constant") # the default estimator is CART

```

##### Split the dataset

```R
# split the data into 3:1
smp_size <- floor(0.75 * nrow(X))
train_ind <- sample(seq_len(nrow(X)), size = smp_size)
X_train <- X[train_ind, ]
y_train <- data.frame(y[train_ind, ])
colnames(y_train) <-"y"
X_test <- X[-train_ind, ]
y_test <- data.frame(y[-train_ind, ])
colnames(y_test) <-"y"
```

##### Compare the original CART and hierarchical shrinkage tree

```R
# original decision tree model
fit <- rpart(y~., data=data.frame(X_train,y_train), control = rpart.control(maxdepth = 5))
fit1 <- HSTreeRegressor(X_train, y_train, shrinkage="constant") # the default estimator is CART
fit2 <- HSTreeRegressor(X_train, y_train, estimator="CART") # the default shrinkage method is node_based
msep = sum((predict(fit, X_test)-y_test[[1]])^2)/nrow(X)
msep1 = sum((predict(fit1, X_test)-y_test[[1]])^2)/nrow(X)
msep2 = sum((predict(fit2, X_test)-y_test[[1]])^2)/nrow(X)
plot(fit1)
text(fit1, use.n = TRUE)
```

##### Compare original random forest and with hierarchical shrinkage

```R
fit <- randomForest(X_train, y_train[[1]], ntree=50, maxnodes=5)
fit1 <- HSTreeRegressor(X_train, y_train, estimator="RandomForest")  # the default shrinkage method is node_based
fit2 <- HSTreeRegressor(X_train, y_train, estimator="RandomForest", shrinkage="constant")
fit3 <- HSTreeRegressor(X_train, y_train, estimator="RandomForest", shrinkage="leaf_based")

msep = sum((predict(fit, X_test)-y_test[[1]])^2)/nrow(X)
msep1 = sum((predict(fit1, X_test)-y_test[[1]])^2)/nrow(X)
msep2 = sum((predict(fit2, X_test)-y_test[[1]])^2)/nrow(X)
msep3 = sum((predict(fit3, X_test)-y_test[[1]])^2)/nrow(X)
```

##### Compare original gradient boosting model and with hierarchical shrinkage

```R
fit <- gbm(y~., data = data.frame(X_train,y_train), n.trees=100, interaction.depth=2)
fit1 <- HSTreeRegressor(X_train, y_train, interaction.depth=2, estimator="GradientBoosting")  # the default shrinkage method is node_based
fit2 <- HSTreeRegressor(X_train, y_train, interaction.depth=2, estimator="GradientBoosting", shrinkage="constant")
msep = sum((predict(fit, X_test)-y_test[[1]])^2)/nrow(X)
msep1 = sum((predict(fit1, X_test)-y_test[[1]])^2)/nrow(X)
msep2 = sum((predict(fit2, X_test)-y_test[[1]])^2)/nrow(X)

# to print the structure for each tree
# single= pretty.gbm.tree(fit, i.tree = 1)
```


