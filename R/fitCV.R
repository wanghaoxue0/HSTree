# R code for hierarchical shrinkage based on original python code
# hierarchical shrinkage with cross validation
# author: Haoxue Wang (hw613@cam.ac.uk)
# University of Cambridge
# date: 6 January, 2022


#' HSTree fit for classification with cross validation
#'
#' modify the decision tree(or each tree in ensemble) structure with the best regularization parameter chosen by cross validation
#'
#'
#' @param X the design matrix
#' @param y the response vector
#' @param cv cross validation level, default is 4 (divide the data into 4 portions, train:test=3:1)
#' @param verbose whether to print the cross validation process and result
#' @param reg_param numerical array of regularization parameter, default is c(0.1, 1, 10, 50, 100, 500)
#' @param max_leaf_nodes the maximum number of leaf nodes, the default is 20
#' @param interaction.depth Integer specifying the maximum depth of each tree (i.e., the highest level of variable interactions allowed). A value of 1 implies an additive model, a value of 2 implies a model with up to 2-way interactions, etc. Default is 1.
#' @param estimator CART decision tree or tree ensemble model (e.g. RandomForest or GradientBoosting)
#' Defaults to CART Classification Tree with 20 max leaf nodes
#' Note: this estimator will be directly modified and keep its original functions
#' @param shrinkage shrinkage methods, default is "node_based", options are:
#'  1. node_based shrinks based on number of samples in parent node
#'  2. leaf_based only shrinks leaf nodes based on number of leaf samples
#'  3. constant shrinks every node by a constant lambda
#'
#' @return call: the input setting
#' @return fit: keep all the output as the orginal estimator, only replace the value in each node based on shrinkage methods
#' @return regularization: chr "HSTree"
#' @return shrinkage: the hierarchical shrinkage method used in this model
#' @return class: the estimator class


#' @usage HSTreeClassifierCV(X, y, cv=4, verbose=FALSE, reg_param=c(0.1, 1, 10, 50, 100, 500), max_leaf_nodes=20, interaction.depth=1, estimator="CART", shrinkage="node_based")


HSTreeClassifierCV <- function(X, y, cv=4, verbose=FALSE, reg_param=c(0.1, 1, 10, 50, 100, 500), max_leaf_nodes=20, interaction.depth=1, estimator="CART", shrinkage="node_based"){
  smp_size <- floor((1-1/cv) * nrow(X))
  train_ind <- sample(seq_len(nrow(X)), size = smp_size)
  X_train <- X[train_ind, ]
  y_train <- data.frame(y[train_ind, ])
  colnames(y_train) <-"y"
  X_test <- X[-train_ind, ]
  y_test <- data.frame(y[-train_ind, ])
  colnames(y_test) <-"y"
  msep = rep(NA,length(reg_param))
  for (i in 1:length(reg_param)){
    reg=reg_param[i]
    fit = HSTreeRegressor(X_train, y_train, reg_param=reg, max_leaf_nodes=max_leaf_nodes, interaction.depth=interaction.dept, estimator=estimator, shrinkage=shrinkage)
    msep[i] = sum((predict(fit, X_test)-y_test[[1]])^2)/nrow(X)
  }
  best=reg_param[which.min(msep)]
  if (verbose==TRUE){
    cat("the best regulization parameter is", best, ", its mean square error is", min(msep))
  }
  fit = HSTreeRegressor(X_train, y_train, reg_param=best, max_leaf_nodes=max_leaf_nodes, interaction.depth=interaction.dept, estimator=estimator, shrinkage=shrinkage)
  structure(fit,
            reg_param=best,
            shrinkage=shrinkage,
            regularization="HStreeCV")
}


#' HSTree fit for regression with cross validation
#'
#' modify the decision tree(or each tree in ensemble) structure with the best regularization parameter chosen by cross validation
#'
#'
#' @param X the design matrix
#' @param y the response vector
#' @param cv cross validation level, default is 4 (divide the data into 4 portions, train:test=3:1)
#' @param verbose whether to print the cross validation process and result
#' @param reg_param numerical array of regularization parameter, default is c(0.1, 1, 10, 50, 100, 500)
#' @param max_leaf_nodes the maximum number of leaf nodes, the default is 20
#' @param interaction.depth Integer specifying the maximum depth of each tree (i.e., the highest level of variable interactions allowed). A value of 1 implies an additive model, a value of 2 implies a model with up to 2-way interactions, etc. Default is 1.
#' @param estimator CART decision tree or tree ensemble model (e.g. RandomForest or GradientBoosting)
#' Defaults to CART Classification Tree with 20 max leaf nodes
#' Note: this estimator will be directly modified and keep its original functions
#' @param shrinkage shrinkage methods, default is "node_based", options are:
#'  1. node_based shrinks based on number of samples in parent node
#'  2. leaf_based only shrinks leaf nodes based on number of leaf samples
#'  3. constant shrinks every node by a constant lambda
#'
#' @return call: the input setting
#' @return fit: keep all the output as the orginal estimator, only replace the value in each node based on shrinkage methods
#' @return regularization: chr "HSTree"
#' @return shrinkage: the hierarchical shrinkage method used in this model
#' @return class: the estimator class


#' @usage HSTreeRegressorCV(X, y, cv=4, verbose=FALSE, reg_param=c(0.1, 1, 10, 50, 100, 500), max_leaf_nodes=20, interaction.depth=1, estimator="CART", shrinkage="node_based")



HSTreeRegressorCV <- function(X, y, cv=4, verbose=FALSE, reg_param=c(0.1, 1, 10, 50, 100, 500), max_leaf_nodes=20, interaction.depth=1, estimator="CART", shrinkage="node_based"){
  smp_size <- floor((1-1/cv) * nrow(X))
  train_ind <- sample(seq_len(nrow(X)), size = smp_size)
  X_train <- X[train_ind, ]
  y_train <- data.frame(y[train_ind, ])
  colnames(y_train) <-"y"
  X_test <- X[-train_ind, ]
  y_test <- data.frame(y[-train_ind, ])
  colnames(y_test) <-"y"
  msep = rep(NA,length(reg_param))
  for (i in 1:length(reg_param)){
  reg=reg_param[i]
  fit = HSTreeRegressor(X_train, y_train, reg_param=reg, max_leaf_nodes=max_leaf_nodes, interaction.depth=interaction.dept, estimator=estimator, shrinkage=shrinkage)
  msep[i] = sum((predict(fit, X_test)-y_test[[1]])^2)/nrow(X)
  }
  best=reg_param[which.min(msep)]
  if (verbose==TRUE){
    cat("the best regulization parameter is", best, ", its mean square error is", min(msep))
  }
  fit = HSTreeRegressor(X_train, y_train, reg_param=best, max_leaf_nodes=max_leaf_nodes, interaction.depth=interaction.dept, estimator=estimator, shrinkage=shrinkage)
  structure(fit,
            reg_param=best,
            shrinkage=shrinkage,
            regularization="HStreeCV")
}


