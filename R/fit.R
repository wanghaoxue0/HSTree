# R code for hierarchical shrinkage based on original python code
# hierarchical shrinkage fit
# author: Haoxue Wang (hw613@cam.ac.uk)
# University of Cambridge
# date: 6 January, 2022

#' HSTree fit for classification
#'
#' modify the decision tree(or each tree in ensemble) structure based on hierarchical shrinkage regularization
#'
#'
#' @param X the design matrix
#' @param y the response vector
#' @param reg_param Higher is more regularization (can be arbitrarily large, should not be < 0)
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


#' @usage HSTreeClassifier(X, y, reg_param=1, max_leaf_nodes=20, interaction.depth=1, estimator="CART", shrinkage="node_based")

HSTreeClassifier <- function(X, y, reg_param=1, max_leaf_nodes=20, interaction.depth=1, estimator="CART", shrinkage="node_based"){
    if (estimator=="CART"){
      fit <- rpart(y~., data = data.frame(X,y), control = rpart.control(maxdepth = max_leaf_nodes)) # we use max depth as an alternative for max leaf nodes
      complexity=fit$frame$complexity
      # Then fit$frame is a dataframe whose rows describe the nodes. The name of the row is the node number. The children of node n are 2n and 2n+1
      frame <- fit$frame
      n <- row.names(frame)
      node <- as.numeric(n)
      val_new = fit$frame$yval
      # keep the root value
      for (i in 1:(max_leaf_nodes+1)){
        left = which(node==2*node[i])
        right = which(node==2*node[i]+1)
        if (shrinkage=="node_based"){
          val_new[left] = (val_new[left]-val_new[i])/(1+reg_param/frame$n[i])+val_new[i]
          val_new[right] = (val_new[right]-val_new[i])/(1+reg_param/frame$n[i])+val_new[i]
        }else if (shrinkage=="leaf_based"){
          val_new[left] = (val_new[left]-val_new[1])/(1+reg_param/frame$n[1])+val_new[1]
          val_new[right] = (val_new[right]-val_new[1])/(1+reg_param/frame$n[1])+val_new[1]
        }else if (shrinkage=="constant"){
          val_new[left] = (val_new[left]-val_new[i])/(1+reg_param)+val_new[i]
          val_new[right] = (val_new[right]-val_new[i])/(1+reg_param)+val_new[i]
        }else{
          stop("please use the right shrinkage scheme")
        }
      }
      fit$frame$yval=val_new  # replace the orginal y value with our HS penelised new value

    }else if (estimator=="RandomForest"){
      fit <- randomForest(X, y[[1]], maxnodes = max_leaf_nodes)
      complexity=fit$frame$complexity
      # Then fit$frame is a dataframe whose rows describe the nodes. The name of the row is the node number. The children of node n are 2n and 2n+1
      frame <- fit$forest
      val_new = frame$nodepred
      # keep the root value
      for (i in 1:ncol(frame$nodestatus)){
        for (j in 1:nrow(frame$nodestatus)){
          left = frame$leftDaughter[j,i]
          right = frame$rightDaughter[j,i]
          if (shrinkage=="node_based"){
            val_new[left,i] = (val_new[left,i]-val_new[j,i])/(1+reg_param)+val_new[j,i]
            val_new[right,i] = (val_new[right,i]-val_new[j,i])/(1+reg_param)+val_new[j,i]
          }else if (shrinkage=="leaf_based"){
            val_new[left] = (val_new[left,i]-val_new[1,i])/(1+reg_param/nrow(X))+val_new[1,i]
            val_new[right] = (val_new[right,i]-val_new[1,i])/(1+reg_param/nrow(X))+val_new[1,i]
          }else if (shrinkage=="constant"){
            val_new[left,i] = (val_new[left,i]-val_new[j,i])/(1+reg_param)+val_new[j,i]
            val_new[right,i] = (val_new[right,i]-val_new[j,i])/(1+reg_param)+val_new[j,i]
          }else{
            stop("please use the right shrinkage scheme")
          }
        }
      }
      fit$forest$nodepred=val_new  # replace the orginal y value with our HS penelised new value
    }else if (estimator=="GradientBoosting"){
      fit <- gbm(y~., data = data.frame(X,y), interaction.depth = interaction.depth)
      # Then fit$frame is a dataframe whose rows describe the nodes. The name of the row is the node number. The children of node n are 2n and 2n+1
      frame <- fit$trees
      yval=rep(NA,length(frame))
      for (i in 1:length(frame)){
        val_new = unlist(frame[[i]][8])
        for (j in 1:length(val_new)){
          left = unlist(frame[[i]][3])[j]
          right = unlist(frame[[i]][4])[j]
          if(left!=right){
            samples = unlist(frame[[i]][7])[j]
            if (shrinkage=="node_based"){
              val_new[left] = (val_new[left]-val_new[j])/(1+reg_param/samples[j])+val_new[j]
              val_new[right] = (val_new[right]-val_new[j])/(1+reg_param/samples[j])+val_new[j]
            }else if (shrinkage=="leaf_based"){
              val_new[left] = (val_new[left]-val_new[1])/(1+reg_param/samples[1])+val_new[1]
              val_new[right] = (val_new[right]-val_new[1])/(1+reg_param/samples[1])+val_new[1]
            }else if (shrinkage=="constant"){
              val_new[left] = (val_new[left]-val_new[j])/(1+reg_param)+val_new[j]
              val_new[right] = (val_new[right]-val_new[j])/(1+reg_param)+val_new[j]
            }else{
              stop("please use the right shrinkage scheme")
            }
          }
        }
        fit$trees[[i]][8]=list(val_new)
        # yval[i]=list(val_new)  store after shrinkage result as a matrix
      }
    }else{
      stop("the estimator is not available right now")
    }
    structure(fit,
              shrinkage=shrinkage,
              regularization="HSTree")
  }




#' HSTree fit for regression
#'
#' modify the decision tree(or each tree in ensemble) structure based on hierarchical shrinkage regularization
#'
#'
#' @param X the design matrix
#' @param y the response vector
#' @param reg_param Higher is more regularization (can be arbitrarily large, should not be < 0)
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


#' @usage HSTreeRegressor(X, y, reg_param=1, max_leaf_nodes=20, interaction.depth=1, estimator="CART", shrinkage="node_based")
#'
HSTreeRegressor <- function(X, y, reg_param=1, max_leaf_nodes=20, interaction.depth=1, estimator="CART", shrinkage="node_based"){
  if (estimator=="CART"){
    fit <- rpart(y~., data = data.frame(X,y), control = rpart.control(maxdepth = max_leaf_nodes)) # we use max depth as an alternative for max leaf nodes
    complexity=fit$frame$complexity
    # Then fit$frame is a dataframe whose rows describe the nodes. The name of the row is the node number. The children of node n are 2n and 2n+1
    frame <- fit$frame
    n <- row.names(frame)
    node <- as.numeric(n)
    val_new = fit$frame$yval
    # keep the root value
    for (i in 1:(max_leaf_nodes+1)){
      left = which(node==2*node[i])
      right = which(node==2*node[i]+1)
      if (shrinkage=="node_based"){
        val_new[left] = (val_new[left]-val_new[i])/(1+reg_param/frame$n[i])+val_new[i]
        val_new[right] = (val_new[right]-val_new[i])/(1+reg_param/frame$n[i])+val_new[i]
      }else if (shrinkage=="leaf_based"){
        val_new[left] = (val_new[left]-val_new[1])/(1+reg_param/frame$n[1])+val_new[1]
        val_new[right] = (val_new[right]-val_new[1])/(1+reg_param/frame$n[1])+val_new[1]
      }else if (shrinkage=="constant"){
        val_new[left] = (val_new[left]-val_new[i])/(1+reg_param)+val_new[i]
        val_new[right] = (val_new[right]-val_new[i])/(1+reg_param)+val_new[i]
      }else{
        stop("please use the right shrinkage scheme")
      }
    }
    fit$frame$yval=val_new  # replace the orginal y value with our HS penelised new value

  }else if (estimator=="RandomForest"){
    fit <- randomForest(X, y[[1]], maxnodes = max_leaf_nodes)
    complexity=fit$frame$complexity
    # Then fit$frame is a dataframe whose rows describe the nodes. The name of the row is the node number. The children of node n are 2n and 2n+1
    frame <- fit$forest
    val_new = frame$nodepred
    # keep the root value
    for (i in 1:ncol(frame$nodestatus)){
      for (j in 1:nrow(frame$nodestatus)){
        left = frame$leftDaughter[j,i]
        right = frame$rightDaughter[j,i]
        if (shrinkage=="node_based"){
          val_new[left,i] = (val_new[left,i]-val_new[j,i])/(1+reg_param)+val_new[j,i]
          val_new[right,i] = (val_new[right,i]-val_new[j,i])/(1+reg_param)+val_new[j,i]
        }else if (shrinkage=="leaf_based"){
          val_new[left] = (val_new[left,i]-val_new[1,i])/(1+reg_param/nrow(X))+val_new[1,i]
          val_new[right] = (val_new[right,i]-val_new[1,i])/(1+reg_param/nrow(X))+val_new[1,i]
        }else if (shrinkage=="constant"){
          val_new[left,i] = (val_new[left,i]-val_new[j,i])/(1+reg_param)+val_new[j,i]
          val_new[right,i] = (val_new[right,i]-val_new[j,i])/(1+reg_param)+val_new[j,i]
        }else{
          stop("please use the right shrinkage scheme")
      }
      }
    }
    fit$forest$nodepred=val_new  # replace the orginal y value with our HS penelised new value
  }else if (estimator=="GradientBoosting"){
    fit <- gbm(y~., data = data.frame(X,y), interaction.depth = interaction.depth)
    # Then fit$frame is a dataframe whose rows describe the nodes. The name of the row is the node number. The children of node n are 2n and 2n+1
    frame <- fit$trees
    yval=rep(NA,length(frame))
    for (i in 1:length(frame)){
      val_new = unlist(frame[[i]][8])
      for (j in 1:length(val_new)){
        left = unlist(frame[[i]][3])[j]
        right = unlist(frame[[i]][4])[j]
        if(left!=right){
        samples = unlist(frame[[i]][7])[j]
        if (shrinkage=="node_based"){
          val_new[left] = (val_new[left]-val_new[j])/(1+reg_param/samples[j])+val_new[j]
          val_new[right] = (val_new[right]-val_new[j])/(1+reg_param/samples[j])+val_new[j]
        }else if (shrinkage=="leaf_based"){
          val_new[left] = (val_new[left]-val_new[1])/(1+reg_param/samples[1])+val_new[1]
          val_new[right] = (val_new[right]-val_new[1])/(1+reg_param/samples[1])+val_new[1]
        }else if (shrinkage=="constant"){
          val_new[left] = (val_new[left]-val_new[j])/(1+reg_param)+val_new[j]
          val_new[right] = (val_new[right]-val_new[j])/(1+reg_param)+val_new[j]
        }else{
          stop("please use the right shrinkage scheme")
         }
        }
      }
      fit$trees[[i]][8]=list(val_new)
      # yval[i]=list(val_new)  store after shrinkage result as a matrix
    }
  }else{
    stop("the estimator is not available right now")
  }
  structure(fit,
            shrinkage=shrinkage,
            regularization="HSTree")
}
