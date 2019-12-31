rm(list = ls()) # clear global environment
graphics.off() # close all graphics
gc()
if(!"pacman" %in% rownames(installed.packages())){
  install.packages(pkgs = "pacman",repos = "http://cran.us.r-project.org")
}
# p_load is equivalent to combining both install.packages() and library()
pacman::p_load(caret,bestglm,DMwR,smotefamily,randomForest,unbalanced,e1071,RSNNS,C50,MASS,ROSE,snow,ranger,parallel,xgboost,gbm,naivebayes,kernlab,pls,Rmpi)


Mainfun<-function(j, data, allfolds,f){
  #library(caret)
  #library(bestglm)
  #library(DMwR)
  #library(smotefamily)
  #library(randomForest)
  #library(unbalanced)
  #library(e1071)
  #library(RSNNS)
  #library(C50)
  #library(earth)  #for earth
  #library(mda)  
  #library(logicFS)  # for logicBag
  #library(fastAdaboost) #for adaboost
  #library(kernlab) #for svmRadial
  #library(sparseLDA)
  #library(deepnet)   #for dnn
  #library(obliqueRF)  #for RRF
  #library(kohonen)  #for xyf self organizing map
  #library (extraTrees) #for extraTrees
  #library (adabag) #for AdaBoost.M1
  #library(rparty) #for ctree
  #library(ranger) #ordinalRF
  #library(ordinalForest) #for ordinalRF
  #library(mboost) #for blackboost
  
  gc()
  if(!"pacman" %in% rownames(installed.packages())){
    install.packages(pkgs = "pacman",repos = "http://cran.us.r-project.org")
  }
  # p_load is equivalent to combining both install.packages() and library()
  pacman::p_load(caret,bestglm,DMwR,smotefamily,randomForest,unbalanced,e1071,RSNNS,C50,MASS,ROSE,snow,ranger,parallel,xgboost,gbm,naivebayes,kernlab,pls,Rmpi)
  
  
  data<- data[which(data$Sex=="Female"),-c(1,2,6,10)]
  data.variables<-data[,c(5:30)] # the categorcal variables
  data.variables[,] <- lapply(data.variables[,], factor) 
  dmy <- dummyVars(" ~ .", data = data.variables)
  trsf <- data.frame(predict(dmy, newdata = data.variables))
  data.continuous<- scale(data[,c(1:4)])
  n.continuous.var<-ncol(data.continuous)
  all.data.sex1<- cbind(data.continuous,trsf,data$X1year)
  colnames(all.data.sex1)[ncol(all.data.sex1)]<- "survival"
  
  mod <- lm(survival ~ ., data=all.data.sex1)
  cooksd <- cooks.distance(mod)
  influential <- data.frame(as.numeric(names(cooksd)[(cooksd > 5*mean(cooksd, na.rm=T))]))
  influential<- influential[complete.cases(influential), ]
  all.data.sex<- all.data.sex1[-c(influential),]
  table(all.data.sex$survival)
  
  #sampling
  X1<-all.data.sex[,-c(ncol(all.data.sex))]
  target1<-all.data.sex[,ncol(all.data.sex)]

  #SMOTE  
  data.smote1<-ubBalance(X= X1, Y=as.factor(target1), type="ubSMOTE", percOver=250,percUnder=150, verbose=TRUE)
  newData<-cbind(data.smote1$X,  as.factor(data.smote1$Y))

  data.smote<-newData[,apply( newData, 2, var, na.rm=TRUE) != 0]
  rownames( data.smote) <- NULL
  for  (i in ((n.continuous.var)+1): (ncol(data.smote)-1)){
    data.smote [,i] <- ifelse(data.smote [,i] >= 0.5, 1, 0)
  }
  
  #cross validation
  all.data.sex<- data.smote
  rownames(all.data.sex) <- NULL
  colnames(all.data.sex)[ncol(all.data.sex)] <- "survival"
  set.seed(123)
  flds <- createFolds(all.data.sex$survival, k = 10, list = TRUE, returnTrain = FALSE)
  f1<- c(unlist((flds[1]),use.names = FALSE))
  f2<- c(unlist((flds[2]),use.names = FALSE))
  f3<- c(unlist((flds[3]),use.names = FALSE))
  f4<- c(unlist((flds[4]),use.names = FALSE))
  f5<- c(unlist((flds[5]),use.names = FALSE))
  f6<- c(unlist((flds[6]),use.names = FALSE))
  f7<- c(unlist((flds[7]),use.names = FALSE))
  f8<- c(unlist((flds[8]),use.names = FALSE))
  f9<- c(unlist((flds[9]),use.names = FALSE))
  f10<- c(unlist((flds[10]),use.names = FALSE))
  
  folds.name<- rep(1:10,times=c(length(f1),length(f2),length(f3),
                                length(f4),length(f5),length(f6),
                                length(f7),length(f8),length(f9),
                                length(f10)))
  
  all.data.sex.nofolds<- rbind(all.data.sex[f1,],all.data.sex[f2,],all.data.sex[f3,],
                               all.data.sex[f4,], all.data.sex[f5,], all.data.sex[f6,],
                               all.data.sex[f7,], all.data.sex[f8,],all.data.sex[f9,],
                               all.data.sex[f10,])
  
  all.data.sex.folds<- cbind(folds.name,all.data.sex.nofolds)
  rownames(all.data.sex.folds) <- NULL
  
  train.index <- which(!(all.data.sex.folds$folds.name %in% allfolds[j]))
  test.index <-which((all.data.sex.folds$folds.name %in% allfolds[j]))
  all.data.sex.folds$survival<- as.factor(all.data.sex.folds$survival)
  test.data<- all.data.sex.folds[test.index,-1] 
  
  # feature selection
  data.smote<-all.data.sex.folds[train.index,-c(1)]
  data.smote<-data.smote[,apply( data.smote, 2, var, na.rm=TRUE) != 0]

  ctrl <- trainControl(method="none")
  mod1 <- caret::train(survival ~ ., method="mlp", data = data.smote)
  imp_names <-row.names(varImp(mod1 , scale=FALSE)[1][[1]])
  imp<- data.frame(varImp(mod1 , scale=FALSE)[1][[1]])
  row.names(imp)<- NULL
  features.notsort<- data.frame(imp_names,imp)
  features<- features.notsort[rev(order(features.notsort[,2])),]
  row.names( features)<- NULL
  sum.target<- sum(features[,2])*0.80
  
  for (i in 1: nrow(features)) {
    sum.imp <- sum(features[,2][1:i])
    if (sum.imp >= sum.target){
      break
    }
  }
  
  BEST<- c(as.character(features[,1][1:i]))
  BEST <- c("survival", BEST)
  Train.0 <- which(data.smote$survival[]==0)
  Train.1 <- which(data.smote$survival[]==1)
  
  m<-30
  accuracy <- rep(NA, m)
  sen <- rep(NA,m)
  spe <- rep(NA, m)
  
  for (k in 1:m){
    set.seed((1994+k))
    index.train <- c(sample(Train.0, length(Train.0), replace=TRUE), sample(Train.1, length(Train.1), replace=TRUE))
    test <- test.data[, BEST]
    train <- data.smote[index.train, BEST]
    not.survival<- length(which(test$survival[]==1))
    survival<- nrow(test)- not.survival
    set.seed((123+k))
    ctrl <- trainControl(method="none")
    mod <- caret::train(survival ~ ., method="mlp", data = train)
    pred.train<- predict(mod, train)
    table(pred.train, train$survival)
    pred <- predict(mod, test)
    comparison <- table(test$survival,pred)
    accuracy[k] <- (comparison[1,1]+comparison[2,2])/nrow(test)
    sen[k] <- comparison[1,1]/ (survival)
    spe[k] <- comparison[2,2]/ (not.survival)
  }
  
  model <- c(j, length(BEST[-c(1)]), round(c( mean(sen), sd(sen), mean(spe), sd(spe),mean(accuracy), sd(accuracy)),4))
  metric <- round(c( mean(sen), sd(sen), mean(spe), sd(spe),mean(accuracy), sd(accuracy)),4)
  best_features1 <-  BEST[-c(1)]
  features.importance <- c(features[1:i,2])
  Optimal_result <- list(model, metric, best_features1, features.importance)
  Optimal_result <- list(model, metric, best_features1)
  return(Optimal_result)
}

setwd("") # set working directory to save the results
data<- data.frame() # read the data
allfolds <- seq(1:10)

cl <- makeCluster(2, type="SOCK") ### number of cores in your computer
time.begin <- proc.time()[3]
ncases <- length(allfolds) 
Result <- parSapply(cl, 1:ncases, Mainfun, data, allfolds)
stopCluster(cl)
time.end <- (proc.time()[3]-time.begin)/60
paste("It took", time.end, "minutes to run the program.")

### Save output values in a dataframe and important features for each case in a list
sen_spe <- data.frame(matrix(0, nrow = ncases, ncol = 8))
colnames(sen_spe) <- c("model","#features","sensitivity","sd(sensitivity)", "specificity", "sd(specificity)","accuracy", "sd(accuracy)") 
best_features <- list()
impor_features <- list()
metric <- list()

for (i in 1:ncases){
  sen_spe[i,] <- as.vector(Result[[(4*i-3)]])
  metric[[i]] <- Result[[(4*i-2)]]
  best_features[[i]] <- Result[[(4*i-1)]]
  impor_features[[i]] <- Result[[(4*i)]]
}

metric1 <- data.frame(plyr::ldply(metric, rbind))
metric2 <- metric1[complete.cases(metric1), ]
mean.model <- apply(metric2, 2, mean) 
sd.model <- apply(metric2, 2, sd) 

all.metrics <- rbind(mean.model,sd.model)
colnames(all.metrics) <- c("sensitivity","sd(sensitivity)","specificity", "sd(specificity)","accuracy", "sd(accuracy)")
model_features <- data.frame(plyr::ldply(best_features, rbind))
importanceof_features <- data.frame(plyr::ldply(impor_features, rbind))
features.all <- data.frame(rbind(importanceof_features,model_features))

model <- "ModelsName"
sampling<- "SMOTE_"
feature.selection <- "Non"
sex <- "Female"

name<-paste(model,sampling,feature.selection, sep="")
write.csv(sen_spe,paste(name,"_","sen_spe.csv", sep=""))
write.csv(model_features,paste(name,"_","_features.csv"))
write.csv(importanceof_features,paste(name,"_","_impfeatures.csv"))
write.csv(all.metrics,paste(name,"_","all.metrics.csv"))









