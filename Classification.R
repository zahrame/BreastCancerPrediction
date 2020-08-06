rm(list = ls()) # clear global environment
graphics.off() # close all graphics
gc()
if(!"pacman" %in% rownames(installed.packages())){
  install.packages(pkgs = "pacman",repos = "http://cran.us.r-project.org")
}
# p_load is equivalent to combining both install.packages() and library()
pacman::p_load(pROC,glmnet,dplyr,imbalance,caret,bestglm,DMwR,smotefamily,randomForest,unbalanced,e1071,RSNNS,C50,MASS,ROSE,snow,ranger,parallel,xgboost,gbm,naivebayes,kernlab,pls)

time.begin <- proc.time()[3]

#set working directory
setwd("")
data<- data.frame(read.csv())
data<- data[which(data$Sex=="Female"),]
data<- data[,-c(1,2,3,10,11,22,24)]  

data<- cbind(data[1:32],data$X6m)        #6m
#data<- cbind(data[1:32],data$X1year)     #1year
#data<- cbind(data[1:32],data$X1.5year)   #1.5year
#data<- cbind(data[1:32],data$X2year)      #2year
#data<- cbind(data[1:32],data$X2.5year)   #2.5year
#data<- cbind(data[1:32],data$X3years)    #3year


data.variables<-data[,c(6:32)] # the categorcal variables


#Select the encoding method

#One-hot encoding
data.variables[,] <- lapply(data.variables, factor) 
dummy<- data.frame(predict(caret::dummyVars(" ~ .", data = data.variables, fullRank=T),
                           newdata = data.variables))
all.data.sex1<-cbind((data[,c(1:5)]),dummy,data$`data$X6m`)
#all.data.sex1<-cbind((data[,c(1:5)]),dummy,data$`data$X1year`)
#all.data.sex1<-cbind((data[,c(1:5)]),dummy,data$`data$X1.5year`)
#all.data.sex1<-cbind((data[,c(1:5)]),dummy,data$`data$X2year`)
#all.data.sex1<-cbind((data[,c(1:5)]),dummy,data$`data$X2.5year`)
#all.data.sex1<-cbind((data[,c(1:5)]),dummy,data$`data$X3years`)
colnames(all.data.sex1)[ncol(all.data.sex1)]<- "survival"



#Integer encoding
#data.variables<-sapply(data.variables,unclass)    # data.frame of all categorical variables now displayed as numeric
#all.data.sex1<-cbind((data[,c(1:5)]),data.variables,data$`data$X6m`)
#all.data.sex1<-cbind((data[,c(1:5)]),data.variables,data$`data$X1year`)
#all.data.sex1<-cbind((data[,c(1:5)]),data.variables,data$`data$X1.5year`)
#all.data.sex1<-cbind((data[,c(1:5)]),data.variables,data$`data$X2year`)
#all.data.sex1<-cbind((data[,c(1:5)]),data.variables,data$`data$X2.5year`)
#all.data.sex1<-cbind((data[,c(1:5)]),data.variables,data$`data$X3year`)
#colnames(all.data.sex1)[ncol(all.data.sex1)]<- "survival"


all.data.sex1<- all.data.sex1[, -nearZeroVar(all.data.sex1[,-ncol(all.data.sex1)])]


pp<- preProcess(all.data.sex1[, c(1:4)], 
                method = c("center", "scale", "YeoJohnson"))
transformed <- predict(pp, newdata = all.data.sex1[, c(1:4)])
all.data.sex1<- cbind(transformed,all.data.sex1[,-c(1:4)] )
colnames(all.data.sex1)[ncol(all.data.sex1)]<- "survival"


mod <- lm(survival ~ ., data=all.data.sex1)
cooksd <- cooks.distance(mod)
sample_size<- nrow(all.data.sex1)
influential <- as.numeric(names(cooksd)[(cooksd > (4/sample_size))])
all.data.sex<- all.data.sex1[-c(influential),]


#Training and testing creation
set.seed(723)
trainIndex<- createDataPartition(all.data.sex$survival, p=0.7, list=F)
pretrain<- all.data.sex[trainIndex, ]
pretest<- all.data.sex[-c(trainIndex),]
pretrain<-pretrain[,apply(pretrain, 2, var, na.rm=TRUE) != 0]


#Prepare data for re-sampling
X1<-pretrain[,-c(ncol(pretrain))]
target1<-pretrain[,ncol(pretrain)]



#Select the resampling method

#SMOTE
#set.seed(333)
#library(imbalance)
#pretrain$survival<- as.factor(pretrain$survival)
#newData<- imbalance::oversample(
#  pretrain,
#  ratio = "insert ratio like 0.7 or 1",
#  method = "SMOTE",
#  filtering = FALSE,
#  classAttr = "survival"
#)



#Adaptive Synthetic 
library(imbalance)
pretrain$survival<- as.factor(pretrain$survival)
newData<- imbalance::oversample(
  pretrain,
  ratio = "insert ratio like 0.7 or 1",
  method = "ADASYN",
  filtering = FALSE,
  classAttr = "survival"
)


#RSLS
#set.seed(333)
#library(imbalance)
#pretrain$survival<- as.factor(pretrain$survival)
#newData<- imbalance::oversample(
#  pretrain,
#  ratio = "insert ratio like 0.7 or 1",
#  method = "RSLSMOTE",
#  filtering = FALSE,
#  classAttr = "survival"
#)



data.smote<-newData
rownames( data.smote) <- NULL
for  (i in (5: (ncol(data.smote)-1))){
  data.smote [,i] <- ifelse(data.smote [,i] >= 0.5, 1, 0)
}

all.data.sex<- data.smote
rownames(all.data.sex) <- NULL
colnames(all.data.sex)[ncol(all.data.sex)] <- "survival"



#Select the feature selection method

#LASSO
set.seed(123)
lambda_seq <- 10^seq(2, -2, by = -.05)
x_vars <- model.matrix(survival~. , all.data.sex)[,-1]
y_var <- as.numeric(all.data.sex$survival)-1
cv_output <- cv.glmnet(x_vars[,], y_var[], 
                       alpha = 1, type.measure="mse", nfolds =5,lambda = lambda_seq)
best_lam <- cv_output$lambda.min
lasso_best <- glmnet(x_vars[,], y_var[], alpha = 1,
                     lambda = best_lam, intercept = FALSE)
nvariables<- data.frame(lasso_best$df)
deviation<- data.frame(lasso_best$dev.ratio)
lambda<-data.frame(lasso_best$lambda)
c<-coef(lasso_best,s=best_lam,exact=TRUE)
inds<-which(c!=0)
BEST<-row.names(c)[inds]
BEST <- c("survival", BEST)

features<-as.matrix(c)
features<- data.frame(features)
features <- tibble::rownames_to_column(features, "VALUE")
features<-features[features$X1!=0,]
features$X1<- round(features$X1,4)




#Random Forest
#set.seed(444)
#all.data.sex$survival<- as.factor(all.data.sex$survival)
#library(randomForest)
#fit_rf=randomForest(survival~., data=all.data.sex)
# Create an importance based on mean decreasing gini
# compare the feature importance with varImp() function
#imp_names <- row.names(varImp(fit_rf, scale=FALSE))
#imp<- data.frame(varImp(fit_rf, scale=FALSE))[,1]
#features.notsort<- data.frame(imp_names,imp)
#features<- features.notsort[rev(order(features.notsort$imp)),]
#sum.target<- sum(features$imp)*0.75
#for (i in 1: nrow(features)) {
#  sum.imp <- sum(features$imp[1:i])
#  if (sum.imp >= sum.target){
#    break
#  }
#}
#BEST<- c(as.character(features[,1][1:i]))
#BEST <- c("survival", BEST)



#Modeling

#Cross validation
ctrl1 <- trainControl(method = "cv", number = 5, returnResamp = "all",savePredictions = TRUE, 
                      classProbs = TRUE, summaryFunction=twoClassSummary,
                      verboseIter = TRUE, allowParallel = TRUE)

m<-30
accuracy <- rep(NA, m)
sen.recall <- rep(NA,m)
spe <- rep(NA, m)
gm <- rep(NA, m)
roc<- rep(NA, m)
model<- data.frame(matrix(0, nrow = m,ncol=6 ))
colnames(model) <- c("#features","sensitivity", "specificity", "accuracy", "gm", "roc") 
Train.0 <- which(all.data.sex$survival[]==0)
Train.1 <- which(all.data.sex$survival[]==1)

for (k in 1:m){
  set.seed((194+k))
  
  #Bootstrapping
  index.train <- c(sample(Train.0, length(Train.0), replace=TRUE), sample(Train.1, length(Train.1), replace=TRUE))
  test <-  pretest[, BEST]
  test$survival<- as.factor(test$survival)
  train <- all.data.sex[index.train, BEST]
  train$survival<- as.factor(train$survival)
  
  not.survival<- length(which(test$survival[]==1))
  survival<- nrow(test)- not.survival
  levels(train$survival) <- c("survival","decease")
  levels(test$survival) <- c("survival", "decease")
  
  #GLM
  mod <- caret::train(survival ~ ., method="glm", data = train,trControl= ctrl1,metric = 'ROC')
  
  
  #XGB
  #mod <- caret::train(survival ~ ., method="xgbTree", data = train,trControl= ctrl1,tuneGrid = xgbGrid,metric = 'ROC')
  
  
  #MLP
  #mlp_grid = expand.grid(layer1 = 10,
  #                       layer2 = 10,
  #                       layer3 = 10)
  #set.seed((156+k))
  #mod = caret::train(survival ~ ., data = train,
  #                   method = "mlpML", 
  #                   preProc =  c('center', 'scale', 'knnImpute', 'pca'),
  #                   trControl = ctrl1,
  #                   tuneGrid = mlp_grid)
  

  
  roc0 <- roc(test$survival, 
              predict(mod, test, type = "prob")[,1], 
              levels = rev(levels(test$survival)))
  pred <- predict(mod, test)
  comparison <- table(test$survival,pred)
  accuracy[k] <- (comparison[1,1]+comparison[2,2])/nrow(test)
  sen[k] <- comparison[2,2]/ (not.survival)
  spe[k] <- comparison[1,1]/ (survival)
  gm [k] <- sqrt(sen.recall[k]*spe[k])
  roc[k]<- roc0$auc
  model[k,] <- c(length(BEST[-c(1)]), round(c(sen[k], spe[k], accuracy[k], gm[k], roc[k]),4))
}

metric <- round(c( length(BEST[-c(1)]), mean(sen), sd(sen), mean(spe), sd(spe),mean(accuracy), sd(accuracy), 
                  mean(gm), sd(gm),mean(roc), sd(roc)),4)

best_features<-  BEST[-c(1)]
best_features<-  features
time.end <- (proc.time()[3]-time.begin)/60
paste("It took", time.end, "minutes to run the program.")


### Save output values in a dataframe and important features
sen_spe <- data.frame(matrix(0, nrow = 1, ncol = 11))
colnames(sen_spe) <- c("#features", "sensitivity","sd(sensitivity)", "specificity", "sd(specificity)",
                       "accuracy", "sd(accuracy)","gm", "sd(gm)","roc","sd(roc)") 

sen_spe[1,]<-c(metric)
modelname <- "inser model's name"
sampling<- "insert re-sampling method"
feature.selection <- "inser feature selecion"
year <- "insert time-point"
name<-paste(year,modelname,sampling,feature.selection, sep="")

#Set working directory
#setwd("")
write.csv(sen_spe,paste(name,"_","sen_spe.csv", sep=""))
write.csv(best_features,paste(name,"_","features.csv"))
write.csv(model,paste(name,"_","model.csv"))
