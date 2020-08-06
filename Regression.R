rm(list = ls()) # clear global environment
graphics.off() # close all graphics
gc()
if(!"pacman" %in% rownames(installed.packages())){
  install.packages(pkgs = "pacman",repos = "http://cran.us.r-project.org")
}
# p_load is equivalent to combining both install.packages() and library()
pacman::p_load(pROC,dplyr,imbalance,caret,bestglm,DMwR,smotefamily,randomForest,unbalanced,e1071,RSNNS,C50,MASS,ROSE,snow,ranger,parallel,xgboost,gbm,naivebayes,kernlab,pls)


#set working directory
setwd("")
data<- data.frame(read.csv())
data<- data[which(data$Sex=="Female"),]
data<- data[,-c(1,2,3,10,11,22,24)]  
data<- cbind(data[1:32],data$X3years)    #3year

data.variables<-data[,c(6:32)] # the categorcal variables

#One-hot encoding
data.variables[,] <- lapply(data.variables, factor) 
dummy<- data.frame(predict(caret::dummyVars(" ~ .", data = data.variables, fullRank=T),
                           newdata = data.variables))
all.data.sex1<-cbind((data[,c(1:6)]),dummy,data$`data$X3years`)
colnames(all.data.sex1)[ncol(all.data.sex1)]<- "survival"
all.data.sex1<- all.data.sex1[, -nearZeroVar(all.data.sex1[,-ncol(all.data.sex1)])]


pp<- preProcess(all.data.sex1[, c(2:5)], 
                method = c("center", "scale", "YeoJohnson"))
transformed <- predict(pp, newdata = all.data.sex1[, c(2:5)])
all.data.sex1<- cbind(all.data.sex1[,1],transformed,all.data.sex1[,-c(1:5)] )
colnames(all.data.sex1)[ncol(all.data.sex1)]<- "survival"
colnames(all.data.sex1)[1]<- "survivalm"


mod <- lm(survival ~ ., data=all.data.sex1[,-1])
cooksd <- cooks.distance(mod)
sample_size<- nrow(all.data.sex1)
influential <- as.numeric(names(cooksd)[(cooksd > (4/sample_size))])
all.data.sex<- all.data.sex1[-c(influential),]


#Training and testing creation
set.seed(723)
trainIndex<- createDataPartition(all.data.sex$survival, p=0.7, list=F)
pretrain<- all.data.sex[trainIndex,]
pretest<- all.data.sex[-c(trainIndex),]
pretrain<-pretrain[,apply(pretrain, 2, var, na.rm=TRUE) != 0]

regtrain.hot<- pretrain[which(pretrain$survival==0),-ncol(pretrain)]
regtest.hot<- pretest[,]
pretrain<-pretrain[,-1]


#Preparing data for re-sampling
X1<-pretrain[,-c(ncol(pretrain))]
target1<-pretrain[,ncol(pretrain)]


#ADASYN re-sampling
set.seed(333)
library(imbalance)
pretrain$survival<- as.factor(pretrain$survival)
newData<- imbalance::oversample(
  pretrain,
  ratio = 1,
  method = "ADASYN",
  filtering = FALSE,
  classAttr = "survival"
)

data.smote<-newData
rownames( data.smote) <- NULL
for  (i in (5: (ncol(data.smote)-1))){
  data.smote [,i] <- ifelse(data.smote [,i] >= 0.5, 1, 0)
}

all.data.sex<- data.smote
rownames(all.data.sex) <- NULL
colnames(all.data.sex)[ncol(all.data.sex)] <- "survival"


#LASSO feature selection
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


#Modeling

#Cross validation
ctrl1 <- trainControl(method = "cv", number = 5, returnResamp = "all",savePredictions = TRUE, 
                      classProbs = TRUE, summaryFunction=twoClassSummary,
                      verboseIter = TRUE, allowParallel = TRUE)

set.seed((195))
Train.0 <- which(all.data.sex$survival[]==0)
Train.1 <- which(all.data.sex$survival[]==1)
index.train <- c(sample(Train.0, length(Train.0), replace=TRUE), sample(Train.1, length(Train.1), replace=TRUE))
test <-  pretest[, BEST]
test$survival<- as.factor(test$survival)
train <- all.data.sex[index.train, BEST]
train$survival<- as.factor(train$survival)
levels(train$survival) <- c("decease","survival")
levels(test$survival) <- c("decease","survival")

#GLM
mod <- caret::train(survival ~ ., method="glm", data = train,trControl= ctrl1,metric = 'ROC')
pred <- predict(mod, test)

pre.reg.test.hot<- cbind(regtest.hot[,c(1,ncol(regtest.hot))],pred,regtest.hot[,-c(1,ncol(regtest.hot))])
pre.reg.test.hot<- pre.reg.test.hot[which(pre.reg.test.hot$survival==1),]
pre.reg.test.hot<- pre.reg.test.hot[which(pre.reg.test.hot$pred=="decease"),-c(2,3)]


#Rgression
time.begin <- proc.time()[3]
if(!"pacman" %in% rownames(installed.packages())){
  install.packages(pkgs = "pacman",repos = "http://cran.us.r-project.org")
}
# p_load is equivalent to combining both install.packages() and library()
pacman::p_load(caret,bestglm,DMwR,smotefamily,randomForest,unbalanced,e1071,RSNNS,C50,MASS,ROSE,snow,ranger,parallel,xgboost,gbm,naivebayes,kernlab,pls,Rmpi,glmnet,boot)

reg.test<- pre.reg.test.hot
regtrain<- regtrain.hot



#Select the feature selection method

#LASSO 
set.seed(123)
lambda_seq <- 10^seq(2, -2, by = -.05)
x_vars <- model.matrix(survivalm~. , regtrain)[,-1]
y_var <- regtrain$survivalm
cv_output <- cv.glmnet(x_vars[,], y_var[], 
                      alpha = 1, lambda = lambda_seq)
best_lam <- cv_output$lambda.min
lasso_best <- glmnet(x_vars[,], y_var[], alpha = 1, lambda = best_lam, intercept = FALSE)
nvariables<- data.frame(lasso_best$df)
deviation<- data.frame(lasso_best$dev.ratio)
lambda<-data.frame(lasso_best$lambda)
c<-coef(lasso_best,s=best_lam,exact=FALSE)
inds<-which(c!=0)
BEST<-row.names(c)[inds]
BEST <- c("survivalm", BEST)
features<-as.matrix(c)
features<- data.frame(features)
features <- tibble::rownames_to_column(features, "VALUE")
features<-features[features$X1!=0,]
features$X1<- round(features$X1,4)
features<- features[rev(order(abs(features$X1))),]


#Random Forest
#set.seed(444)
#fit_rf=randomForest(survivalm~., data=regtrain,importance=T)
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
#BEST <- c("survivalm", BEST)



#Modeling

#Cross validation
ctrl1 <- trainControl(method = "cv", number = 5, returnResamp = "all")

m<-30
std<-rep(NA,m)
RMSE.t <- rep(NA,m)
MAE.t <- rep(NA, m)

for (k in 1:m){
  set.seed((1994+k))
  train <- regtrain[, BEST]
  test <- reg.test[, BEST]
  #Bootstrapping
  train.s<- train[sample(nrow(train), size = nrow(train), replace = TRUE),]
  train.s$survivalm<- as.numeric(train.s$survivalm)
  
  #GLM
  model <- caret::train(survivalm ~ ., data = train.s, "glm",trControl= ctrl1)
  
  #XGB
  #model <- caret::train(survivalm ~ ., data = train.s, "xgbTree",trControl= ctrl1)
  
  #MLP
  #mlp_grid = expand.grid(layer1 = 10,
  #                       layer2 = 10,
  #                       layer3 = 10)
  
  #set.seed((1994+k))
  #model = caret::train(survivalm ~ ., data = train,
  #                     method = "mlpML", 
  #                     preProc =  c('center', 'scale', 'knnImpute', 'pca'),
  #                     trControl = ctrl1,
  #                     tuneGrid = mlp_grid)

  pred <- predict(model, test)
  results<- postResample(pred =  pred, obs = test$survivalm)
  std[k]<- sd(pred)
  RMSE.t[k]<- results[1]
  MAE.t[k]<-results[3]
  
  
  model <- data.frame(matrix(0, nrow = 1, ncol = 7))
  colnames(model) <- c("#features","SD", "SD(STD)", "RSME", "SD(RSME)","MAE", "SD(MAE)")  
  model[1,c(1:7)] <- c(length(BEST[-c(1)]), round(c( mean(std), sd(std), mean(RMSE.t), sd(RMSE.t), mean(MAE.t), sd(MAE.t)),4))
  features <-  data.frame(features)
  prediction.t<- data.frame(pred,test$survivalm)
}

time.end <- (proc.time()[3]-time.begin)/60
paste("It took", time.end, "minutes to run the program.")


modelname <- "insert model's name"
feature.selection <- "insert feature selection method"
name<-paste(modelname,feature.selection, sep="")

#Set working directory
#setwd("")
write.csv(model,paste(name,"_","model.csv", sep=""))
write.csv(features,paste(name,"_","features.csv"))
write.csv(prediction.t,paste(name,"_","prediction.csv"))
