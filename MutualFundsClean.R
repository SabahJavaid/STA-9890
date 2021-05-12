setwd("~/Desktop/Grad School/Stat 9890/Project")

install.packages('ISLR')
install.packages("glmnet")
install.packages("randomForest")
install.packages("data.table")
install.packages("caret")
install.packages("dplyr")
install.packages("Matrix")
install.packages("lattice")
install.packages("ggplot2")

rm(list=ls())
cat("\014")

library(ISLR)
library(glmnet)
library(randomForest)
library(data.table)
library(caret)
library(dplyr)
library(Matrix)
library(ggplot2)
library (readr)

MutualFundsCleaned <- read_csv("MutualFundsCleaned.csv")
View(MutualFundsCleaned)

#Cleaning Data
sum(is.na(MutualFundsCleaned))


library(gridExtra)
MutualFundsCleaned$fund_symbol=as.factor(MutualFundsCleaned$fund_symbol)
MutualFundsCleaned$investment_type=as.factor(MutualFundsCleaned$investment_type)
str(MutualFundsCleaned)
dim(MutualFundsCleaned)
summary(MutualFundsCleaned)

# standardizing
MutualFundsCleaned%>%select(-fund_return_2019, -fund_symbol, -investment_type)%>%
  mutate_all(.funs = function(x){
  x/sqrt(mean((x-mean(x))^2))})


######## Part 3A ########
#### Randomly split the dataset into two mutually exclusive datasets Dtest and Dtrain
#### with size ntest and ntrain such that ntrain + ntest = n.

### split the dataset into training and test set with the split ratio.

# creates a value for dividing the data into train and test. In this case the value is defined as 80% of 
# the number of rows in the dataset

# creating X matrix
x=data.matrix(MutualFundsCleaned%>%select(-fund_return_2019))
dim(x)

#distribution check
hist(MutualFundsCleaned$fund_return_2019)
y=MutualFundsCleaned$fund_return_2019

#identifying parameters
n=nrow(x)
p=ncol(x)
n.train=floor(0.8*n)
n.test=n-n.train

# set seed to ensure you always have same random numbers generated
######## Part 3B ########
### Use Dlearn to fit lasso, elastic-net alpha = 0.5, ridge, and random forest.
####### Part 3C ########
### Tune the lambdas using 10-fold CV.

K=100
Rsq.test.rid = rep(0,K)
Rsq.test.las = rep(0,K)
Rsq.test.net = rep(0,K)
Rsq.test.rf = rep(0,K)

Rsq.train.rid = rep(0,K)
Rsq.train.las = rep(0,K)
Rsq.train.net = rep(0,K)
Rsq.train.rf = rep(0,K)


for (k in c(1:K)) {
  shuffled_indexes=sample(n)
  train            =  shuffled_indexes[1:n.train]
  test             =  shuffled_indexes[(1+n.train):n]
  x_train          =  x[train, ]
  y_train          =  y[train]
  x_test           =  x[test, ]
  y_test           =  y[test]
  
  #fit Ridge and calculate and record train and test R-squared and the Residuals  
  set.seed(1)
  model_ridge_k = cv.glmnet(x_train, y_train, 
                            alpha=0, nfolds = 10)
  model_ridge = glmnet(x_train, y_train, 
                       alpha=0, lambda = model_ridge_k$lambda.min)
  y_hat_test = predict(model_ridge, newx=x_test, type="response")
  y_hat_train = predict(model_ridge, newx=x_train, type="response")
  Rsq.test.rid[k] = 1-mean((y_test - y_hat_test)^2)/mean((y_test - mean(y_test))^2)
  Rsq.train.rid[k] = 1-mean((y_train - y_hat_train)^2)/mean((y_train - mean(y_train))^2)
  
  resid_test_rid = y_test-y_hat_test 
  resid_test_rid =as.vector(resid_test_rid)
  resid_train_rid = y_train-y_hat_train
  resid_train_rid =as.vector(resid_train_rid)
  
  
  # fit Lasso and calculate and record train and test R-squared and the Residuals
  set.seed(1)
  model_lasso_k = cv.glmnet(x_train, y_train, 
                            alpha=1, nfolds = 10)
  model_lasso = glmnet(x_train, y_train, 
                       alpha=1, lambda = model_lasso_k$lambda.min)
  y_hat_test = predict(model_lasso, newx=x_test)
  y_hat_train = predict(model_lasso, newx=x_train)
  Rsq.test.las[k] = 1-mean((y_test - y_hat_test)^2)/mean((y_test - mean(y_test))^2)
  Rsq.train.las[k] = 1-mean((y_train - y_hat_train)^2)/mean((y_train - mean(y_train))^2)
  
  resid_test_las = y_test-y_hat_test 
  resid_test_las =as.vector(resid_test_las)
  resid_train_las = y_train-y_hat_train
  resid_train_las =as.vector(resid_train_las)
  
  
  # fit Elastic Net and calculate and record train and test R-squared and the Residuals
  set.seed(1)
  model_net_k = cv.glmnet(x_train, y_train, 
                          alpha=.5, nfolds = 10)
  model_net = glmnet(x_train, y_train, 
                     alpha=.5, lambda = model_net_k$lambda.min)
  y_hat_test = predict(model_net, newx=x_test)
  y_hat_train = predict(model_net, newx=x_train)
  Rsq.test.net[k] = 1-mean((y_test - y_hat_test)^2)/mean((y_test - mean(y_test))^2)
  Rsq.train.net[k] = 1-mean((y_train - y_hat_train)^2)/mean((y_train - mean(y_train))^2)
  
  resid_test_net = y_test-y_hat_test 
  resid_test_net =as.vector(resid_test_net)
  resid_train_net = y_train-y_hat_train
  resid_train_net =as.vector(resid_train_net)
  
  
  # fit Random Forest and calculate and record train and test R-squared and the Residuals
  set.seed(1)
  model_rf = randomForest(x_train, y_train, mtry=p/3,importance = TRUE)
  length(y_test)
  length(y_hat_test)
  
  y_hat_test = predict(model_rf, x_test)
  y_hat_train = predict(model_rf, x_train)
  Rsq.test.rf[k] = 1-mean((y_test - y_hat_test)^2)/mean((y_test - mean(y_test))^2)
  Rsq.train.rf[k] = 1-mean((y_train - y_hat_train)^2)/mean((y_train - mean(y_train))^2)
  
  resid_test_rf = y_test-y_hat_test 
  resid_test_rf =as.vector(resid_test_rf)
  resid_train_rf = y_train-y_hat_train
  resid_train_rf =as.vector(resid_train_rf)
}


####Part 4B####
##Show the side-by-side boxplots of R2 test;R2 train. We want to see two panels. One for training,
##and the other for testing.
par(mfrow=c(1,2))
boxplot(Rsq.train.rf, Rsq.train.rid, Rsq.train.net, Rsq.train.las,
        main = "R-squared, train subset",
        at = c(1,4,7,10),
        names = c("RF", "Ridge", "E-net", "Lasso"),
        col = c("green","yellow","orange", "blue"),
        ylim=c(0.6, 1))

boxplot(Rsq.test.rf, Rsq.test.rid, Rsq.test.net, Rsq.test.las,
        main = "R-squared, test subset",
        at = c(1,4,7,10),
        names = c("RF", "Ridge", "E-net", "Lasso"),
        col = c("green","yellow","orange", "blue"),
        ylim=c(0.6,1))

####Part 4C####
##Record and present the time it takes to cross-validate ridge/lasso/elastic-net regression.
#lam.las = c(seq(1e-3,0.1,length=100),seq(0.12,2.5,length=100)) 
lam.rid=10^seq(-2, 3, length.out = 100)
lam.net = c(seq(1e-2,0.1,length=100),seq(0.12,2.5,length=100))
lam.las=lam.net
time_ridge = system.time(cv_rid <- cv.glmnet(x_train, y_train, alpha = 0, 
                                             lambda=lam.rid, nfolds = 10))   #ridge
time_elastic_net = system.time(cv_net <- cv.glmnet(x_train, y_train, alpha = 0.5, 
                                             lambda=lam.net, nfolds = 10))  #net
time_lasso = system.time(cv_las <- cv.glmnet(x_train, y_train, alpha = 1,
                                             lambda=lam.las, nfolds = 10))   #las
rbind(time_ridge, time_elastic_net, time_lasso)

##For one on the 100 samples, create 10-fold CV curves for lasso, elastic-net alpha= 0:5, ridge. 
par(mfrow=c(1,3))
plot(cv_rid)
title("Ridge", line = 2.5)
plot(cv_net)
title("Elastic Net", line = 2.5)
plot(cv_las)
title("Lasso", line = 2.5)

####Part 4D####
##(d) For one on the 100 samples, show the side-by-side boxplots of train and test residuals

par(mfrow=c(1,2))
boxplot(resid_train_rf, resid_train_rid, resid_train_net, resid_train_las,
        main = "Train Residuals",
        at = c(1,4,7,10),
        names = c("RF", "Ridge", "E-net", "Lasso"),
        col=c("green", "yellow", "orange", "blue"),
        ylim=c(-15,15))

boxplot(resid_test_rf, resid_test_rid, resid_test_net, resid_test_las,
        main = "Test Residuals",
        at = c(1,4,7,10),
        names = c("RF", "Ridge", "E-net", "Lasso"),
        col=c("green", "yellow", "orange", "blue"),
        ylim=c(-15,15))


#####Part 5-Bullet 1######
#For all the data Use the 10-fold cross validation, fit ridge, lasso and elastic-net. Also fit random forest.
##.......

dim(x)
length(y)

# Ridge
set.seed(1)
ridge_cv_full = cv.glmnet(x, y, alpha=0, nfolds = 10)
model_ridge_full = glmnet(x, y, alpha=0, lambda = ridge_cv_full$lambda.min)
ridge_cv_full$lambda.min
model_ridge_full
y_hat_cv_full_rid = predict(model_ridge_full, x, type="response")

# Lasso
set.seed(1)
lasso_cv_full = cv.glmnet(x, y, alpha=1, nfolds = 10)

model_lasso_full = glmnet(x, y, 
                          alpha=1, lambda = lasso_cv_full$lambda.min)
lasso_cv_full$lambda.min
model_lasso_full
y_hat_cv_full_las = predict(model_lasso_full, x, type="response")

# Elastic Net
set.seed(1)
elnet_cv_full = cv.glmnet(x, y, alpha=.5, nfolds = 10)

model_elnet_full = glmnet(x, y,alpha=.5, lambda = elnet_cv_full$lambda.min)
elnet_cv_full$lambda.min
model_elnet_full
y_hat_cv_full_elnet = predict(model_elnet_full, x, type="response")


# Random Forest
set.seed(1)
model_rf_full = randomForest(x,y, mtry=p/3,importance = TRUE)
model_rf_full

y_hat_rf_full = predict(model_rf_full, x, type="response")


#####Part 5-Bullet 2 ######
#Also record the time it takes to fit a single ridge/lasso/elastic-net regression (including the time needed to perform cross-validation parameter tuning), and random forrest. Create a table 4 ? 2 table, the 4 rows corresponding to the 4
#methods, and the two columns for test R2 and time. Specifically, the first column
#should show a 90% test R2
#interval based on the 100 samples, and the second
#column the time it takes to fit the model on all the data (as described in the
#sentences above). Is there a trade-off between the time it takes to train a model
#and it's predictive performance?

# time needed to perform CV
time_ridge_full_cv = system.time(ridge_cv<-cv.glmnet(x, y, 
                                                     alpha=0, nfolds = 10))   #ridge
time_elastic_net_full_cv = system.time(elnet_cv<-cv.glmnet(x, y, 
                                                           alpha=.5, nfolds = 10))  #elnet
time_lasso_full_cv = system.time(lasso_cv<-cv.glmnet(x, y, 
                                                     alpha=1, nfolds = 10))   #lasso
rbind(time_ridge_full_cv, time_elastic_net_full_cv, time_lasso_full_cv)

#time needed to fit a model
time_ridge_full_fit = system.time(ridge_full_fit<-glmnet(x, y, 
                                                         alpha=0, lambda = ridge_cv_full$lambda.min))   #ridge
time_elastic_net_full_fit = system.time(elnet_full_fit<-glmnet(x, y, 
                                                               alpha=.5, lambda = elnet_cv_full$lambda.min))  #elnet
time_lasso_full_fit = system.time(lasso_full_fit<-glmnet(x, y, 
                                                         alpha=1, lambda = lasso_cv_full$lambda.min))   #lasso
time_rf_full_fit=system.time(rf_full_fit<-randomForest(x, y, mtry=p/3,
                                                       importance = TRUE))     #random forest
rbind(time_ridge_full_fit, time_elastic_net_full_fit, time_lasso_full_fit, time_rf_full_fit)


#obtaining the 90% confidence interval for test Rsqr on the 100 samples
CI.rid=quantile(Rsq.test.rid, c(0.1,0.90))
CI.las=quantile(Rsq.test.las, c(0.1,0.90))
CI.elnet=quantile(Rsq.test.net, c(0.1,0.90))
CI.rf=quantile(Rsq.test.rf, c(0.1,0.90))
# Result for CI
CI_90.testRsq=rbind(CI.rid,CI.las,CI.elnet,CI.rf)
CI_90.testRsq


###5.3##
#Bar plots of the estimated coefficients, importance of the parameters
#obtaining estimated coefficients for each model

#Coefficients Ridge
beta.coef_rid=data.frame(c(1:p), as.vector(model_ridge_full$beta))
colnames(beta.coef_rid)=c("feature", "value")

#Coefficients Lasso
beta.coef_las=data.frame(c(1:p), as.vector(model_lasso_full$beta))
colnames(beta.coef_las)=c("feature", "value")

#Coefficients El-net
beta.coef_elnet=data.frame(c(1:p), as.vector(model_elnet_full$beta))
colnames(beta.coef_elnet)=c("feature", "value")

#Coefficients Random Forest
beta.coef_rf=data.frame(c(1:p), as.vector(model_rf_full$importance[1:p]))
colnames(beta.coef_rf)=c("feature", "value")

#changing the order of factor levels by specifying the order explicitly
# specifically use elastic-net estimated coefficients to create an order based on descending order
# use this order to present plots of the estimated coefficients of all 4 models
# Ridge
beta.coef_rid$feature=factor(beta.coef_rid$feature, levels = beta.coef_elnet$feature
                             [order(beta.coef_elnet$value, decreasing = TRUE)])
beta.coef_elnet$feature=factor(beta.coef_elnet$feature, levels = beta.coef_elnet$feature
                               [order(beta.coef_elnet$value, decreasing = TRUE)])
beta.coef_las$feature=factor(beta.coef_las$feature, levels = beta.coef_elnet$feature
                             [order(beta.coef_elnet$value, decreasing = TRUE)])
beta.coef_rf$feature=factor(beta.coef_rf$feature,levels = beta.coef_elnet$feature
                            [order(beta.coef_elnet$value, decreasing = TRUE)])

#Coefficients plots
RidPlot=ggplot(beta.coef_rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Ridge))+
  ylim(-2,6)

# Lasso Plot
LasPlot=ggplot(beta.coef_las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="blue", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Lasso))+
  ylim(-2,6)

# Elastic-Net Plot
ElNetPlot=ggplot(beta.coef_elnet, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="yellow", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Elastic-Net))+
  ylim(-2,6)

# Random Forrest
RFPlot=ggplot(beta.coef_rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="green", colour="black")    +
  labs(x = element_blank(), y = "Importance", title = expression(RandomForest))

library(gridExtra)
Coef.Plot=grid.arrange(RidPlot, LasPlot, ElNetPlot, RFPlot, nrow = 4)
importance(model_rf_full)

