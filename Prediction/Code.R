############################################
#    *** HOW TO READ THE SCRIPT ***        #
#
# Each step has:                           #
#                #  * PHASE OF ANALYSIS *  #
#                ##   TITLE                #
#                ->   comments             #
#                                          # 
# 1) CV analysis                           #
# 2) Prediction                            #
# 3) Clustering (Usage)                    #
# 4) Appendix (i kept it for further use)  #
#                                          #
############################################

######################
# * IMPORTING DATA * #
#     Libraries      #
######################

setwd("~/projects/Karlis")

## Data manipulation
library(readxl)     
library(tidyverse) 
library(rowr)
library(jtools)
library(ggstance)
library(stargazer)
library(xtable)
library(plyr) 
options(xtable.floating = FALSE)
options(xtable.timestamp = "")

## Plotting
library(ggplot2)  
library(ggcorrplot)
library(cowplot)

## Clustering
library(corrgram)
library(HDclassif)
library(cluster)
library(mclust)
library(FactMixtAnalysis)
library(nnet)
library(class)
library(tree)
library(pgmm)
library(clValid)
library(caret)

## Modelling
library(aod)        
library(MASS)
library(car)
library(caTools)
library(pROC)
library(gridExtra)
library(psych)
library(ranger)
library(adabag)
library(e1071)
library(rpart)
library(gmodels)
library(neuralnet)
library(randomForest)
library(gbm)

## LOAD DATA AND DESCRIPTION 
# -> State - the state to which the subscriber belongs 
# -> Account length - no of weeks the account has been with the company 
# -> Area code - area code of the subscriber 
# -> international plan - weather the subscriber has international calling 
# -> voice mail plan - if the user is subscribed to voice mail service 
# -> Number_vmail messages - number of messages in user voicemail inbox 
# -> total_day_minutes - total minutes of callas made in the day per month
# -> total_eve_minutes - total minutes of callas made in the evening per month 
# -> total_night_minutes - total minutes of calls made in the  night per month
# -> total_day_calls - total number of calls made during the day per month 
# -> total_eve_calls - total number of calls made during the evening per month 
# -> total_night_calls - total number of calls made during the night per month 
# -> total_day_charge- total billed amount for day calls per month 
# -> total_eve_charge- total billed amount for evening calls per month 
# -> total_night_charge- total billed amount for night calls per month 
# -> total_intl_minutes - total minutes of calls made internationally per month
# -> total_intl_calls - total number of international calls made  per month 
# -> total_intl_charge- total billed amount for international calls per month 
# -> number_customer_service_calls - calls made to customer service per month 
# -> churn - binary variable describing if the customer has remained with the comppany or churned 

df = read_excel("churn.xls")
str(df)

## Rename columns
names(df) = gsub(" ", "_", names(df)) # Replace blank space with _
names(df) = gsub("'", "", names(df))  # Replace ' with no space 
colnames = names(df) ; print(colnames)

## Check for missing values
# -> there are not NA's 
sapply(df,function(x) sum(is.na(x)))

## Search dicotimic variables and uniqueness of each predictors
search_categorical = as.array(sapply(df, function(x) length(unique(x)))); print(search_categorical)
categorical = search_categorical[search_categorical <= 2]; print(categorical)

## Transformation
# -> each dicotomic variable is transformed into a factor 
# -> in order to use it properly in the modelling phase 
df$Churn  = as.factor(df$Churn)
df$Gender = as.factor(df$Gender)
df$Intl_Plan = as.factor(df$Intl_Plan)
df$VMail_Plan = as.factor(df$VMail_Plan)

# -> it's necessary to transform also charachers variables into categorical ones!
# -> we don't have ordinal features (are nominal) so use factor 
df$Area_Code = as.factor(df$Area_Code)
df$State = as.factor(df$State)

#  * assumption *   
# -> do not transform CustServ_Calls unique(df$CustServ_Calls)
# -> even if has just 9 values it's a strech to put it as a category!
# -> we also face issues if we want to use the model for prediction with different 
# -> customer's data in the presence of a customers with call the customer service
# -> more than 9 times 

################
### DATASETS ###
################

## *  Prediction 
df = df[,-which(names(df) %in% c("Day_Charge","Eve_Charge","Night_Charge","Intl_Charge"))]
print(length(df))

## * Clustering on Usage
num_var = which(sapply(df, is.numeric))
df_numerical = df[,num_var]; length(df_numerical)
summary(df_numerical)

df_scaled = as.data.frame(scale(df_numerical))
colMeans(df_scaled)     #faster version of apply(scaled.dat, 2, mean)
apply(df_scaled, 2, sd)
apply(df_scaled,2, mean)
summary(df_scaled)

##############################
#      * MODELLING *         #
# 1 CrossValidation-analysis #
##############################

## Let's find the baseline accuracy!
# All results should be higher! (0.8305)
baseline = 1 - sum(df$Churn == 1)/sum(df$Churn == 0); baseline

# --------------------------------------------------------------
## *1* REPEATED TRAIN/TEST SPLIT , 
#      also know as : Leave-group-out CV | Montecarlo CV | Random CV

# This technique simply creates multiple splits
# of the data into modeling and prediction sets
# The number of repetitions is important.
# Increasing the number of subsets has the effect
# of decreasing the uncertainty of the performance estimates.

acc_logistic_full = NULL 
k = 100

pbar = create_progress_bar("text")
pbar$init(k)
tic = proc.time()
set.seed(54321)

for(i in 1:k) {
  
  #  **  DATA SPLITTING ** 
  cat("\nRunning sample ",i,"\n")
  # stratified random sample 
  index<- createDataPartition(df$Churn, p = .8,list = FALSE)
  # use 80% of the original training data for training
  train <- df[index,] 
  # use the remaining 20% of the original training data for validation
  test  <- df[-index,] 
  
  #  ** MODEL BUILDING ** 
  
  # -> Stepwise Logistic Regression <-
  
  # run the full model 
  model = glm(Churn ~. , family = "binomial", data = train)
  # predict with new test data
  prediction = predict(model, newdata = test, type= "response")
  # cut-off 0.5
  results = as.factor(ifelse(prediction > 0.5,1,0 ))
  # true answers from test set
  answers = test$Churn
  # collecting results
  misClassError  = mean(answers!=results)
  acc_logistic_full[i] = 1-misClassError 
  
  pbar$step()
  time_log = proc.time() - tic
  
}  

# elapsed is the total #seconds to run this chunk of code
time_log # 20sec

# I cannot take CI because samples are not indipendent
# -> i will use standard deviations of the values
#    as a measure of stability of the model 
mean_full_logit = mean(acc_logistic_full); mean_full_logit
sd_full_logit = sd(acc_logistic_full); sd_full_logit

# ----------------------------------------------------

## *2* BOOTSTRAP 

# The bootstrap sample is the same size as the original data set.
# As a result, some samples will be represented multiple times 
# in the bootstrap sample while others will not be selected at all.
# The samples not selected are usually referred to as the “out-of-bag” samples.
# For a given iteration of bootstrap resampling, a model is built on the selected
# samples and is used to predict the out-of-bag samples

# In general, bootstrap error rates tend to have less uncertainty than k-fold cross-validation
# However, on average, 63.2 % of the data points the bootstrap sample are represented at least 
# once, so this technique has bias approximately similar
# to k-fold cross-validation when k ≈ 2.
# If the training set size is small, this bias may be problematic,
# but will decrease as the training set sample size becomes larger.

R = 1000
n = nrow(df)

acc_boot = NULL

pbar = create_progress_bar("text")
pbar$init(R)
tic = proc.time()
set.seed(54321)

for(i in 1:R){
  
  # draw a random sample
  obs.boot  = sample(x = 1:n, size = n, replace = T)
  checklenght = length(unique(obs.boot))
  checkprop   = checklenght/nrow(df)
  train_boot  = df[unique(obs.boot), ]
  test_boot = df[-unique(obs.boot),]
  
  # fit the model on bootstrap sample
  logit.boot <- glm(Churn ~. , 
                    data=train_boot,
                    family = "binomial")
  
  # apply model to bootstrap data
  prob = predict(logit.boot, type='response', test_boot)
  # cut-off 0.5
  results = as.factor(ifelse(prob > 0.5,1,0 ))
  # true answers from test set
  answers = test_boot$Churn
  # collecting results
  misClassError  = mean(answers!=results)
  acc_boot[i] = 1 - misClassError
  
  pbar$step()
  time_log_boot = proc.time() - tic
  
}
time_log_boot # 12 sec 

mean_boot = mean(acc_boot); mean_boot
sd_boot = sd(acc_boot); sd_boot

# It is not applied "632 method"
# The formula for further development
# --> (0.632 × simple bootstrap estimate) + (0.368 × apparent error rate).
# The modified bootstrap estimate reduces the bias,
# but can be unstable with small samples sizes.
# This estimate can also result in unduly optimistic results 
# when the model severely over-fits the data, since the apparent error rate will be close to zero!

# -------------------------------------------------------------
## *3* K-FOLD, with fold = 10

acc_10CV = NULL

#Randomly shuffle the data
dfCV <-df[sample(nrow(df)),]

#Create 10 equally size folds respecting the proportion (function from caret)
folds <- createFolds(df$Churn, k = 10, list = FALSE)

pbar = create_progress_bar("text")
pbar$init(R)
tic = proc.time()
set.seed(54321)

#Perform 10 fold cross validation
for(i in 1:10){
  
  #  **  DATA SPLITTING ** 
  cat("\nRunning sample ",i,"\n")
  index <- which(folds==i,arr.ind=TRUE)
  test <- df[index, ]
  train <- df[-index, ]
  
  #  ** MODEL BUILDING ** 
  
  # -> Logistic Regression <-
  
  # run the full model 
  model = glm(Churn ~. , family = "binomial", data = train)
  # predict with new test data
  prediction = predict(model, newdata = test, type= "response")
  # cut-off 0.5
  results = as.factor(ifelse(prediction > 0.5,1,0 ))
  # true answers from test set
  answers = test$Churn
  # collecting results
  misClassError  = mean(answers!=results)
  acc_10CV[i] = 1-misClassError 
  
  pbar$step()
  time_CV = proc.time() - tic
  
}

time_CV # 2sec
mean_acc10CV = mean(acc_10CV); mean_acc10CV
sd_acc10CV = sd(acc_10CV); sd_acc10CV

#----------------------------------------------------------------------------
## *4* K-FOLD, with fold = 10, repeated 10 times
pbar = create_progress_bar("text")
pbar$init(R)
tic = proc.time()
set.seed(54321)

times = 10 
cv_errors = matrix(NA ,times,10, dimnames =list(NULL , paste (1:10) ))

times = 10
for (j in 1:times){
  cat("\nRunning block ",i,"\n")
  #Randomly shuffle the data
  dfCV <-df[sample(nrow(df)),]
  
  #Create 10 equally size folds respecting the proportion (function from caret)
  folds <- createFolds(df$Churn, k = 10, list = FALSE)
  
  #Perform 10 fold cross validation
  for(i in 1:10){
    
    #  **  DATA SPLITTING ** 
    cat("\nRunning sample ",i,"\n")
    index <- which(folds==i,arr.ind=TRUE)
    test <- df[index, ]
    train <- df[-index, ]
    
    #  ** MODEL BUILDING ** 
    
    # -> Logistic Regression <-
    
    # run the full model 
    model = glm(Churn ~. , family = "binomial", data = train)
    # predict with new test data
    prediction = predict(model, newdata = test, type= "response")
    # cut-off 0.5
    results = as.factor(ifelse(prediction > 0.5,1,0 ))
    # true answers from test set
    answers = test$Churn
    # collecting results
    misClassError  = mean(answers!=results)
    cv_errors[j,i] = 1-misClassError 
    
    pbar$step()
    time_CVCV = proc.time() - tic
    
  }
}
time_CVCV #14sec

mean_acc1010CV = mean(apply(cv_errors,1,mean)); mean_acc1010CV
sd_acc1010CV = mean(apply(cv_errors,1,sd)); sd_acc1010CV

# ----------------------------------------------------------
# COMPARISON 
matrix = matrix(c(mean_full_logit,sd_full_logit, 
                  mean_boot, sd_boot,
                  mean_acc10CV, sd_acc10CV,
                  mean_acc1010CV, sd_acc1010CV),
                nrow = 2, ncol = 4)
colnames(matrix) <- c("randomCV","bootstrap","kfold10","10xKfold10"); matrix

## -->  Bootstrap method tends to have less uncertainity 


####################
#   2.PREDICTION   #
####################

#  Very bad results obtained with logistic regression 
#  (just 2.8% more accurate than all prediction 0)
difference_from_baseline = mean_boot - baseline; difference_from_baseline

## ---- ! START ! ----
set.seed(54321)
test <- createDataPartition( df$Churn, p = .2, list = FALSE )
data_train <- df[ -test, ]
data_test  <- df[ test, ]

##  Check LDA | QDA ? --> we should prefer QDA
library(heplots)
boxM(Y = df_numerical,  group = df$Churn)

model_lda = lda(Churn ~. , data = data_train)
pred_lda =  predict(model_lda, data_test)$class
accuracy_lda = 1 - mean(pred_lda!=data_test$Churn)
cat("\nThe accuracy of LDA is", accuracy_lda)

model_qda = qda(Churn ~. , data = data_train)
pred_qda =  predict(model_qda, data_test)$class
accuracy_qda = 1 - mean(pred_qda!=data_test$Churn)
cat("\nThe accuracy of QDA is", accuracy_qda)

# --> Apparently for accuracy purpose QDA doesn't work at all 

# ------------------------------------------------------------
## NAIVE BAYES 
#  It's a silly tentative: not clear how the function treat num/cat variables.
#  Laplace indeed is for categorical, but here improved the results. 
#  Anyway i will run a quick NB classifier, to get a first impression about accuracy 
#  Look the description-value-tables

model_naive = naiveBayes(Churn ~.,data = data_train,type=c("class"))
pred_naive  = predict(model_naive,data_test)

accuracy_nb = 1 - mean(pred_naive!=data_test$Churn)
cat("\nThe accuracy of NB Classifier is:",accuracy_nb*100)

model_naive_lap = naiveBayes(Churn ~.,data = data_train,type=c("class"),lap = 1/2)
pred_naive_lap  = predict(model_naive_lap,data_test)

accuracy_nb_lap = 1 - mean(pred_naive_lap!=data_test$Churn)
cat("\nThe accuracy of NB Classifier laplace = 1/2 is:",accuracy_nb_lap*100)

## Apparently NB doesn't perform well

# ------------------------------------------------------------
## SVM 
# -> bad tentative (i read later that svm work well with small datasets (<1000 rows))

# Setting dummy variables (works with distances, require numeric input)
dummies <- dummyVars(~ ., data_train[,-8])
c2 <- predict(dummies, data_train[,-8])
d_train = as.data.frame(c2)
# Normalize
d_train = as.data.frame(scale(d_train[,-72]))
d_train["Churn"] = data_train$Churn 

dummies <- dummyVars(~ ., data_test[,-8])
c2 <- predict(dummies, data_test[,-8])
d_test = as.data.frame(c2)
# Normalize 
d_test = as.data.frame(scale(d_test[,-72]))
d_test["Churn"] = data_test$Churn 

# tune on : 
# 1) C : regularization parameter of the error term 
# 2) Gamma : for non linear kernel
# 3) Kernel type

# Linear
costs <- c(0.001, 0.1, 1 ,50)
item = length(costs)

for(cost in costs) {
  svmfit <- svm(Churn ~ ., data = d_train, kernel = 'linear', cost = cost, scale = FALSE)
  cat('cost: ', cost,  ', training error: ', mean(predict(svmfit) != data_train$Churn), '\n')  
  cat('test error: ', round(mean(predict(svmfit, d_test) != d_test$Churn), 3), '\n')
}

# Output: Test error = 0.145

# Poly 

degrees <- c(2:5)
costs <- c(.001, 0.1, 1 , 4, 16, 32, 64)


for(cost in costs) {
  for(degree in degrees) {
    svmfit_poly <- svm(Churn ~ ., data = d_train, kernel = 'poly',
                       degree = degree, cost = cost, scale = FALSE, gamma = )
    cat('cost: ', cost,' degree: ', degree, ', training error: ', mean(predict(svmfit_poly) != d_train$Churn), '\n')
    cat('test error: ', round(mean(predict(svmfit_poly, d_test) != d_test$Churn), 3), '\n')
  }
 }

# paste output: cost:  1  degree:  2 , training error:  0.1102776  test error:  0.133 


# Radial kernel 
for(cost in costs) {
  svmfit_poly <- svm(Churn ~ ., data = d_train, kernel = 'radial',
                     degree = degree, cost = cost, scale = FALSE, gamma = )
  cat('cost: ', cost,'training error: ', mean(predict(svmfit_poly) != d_train$Churn), '\n')
  cat('test error: ', round(mean(predict(svmfit_poly, d_test) != d_test$Churn), 3), '\n')
}

# paste output : cost:  1 training error:  0.1155289test error:  0.141 

# We can see how parameter increasing C  overfits the training data
# However the result obtained suggest to use another method:
# According to this random split best results are obtained with cost = 1
# I should use some CV to be confident, 
# but this results are really close to the baseline accuracy
# that imply that other methods should perform well 

# ------------------------------------------------------------------------
## DECISION TREE + RANDOM FOREST 
# I will try decision tree + Ensembling in the next steps
# that usually performs well in many situations.
# Ensembling could improve again the performarce, but is it not developed!
# Much easier with Caret functionalities 

## Tree 
model_dtree = rpart(Churn~.,method='class',data=data_train, 
                    parms = list (split = 'information'))

pred<-predict(model_dtree,data_test,type=c("class"))

accuracy_tree = 1 - mean(pred!=data_test$Churn)
cat("\nThe accuracy of Decision tree Classifier is:",accuracy_tree*100)

## Rf 
model_forest = randomForest(Churn ~.,data=data_train,mtype=12, ntree=1000)
pred  = predict(model_forest , newdata= data_test)
accuracy_rf <- 1 - mean(pred!=data_test$Churn)
cat("\n Accuracy for Random Forest", accuracy_rf)  

# plot : after 200 tree we see that error is stable 
plot(model_forest)

## Tune RF -- > suggest mtry = 5-7
model_rf_tuned = tuneRF(x = data_train[,-8],
                        y = data_train$Churn,
                        ntreeTry = 300,
                        mtryStart  = 5,
                        stepFactor = 1.5,
                        improve    = 0.01,
                        trace      = FALSE )


# hyperparameter grid search
# Use ranger to tune which is more faster!
hyper_grid <- expand.grid(
  mtry       = seq(3, 8, by = 1),     # var to use. Theory is sqrt(p) = 4/5
  node_size  = seq(2, 9, by = 1),     # min. size of terminal nodes
  sampe_size = c(.632, .70, .80, .9, 1),
  OOB_class   = 0)

# We loop over 240 rows for 3cols and fill the 4col
nrow(hyper_grid)

for(i in 1:nrow(hyper_grid)) {
  
  # train model
  model <- ranger(
    formula         = Churn ~ ., 
    data            = data_train, 
    num.trees       = 300,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 54321
  )
  
  # add OOB error to grid
  hyper_grid$OOB_class[i] <- model$prediction.error
}

hyper_grid %>% 
  dplyr::arrange(OOB_class) %>%
  head(10)

min = min(hyper_grid$OOB_class)
selectbest = hyper_grid[hyper_grid$OOB_class == min,]; print(selectbest)


# -> OBB ERROR FROM 6.26% IS DECREASE TO 4.61%

# Paste output 
#       mtry node_size sampe_size  OOB_class
# 102    8         2        0.8    0.04613653
# 132    8         7        0.8     0.04613653

# Predict

model_ranger <- ranger(
  formula         = Churn ~ ., 
  data            = data_train, 
  num.trees       = 300,
  mtry            = 8,
  min.node.size   = 2,
  sample.fraction = 0.8,
  seed            = 54321
)
model_ranger

pred_ranger <- predict(model_ranger, data_test)
accuracy_rf_tuned <- 1 - mean(pred_ranger$predictions != data_test$Churn)
cat("\n Accuracy for Random Forest tuned ", accuracy_rf_tuned)  

# Ok we improved the result!

# --------------------------------------------------

#####################
## FINAL ASSESMENT ##
#####################

# A) Bootstrap 

R = 100
n = nrow(df)

acc_boot_rftuned = NULL

pbar = create_progress_bar("text")
pbar$init(R)
tic = proc.time()
set.seed(54321)

for(i in 1:R){
  
  # draw a random sample
  obs.boot  = sample(x = 1:n, size = n, replace = T)
  checklenght = length(unique(obs.boot))
  checkprop   = checklenght/nrow(df)
  train_boot  = df[unique(obs.boot), ]
  test_boot = df[-unique(obs.boot),]
  
  # fit the model on bootstrap sample
  model_ranger <- ranger(
    formula         = Churn ~ ., 
    data            = train_boot, 
    num.trees       = 300,
    mtry            = 8,
    min.node.size   = 2,
    sample.fraction = 0.8,
    seed            = 54321
  )
  
  # Predict 
  pred_ranger <- predict(model_ranger, test_boot)
  acc_boot_rftuned[i] <- 1 - mean(pred_ranger$predictions != test_boot$Churn)
  
  pbar$step()
  time_log_boot = proc.time() - tic
  
}

time_log_boot # 15sec 
mean_boot = mean(acc_boot_rftuned); mean_boot
sd_boot = sd(acc_boot_rftuned); sd_boot


# B) Repeated cross-validation (10x10)

pbar = create_progress_bar("text")
pbar$init(R)
tic = proc.time()
set.seed(54321)

times = 10
cv_errors = matrix(NA ,times,10, dimnames =list(NULL , paste (1:10) ))

for (j in 1:times){
  cat("\nRunning block ",i,"\n")
  #Randomly shuffle the data
  dfCV <-df[sample(nrow(df)),]
  
  #Create 10 equally size folds respecting the proportion (function from caret)
  folds <- createFolds(df$Churn, k = 10, list = FALSE)
  
  #Perform 10 fold cross validation
  for(i in 1:10){
    
    #  **  DATA SPLITTING ** 
    cat("\nRunning sample ",i,"\n")
    index <- which(folds==i,arr.ind=TRUE)
    test <- df[index, ]
    train <- df[-index, ]
    
    #  ** MODEL BUILDING ** 
    
    # -> RF <-

    # Build
    model_ranger <- ranger(
      formula         = Churn ~ ., 
      data            = train, 
      num.trees       = 300,
      mtry            = 8,
      min.node.size   = 2,
      sample.fraction = 0.8,
      seed            = 54321
    )
    
    # Predict 
    pred_ranger <- predict(model_ranger, test)
    misClassError <-mean(pred_ranger$predictions != test$Churn)
    cv_errors[j,i] = 1-misClassError 
    
    pbar$step()
    time_CVCV = proc.time() - tic
    
  }
}
time_CVCV #22sec

mean_acc1010CV = mean(apply(cv_errors,1,mean)); mean_acc1010CV
sd_acc1010CV = mean(apply(cv_errors,1,sd)); sd_acc1010CV


## COMPARE

finalmatrix = matrix(c(mean_boot, sd_boot,  
                       mean_acc1010CV, sd_acc1010CV),
                       nrow = 2, ncol = 2)
colnames(finalmatrix) <- c("Bootstrap_RF","10xKfold10_RF"); finalmatrix

## ---------------------------------------------------------------------

#########################
###     CLUSTERING    ### 
#########################

# !!! ---> The code here is a little bit messy due to various tentative of clustering 
#          However this is the approach that i follow.


# In practice, we try several different choices,
# and look for the one with the most useful or interpretable solution.
# With these methods, there is no single right answer -
# any solution that exposes some interesting aspects of the data should be considered.

### USAGE ###
num_var = which(sapply(df, is.numeric))
df_numerical = df[,num_var]; length(df_numerical)
summary(df_numerical)

### COMPRESS INFO ###
# -> After several tentative, 
#    it is useful to compress the information contained 
#    in the total mins / calls by period of the day of each customer  
df_numerical["Total_Mins"] = df_numerical$Day_Mins + df_numerical$Eve_Mins + df_numerical$Night_Mins 
df_numerical["Total_Calls"] =  df_numerical$Day_Calls + df_numerical$Eve_Calls + df_numerical$Night_Calls 

#df_numerical["Total_time_Intl"] =  (df_numerical$Intl_Mins / df_numerical$Intl_Calls)

df_restricted = df_numerical[,c("Total_Calls","Total_Mins","Intl_Mins")]
df_restricted = na.omit(df_restricted)
summary(df_restricted)
df_scaled = as.data.frame(scale(df_restricted))
apply(df_scaled, 2, sd)
apply(df_scaled,2, mean)
summary(df_scaled)

# ---------------------------------------------------

## 1.HIERARCHICAL CLUSTERING

## SINGLE: 
#  -> height low, high chain effect (discard)
hc_single<-hclust(dist(df_scaled),method="single")
plot(hc_single, main="Single Linkage", xlab="", sub="", cex =.9)
rect.hclust(hc_single, k=3, border="red") # take the 3 clusters
plot(hc_single$height)  

##  COMPLETE : height low,  btw 2-4 (probably discard)
hc_complete<-hclust(dist(df_scaled),method="complete")
plot(hc_complete, main="Complete Linkage", xlab="", sub="", cex =.9)
rect.hclust(hc_complete, k=3, border="red") # take the 3 clusters
plot(hc_complete$height) 

## WARD : graphically best :  k = 2 / 5 should be fine
hc_ward<-hclust(dist(df_scaled),method="ward.D2")
plot(hc_ward, main="Ward Linkage", xlab="", sub="", cex =.9)
rect.hclust(hc_ward, k=5, border=2:5) # take the 5 clusters
plot(hc_ward$height)  

## AVERAGE : height low, bad result 
hc_avg<-hclust(dist(df_scaled),method="average")
plot(hc_avg, main="Avg Linkage", xlab="", sub="", cex =.9)
rect.hclust(hc_avg, k=3, border=2:6) # take the 3 clusters
plot(hc_avg$height) 

### CHOOSE METHOD ###

# Let's calculate the value produced 
# by agnes function used to perform hierarchical clustering 
# I can get the agglomerative coefficient, which measures the 
# amount of clustering structure found
# (values closer to 1 suggest strong clustering structure)

# library(purrr)
# 
# # methods to assess
# m <- c( "average", "single", "complete", "ward")
# names(m) <- c( "average", "single", "complete", "ward")
# 
# # function to compute coefficient
# ac <- function(x) {
#   agnes(df_scaled, method = x)$ac
# }
# 
# map_dbl(m, ac)

# *OUTPUT* : 
#  average    single  complete      ward 
#  0.8975368 0.7564674 0.9332552 0.9910372 

## --> According to that results we should use Ward linkage
##  Moreover, I see my customer segments as types ,
##  more or less spherical shapes with compaction(s) 
##  in the middle I'll choose Ward's linkage method or K-means,
##  but never single linkage method

## CREATE CLASSIFICATIONS

# https://stats.stackexchange.com/questions/21807/evaluation-measures-of-goodness-or-validity-of-clustering-without-having-truth
# I SHOULD compare externally ( forget about churn )


# Connectivity and Silhouette are both measurements of connectedness
# while the Dunn Index is the ratio of the smallest distance between 
# observations not in the same cluster to the largest intra-cluster distance.
# Recall that the connectivity should be minimized,
# while both the Dunn index and the silhouette width should be maximized.

## --> Remember we cannot compare different silhoutte
##     produced by different method.
##     This analyisis aims to just select the optimal number of cluster for each linkage
##     --> Looking ward column!

clas1<-cutree(hc_single, k=2:10)   # single
clas2<-cutree(hc_complete, k=2:10) # complete
clas3<-cutree(hc_avg, k=2:10)      # average
clas4<-cutree(hc_ward, k=2:10)     # ward


silo_single = NULL
silo_complete = NULL
silo_avg = NULL
silo_ward = NULL 
for (i in 1:ncol(clas1)){
  
  calc_sing = silhouette(clas1[,i] , dist(df_scaled))
  calc_comp = silhouette(clas2[,i] , dist(df_scaled))
  calc_avg = silhouette(clas3[,i] , dist(df_scaled))
  calc_ward = silhouette(clas4[,i] , dist(df_scaled))
  
  silo_single[i] =  summary(calc_sing)$avg.width
  silo_complete[i] =  summary(calc_comp)$avg.width
  silo_avg[i] =  summary(calc_avg)$avg.width
  silo_ward[i] =  summary(calc_ward)$avg.width
  
}
bigmatrix = t(matrix(c(silo_single,silo_complete,silo_avg, silo_ward),
                     nrow = 4, ncol = 9, byrow = T))
colnames(bigmatrix) = c("single","complete","average","ward")
rownames(bigmatrix) = c(2,3,4,5,6,7,8,9,10)
bigmatrix

plot(silo_single, ylim = c(0,1), type = "b", col='black',pch = 4, main = "Silhouette vs Linkage",  xaxt='n')
lines(silo_complete, col='red', type = "b",pch = 4)
lines(silo_avg, col='green',type = "b", pch = 4)
lines(silo_ward, col = "yellow", type = "b", pch = 4)
axis(1,at = 1:9, labels = 2:10)
# Add a legend
legend(1, 0.95, legend=c("Single", "Complete","Average","Ward"),
       col=c("black", "red","green","yellow"), lty=1:2, cex=0.8)


#plot(silhouette(clas4[,1] , dist(df_scaled)))

## Let's try with DUNN INDEX 

dunn_single = NULL
dunn_complete = NULL
dunn_avg = NULL
dunn_ward = NULL 

for (i in 1:ncol(clas1)){
  
  calc_sing = dunn(clusters = clas1[,i] , Data =  df_scaled, method = "euclidean")
  calc_comp = dunn(clusters = clas2[,i] , Data = df_scaled, method = "euclidean")
  calc_avg = dunn(clusters  = clas3[,i] , Data = df_scaled, method = "euclidean")
  calc_ward = dunn(clusters = clas4[,i] ,  Data = df_scaled, method = "euclidean")
  
  dunn_single[i] =  calc_sing
  dunn_complete[i] =  calc_comp
  dunn_avg[i] =  calc_avg
  dunn_ward[i] =  calc_ward
  
}

bigmatrix2 = t(matrix(c(dunn_single,dunn_complete,dunn_avg, dunn_ward),
                      nrow = 4, ncol = 9, byrow = T))
colnames(bigmatrix2) = c("single","complete","average","ward")
rownames(bigmatrix2) = c(2,3,4,5,6,7,8,9,10)
bigmatrix2

plot(dunn_single, ylim = c(0,1), type = "b", col='black',pch = 4, main = "Dunn vs Linkage",  xaxt='n')
lines(dunn_complete, col='red', type = "b",pch = 4)
lines(dunn_avg, col='green',type = "b", pch = 4)
lines(dunn_ward, col = "yellow", type = "b", pch = 4)
axis(1,at = 1:9, labels = 2:10)
# Add a legend
legend(1, 0.95, legend=c("Single", "Complete","Average","Ward"),
       col=c("black", "red","green","yellow"), lty=1:2, cex=0.8)

# --- > K = 2 / 3 in each linkage 

### PACKAGES ###

# Compute the number of clusters
# Take approximately 8-10 mins 
# library(NbClust)
# nb <- NbClust(df_scaled, distance = "euclidean", min.nc = 2,
#               max.nc = 10, method = "ward.D2", index ="all")
# 
# fviz_nbclust(nb) + theme_minimal()

# Paste the output
# ******************************************************************* 
#   * Among all indices:                                                
#   * 8 proposed 2 as the best number of clusters 
#   * 4 proposed 3 as the best number of clusters 
#   * 5 proposed 4 as the best number of clusters 
#   * 4 proposed 7 as the best number of clusters 
#   * 1 proposed 9 as the best number of clusters 
#   * 1 proposed 10 as the best number of clusters 
# 
# ***** Conclusion *****                            
#   
#   * According to the majority rule, the best number of clusters is  2 
# 
# 
# ******************************************************************* 

# PLOT
# ---> Let's try to see the results of hierarchical clustering 
##      Remember both suggest K = 2  (or K = 3 still acceptable)

choose = cbind(bigmatrix[,4], bigmatrix2[,4])
colnames(choose) = c("Silhouette","Dunn"); choose

# from the graph we can take btw 2-5 clusters
plot(hc_ward) 
rect.hclust(hc_ward, k = 2 , border = 2:6)
rect.hclust(hc_ward, k = 3 , border = 2:6)
rect.hclust(hc_ward, k = 4 , border = 2:6)
rect.hclust(hc_ward, k = 5 , border = 2:6)

final2 = cutree(hc_ward, k=2)
final3 = cutree(hc_ward, k=3)
final4 = cutree(hc_ward, k=4)
final5 = cutree(hc_ward, k=5)

table(final2)
table(final3)
table(final4)
table(final5)

df_cl2 =  mutate(df_restricted, cluster = final2)
df_cl3 =  mutate(df_restricted, cluster = final3)
df_cl4 =  mutate(df_restricted, cluster = final4)
df_cl5 =  mutate(df_restricted, cluster = final5)

# graphical evaluation
library(factoextra)
fviz_cluster(list(data = df_scaled, cluster = final2))
fviz_cluster(list(data = df_scaled, cluster = final3))
fviz_cluster(list(data = df_scaled, cluster = final4))
fviz_cluster(list(data = df_scaled, cluster = final5))

open3d()
plot3d(df_scaled, size=5, col=final4, aspect = F) 


df_groupmean2 = as.data.frame(df_restricted) %>%
                       mutate(Cluster = df_cl2$cluster) %>%
                       group_by(Cluster) %>%
                       summarise_all("mean") 

df_groupmean3 = as.data.frame(df_restricted) %>%
                        mutate(Cluster = df_cl3$cluster) %>%
                        group_by(Cluster) %>%
                        summarise_all("mean") 

df_groupmean4 = as.data.frame(df_restricted) %>%
                        mutate(Cluster = df_cl4$cluster) %>%
                        group_by(Cluster) %>%
                        summarise_all("mean") 

df_groupmean5 = as.data.frame(df_restricted) %>%
                        mutate(Cluster = df_cl5$cluster) %>%
                        group_by(Cluster) %>%
                        summarise_all("mean") 

## PLOTTING FOR K = 2,3,4,5

# Radar Plot K = 2
library(ggplot2)
library(ggiraphExtra)

df_df <- as.data.frame(df_restricted) %>% rownames_to_column()
cluster_pos <- as.data.frame(df_cl2$cluster) %>% rownames_to_column()
colnames(cluster_pos) <- c("rowname", "cluster")
df_radar <- inner_join(cluster_pos, df_df)
print(df_radar)

ggRadar(df_radar[-1], aes(group = cluster), rescale = T, legend.position = "none", size = 1, interactive = FALSE, use.label = TRUE) + 
  facet_wrap(~cluster) +
  scale_y_discrete(breaks = T) +
  theme(axis.text.x = element_text(size = 10)) + scale_fill_manual(values = rep("#1c6193", nrow(df_final))) +
  scale_color_manual(values = rep("#1c6193", nrow(df_final))) +
  ggtitle("Clusters groups")

# Pairs-panel Plot
library(GGally)
df_pair <- as.data.frame(df_restricted)
df_pair$cluster <- df_cl3$cluster
df_pair$cluster <- as.character(df_pair$cluster)
ggpairs(df_pair, 1:4, mapping = ggplot2::aes(color = cluster, alpha = 0.5), 
        diag = list(continuous = wrap("densityDiag")), 
        lower=list(continuous = wrap("points", alpha=0.9)))


### 2. K-MEANS WITH PCA ### 
min.clusters <- 2
max.clusters <- 10

clusters.num <- min.clusters:max.clusters
ratio.ss <- rep(0, max.clusters-min.clusters+1)
clusters <-matrix(nrow = nrow(df_scaled), ncol =  max.clusters-min.clusters+1)

cl <- kmeans(df_scaled,2, iter.max=40)

for( i in min.clusters:max.clusters ){
  cl <- kmeans(df_scaled,i, iter.max=40)
  ratio.ss[i-min.clusters+1] <- cl$tot.withinss / cl$totss
  clusters[,i-min.clusters+1] <- cl$cluster
}
clusters
ratio.ss

ggplot( as.data.frame(cbind(clusters.num, ratio.ss)), aes(x=clusters.num, y=ratio.ss)) +
  geom_line() +
  theme_light() +
  xlab("Number of clusters") +
  ylab("Total within-cluster SS / Total SS")

# We don't see huge gap. quite arbitrary (3-8)


# Use PCA to reduce dimensionality
pca.out <-as.data.frame(princomp(df_scaled)$scores[,1:3])

gp <- list()

for( i in 1:9){
  g <- ggplot(data = pca.out, aes(x=Comp.1, y=Comp.2)) +
    geom_point(colour=as.factor(clusters[,i]), shape=as.factor(clusters[,i])) +
    theme_light() +
    xlab("") +
    ylab("") +
    ggtitle( paste("Clusters:", i+min.clusters-1, "Ratio:", ratio.ss[i]) )
  gp <- c(gp, list(g))
}

library(pdp)
do.call("grid.arrange", c(gp, ncol=3))


## Results
library(rgl)
as = kmeans(df_scaled,3,nstart= 10, iter.max = 30)
plot(df_scaled,col = as$cluster, pch=19)

open3d()
plot3d(df_scaled, size=5, col=as$cluster, aspect = F) 
plot3d(df_scaled[,4:6], size=5, col=as$cluster, aspect = F)
plot3d(df_scaled[,6:9], size=5, col=as$cluster, aspect = F) 

table(as$cluster,clas4[,2]) 
adjustedRandIndex(clas4[,2],as$cluster)

# Plot agaìn

dfK_groupmean = as.data.frame(df_restricted) %>%
                      mutate(Cluster = as$cluster) %>%
                      group_by(Cluster) %>%
                      summarise_all("mean") 

# Radar Plot
df_df <- as.data.frame(df_numerical) %>% rownames_to_column()
cluster_pos <- as.data.frame(as$cluster) %>% rownames_to_column()
colnames(cluster_pos) <- c("rowname", "cluster")
df_radar <- inner_join(cluster_pos, df_df)

library(ggplot2)
library(ggiraph)
ggRadar(df_radar[-1], aes(group = cluster), rescale = T, legend.position = "none", size = 1, interactive = FALSE, use.label = TRUE) + 
  facet_wrap(~cluster) +
  scale_y_discrete(breaks = T) +
  theme(axis.text.x = element_text(size = 10)) + scale_fill_manual(values = rep("#1c6193", nrow(df_final))) +
  scale_color_manual(values = rep("#1c6193", nrow(df_final))) +
  ggtitle("Clusters groups")

# Pairs-panel Plot
library(GGally)
df_pair <- as.data.frame(df_scaled)
df_pair$cluster <- as$cluster
df_pair$cluster <- as.character(df_pair$cluster)
ggpairs(df_pair, 1:4, mapping = ggplot2::aes(color = cluster, alpha = 0.5), 
        diag = list(continuous = wrap("densityDiag")), 
        lower=list(continuous = wrap("points", alpha=0.9)))

df_groupmean = as.data.frame(df_numerical) %>%
  mutate(Cluster = df_cl3$cluster) %>%
  group_by(Cluster) %>%
  summarise_all("mean") 


#############################################################
##############             APPENDIX           ###############
#############################################################



### ---------------- ! DON'T RUN ! ------------------------



#########################
#  IMPROVING LOGISTIC   #
# -> to play with Fp,Fn #
#########################

# Source: https://github.com/pmaji/data-science-toolkit/blob/master/classification/logit/logistic_regression.md
library(scales)
library(ggplot2)
library(ggthemr)
library(ggthemes)

library(ROCR)
library(scales)
library(grid)
library(broom)
library(tidyr)
library(dplyr)

# We'd like to see that the 2 classes are separated,
# with the score of the negative instances to be on the 
# left and the score of the positive instance to be on the right

set.seed(54321)
test <- createDataPartition( df$Churn, p = .2, list = FALSE )
data_train <- df[ -test, ]
data_test  <- df[ test, ]

# traing logistic regression model
model_glm <- glm( Churn ~ . , data = data_train, family = binomial(logit) )
summary_glm <- summary(model_glm)

# prediction
prediction_train <- predict( model_glm, newdata = data_train, type = "response" )
predictions_train_full <- data.frame(prediction = prediction_train, Churn = data_train$Churn)

prediction_test <- predict( model_glm, newdata = data_test , type = "response" )
predictions_test_full <- data.frame(prediction = prediction_test, Churn = data_test$Churn)

# distribution of the prediction score grouped by known outcome
ggplot( data_train, aes( prediction_train, color = as.factor(Churn) ) ) + 
  geom_density( size = 1 ) +
  ggtitle( "Training Set's Predicted Score" ) + 
  scale_color_economist( name = "data", labels = c( "negative", "positive" ) ) + 
  theme_economist()

# Provides functions to allow one to compose general HTTP requests, etc. in R
library(RCurl) 
# grabbing the raw info from my GitHub to turn into a text object
script <- getURL("https://raw.githubusercontent.com/pmaji/r-stats-and-modeling/master/classification/useful_classification_functions.R", ssl.verifypeer = FALSE)
# sourcing that code just like you might source an R Script locally
eval(parse(text = script))

# using newly-sourced function AccuracyCutoffInfo to test for optimal cutoff visually
accuracy_info <- AccuracyCutoffInfo(train = predictions_train_full, 
                                    test = predictions_test_full, 
                                    predict = "prediction", 
                                    actual = "Churn",
                                    # iterates over every cutoff value from 1% to 99% 
                                    # steps in units of 10 bps
                                    cut_val_start = 0.01,
                                    cut_val_end = 0.99,
                                    by_step_size = 0.001)

# from the plot below we can begin to eyeball what the optimal cutoff might be 
accuracy_info$plot

# Moving on to using receiver operating characteristic (ROC) Curves to pinpoint optimal cutoffs

# user-defined costs for false negative and false positive to pinpoint where total cost is minimized
cost_fp <- 1 # cost of false positive
cost_fn <- 1 # cost of false negative

# creates the base data needed to visualize the ROC curves
roc_info <- ROCInfo(data = predictions_test_full, 
                    predict = "prediction", 
                    actual = "Churn", 
                    cost.fp = cost_fp, 
                    cost.fn = cost_fn )
# plot the roc / cutoff-selection plots
# color on the chart is cost -- darker is higher cost / greener is lower cost
# ---- remove previous plot -----
grid.draw(roc_info$plot)

# visualize a particular cutoff's effectiveness at classification
cm_info <- ConfusionMatrixInfo(data = predictions_test_full, 
                               predict = "prediction", 
                               actual = "Churn", 
                               cutoff = .81) # (determined by roc_info$plot above)

# prints the visualization of the confusion matrix (use print(cm_info$data) to see the raw data)
cm_info$plot

# getting model probabilities for our testing set
simple_logit_fit_probs <- predict(model,
                                  newdata = data_test,
                                  type = "response")

# turning these probabilities into classifications using the cutoff determined above 
simple_logit_fit_predictions <- factor(ifelse(simple_logit_fit_probs > 0.81, 1, 0),levels=c('0','1'))

# builiding a confusion matrix 
simple_logit_conmatrix <- caret::confusionMatrix(simple_logit_fit_predictions, data_test$Churn, positive='1')
simple_logit_conmatrix


#################
### Too slow  ###
################# 

## --> USE CARET for this kind of model 
## --  I will explain in the report why i didn't select
## --  models produced by Caret.
 
# BOOSTING 
adaBoostingModel <- boosting(Churn ~. , data = data_train,
                              mfinal = 10 , control = rpart.control(maxdepth = 1))
 
 
# BAGGING
 baggingModel <- bagging(Churn ~. , data = data_train,
                         mfinal = 10, control = rpart.control(maxdepth = 1))
 
 
#################
###   CARET   ###
################# 

# Consideration : Caret is great, full documentated in the book of Max Kunh
# but its very slow to fit advanced models.
# They should be run in parallel, otherwise its impossible to conduct the analysis
# Indeed the typical pipeline is to 
# 1) Train in CV different parameters
# 2) Test in validation set 
# 3) Pick the best
# 4) Use only 1 test set (Real) because the evidence is that the procedure works well  

set.seed(998)
inTraining <- createDataPartition(df$Churn, p = .8, list = FALSE, times = 3)
# use 80% of the original training data for training
training <- df[ inTraining[,1],] ; nrow(training)
# use 20% of the original training data for testing ( ie Validation)
testing  <- df[-inTraining,] ; nrow(testing)
 
 
# Specify type of resampling 
# cross validation is repeated 10 times 
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10)
 
# fit a boosted tree model 
set.seed(825) #  to assure that same resamples are used
 
# takes 3-4 mins
gbmFit1 <- train(Churn ~ ., data = training, 
                  method = "gbm", 
                  trControl = fitControl,
                  ## This last option is actually one
                  ## for gbm() that passes through
                  verbose = FALSE)
gbmFit1
 
# alternate tune grid
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                         n.trees = c(50,250,500,1000), 
                         shrinkage = 0.1,
                         n.minobsinnode = 20)
 
nrow(gbmGrid)
 
# more or less 30 mins 
set.seed(825)
gbmFit2 <- train(Churn ~ ., data = training, 
                  method = "gbm", 
                  trControl = fitControl, 
                  verbose = FALSE, 
                  ## Now specify the exact models 
                  ## to evaluate:
                  tuneGrid = gbmGrid)
gbmFit2
 
trellis.par.set(caretTheme())
plot(gbmFit2) 
plot(gbmFit2, metric = "Accuracy", plotType = "level",
      scales = list(x = list(rot = 90)))
ggplot(gbmFit2)
 
gbmpred = predict(gbmFit2, newdata = testing)
predict(gbmFit2, newdata = testing, type = "prob")
 
confusionMatrix(data = gbmpred, reference = testing$Churn)
 
accuracyCV = 1 - mean(gbmpred!=testing$Churn); accuracyCV
 
## exploring
densityplot(gbmFit2, pch = "|")
 
## comparing models
 
# takes 10 min 
set.seed(825)
svmFit <- train(Churn ~ ., data = training, 
                 method = "svmRadial", 
                 trControl = fitControl, 
                 preProc = c("center", "scale"),
                 tuneLength = 8)
svmFit
 
##
library(klaR)
set.seed(825)
rdaFit <- train(Churn~ ., data = training, 
                 method = "rda", 
                 trControl = fitControl, 
                 tuneLength = 4)
rdaFit 
 
##
resamps <- resamples(list(GBM = gbmFit2,
                           SVM = svmFit,
                           RDA = rdaFit))
resamps
 
summary(resamps)
par(mfrow = c(1,1))
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1))
 
### tune on the best model
fitControl <- trainControl(method = "none", classProbs = TRUE)
 
set.seed(825)
gbmFit4 <- train(Churn~ ., data = training, 
                  method = "gbm", 
                  trControl = fitControl, 
                  verbose = FALSE, 
                  ## Only a single model can be passed to the
                  ## function when no resampling is used:
                  tuneGrid = data.frame(interaction.depth = 9,
                                        n.trees = 500,
                                        shrinkage = .1,
                                        n.minobsinnode = 20))
summary(gbmFit4)
 
gbm4 = predict(gbmFit4, newdata = testing)
confusionMatrix(data = gbm4, reference = testing$Churn)
 
accuracyCV = 1 - mean(gbm4!=testing$Churn); accuracyCV
 
## -----------------------------------------------------------
 
## Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

## Benchmark - simple logistic regression model 
default_glm_mod = train(
  form = Churn ~ .,
  data = training,
  trControl = control,
  method = "glm",
  family = "binomial"
)
default_glm_mod$results

 
 #1 (kknn) k-Nearest Neighbors
 set.seed(12345)
 library(kknn)
 m_kknn <- train(Churn~., data=training, method="kknn", metric=metric, 
                 trControl=control, preProcess = c("center", "scale") )
 print(m_kknn)
 
 #2 (pda) Penalized Discriminant Analysis
 set.seed(12345)
 library(mda)
 m_pda <- train(Churn~., data=training, method="pda", metric=metric, 
                trControl=control, preProcess = c("center", "scale") )
 print(m_pda)
 
 #3 sda (Shrinkage Discriminant Analysis)
 set.seed(12345)
 library(sda)
 m_sda <- train(Churn~., data=training, method="sda", metric=metric, 
                trControl=control, preProcess = c("center", "scale") )
 print(m_sda)
 
 #4 (slda) Stabilized Linear Discriminant Analysis
 set.seed(12345)
 m_slda <- train(Churn~., data=training, method="slda", metric=metric, 
                 trControl=control, preProcess = c("center", "scale") )
 print(m_slda)
 
 #5 (hdda) High Dimensional Discriminant Analysis
 set.seed(12345)
 m_hdda <- train(Churn~., data=training, method="hdda", metric=metric, 
                 trControl=control, preProcess = c("center", "scale") )
 print(m_hdda)
 
 #6 (pam) Nearest Shrunken Centroids
 set.seed(12345)
 library(pamr)
 m_pam <- train(Churn~., data=training, method="pam", metric=metric, 
                trControl=control, preProcess = c("center", "scale") )
 print(m_pam)
 
 #7 C5.0Tree (Single C5.0 Tree)
 set.seed(12345)
 library(C50)
 m_C5 <- train(Churn~., data=training, method="C5.0Tree", metric=metric, 
               trControl=control, preProcess = c("center", "scale") )
 print(m_C5)

 
 ## 
 # calculate resamples // exclude SIMCA and PLS
 resample_results <- resamples(list(KKNN=m_kknn,PDA=m_pda,SDA=m_sda, 
                                    SLDA=m_slda, HDDA=m_hdda, PAM=m_pam, C5TREE=m_C5))
 
 
# --------------------------------------------------------------------
 

 ###   DF with all variables scaled    ###
 ## -> not showing interesting results  ##
 
 # df_scaled = as.data.frame(scale(df_numerical))
 # colMeans(df_scaled)  # faster version of apply(scaled.dat, 2, mean)
 # apply(df_scaled, 2, sd)
 # apply(df_scaled,2, mean)
 # summary(df_scaled)

 
# -------------------------------------------------- 

# http://www.sthda.com/english/wiki/print.php?id=243  
# To find optimal K, method ! 
# Misleading result! i should investigate


 library(clValid)
 df_scaled$ID  = rep(seq(1:3333))
 rownames(df_scaled) <- df_scaled$ID
 head(df_scaled)
 clmethods <- c("hierarchical","kmeans")
 
 # Check goodness by internal meausure 
 intern <- clValid(df_scaled, nClust = 2:8,
                   clMethods = clmethods, validation = "internal")
 
 # Summary
 summary(intern)  
 par(mfrow = c(1,3))
 plot(intern)
 
 # PASTE Optimal Scores:
 
 #             Score  Method     Clusters
 #Connectivity 4.9258 hierarchical 2       
 #Dunn         0.0080 kmeans       8       
 #Silhouette   0.6265 kmeans       2 