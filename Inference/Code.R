############################################
# *** HOW TO READ THE SCRIPT ***           #
#
# Each step has:                           #
#                #  * PHASE OF ANALYSIS *  #
#                ##   TITLE                #
#                ->   comments             #
############################################

######################
# * IMPORTING DATA * #
######################

## LIBRARIES 
setwd("~/projects/Karlis")

library(readxl)     # data manipulation
library(tidyverse) 
library(rowr)
library(jtools)
library(ggstance)
library(stargazer)
library(xtable)
options(xtable.floating = FALSE)
options(xtable.timestamp = "")

library(ggplot2)    # plotting 
library(ggcorrplot)
library(cowplot)

library(aod)        # analysis  
library(MASS)
library(car)
library(caret)
library(caTools)
library(pROC)
library(gridExtra)
library(psych)
library(e1071)

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

## RENAME COLUMNS 
names(df) = gsub(" ", "_", names(df)) # Replace blank space with _
names(df) = gsub("'", "", names(df))  # Replace ' with no space 
colnames = names(df) ; print(colnames)

## CHECK FOR MISSING VALUES
# -> there are not NA's 
sapply(df,function(x) sum(is.na(x)))

## SEARCH DICOTIMIC VARIABLES, AND UNIQUENESS OF EACH PREDICTORS
search_categorical = as.array(sapply(df, function(x) length(unique(x)))); print(search_categorical)
categorical = search_categorical[search_categorical <= 2]; print(categorical)

## TRANSFORMATION 
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

#################################
# * EXPLORATORY DATA ANALYSIS * #
#################################

## PLOTTING OUTCOME 
# -> classes are not pefectly balanced  
table(df$Churn)
prop = 483/(2850+483); print(prop)

df %>% 
  group_by(Churn) %>% 
  summarise(Count = n()) %>% 
  mutate(percent = prop.table(Count)*100) %>% 
  ggplot(aes(Churn, percent), fill = Churn) +
  geom_col(fill = c("#FC4E07", "#E7B800")) + 
  geom_text(aes(label = sprintf("%.2f%%", percent)),
            hjust = 0.01,vjust = -0.5, size =4) + 
  theme_bw() + 
  xlab("Churn") +
  ylab("Percent") + 
  ggtitle("Churn Percent")+
  theme(plot.title = element_text(size = 30))+
  theme(axis.title = element_text(size = 15))


## PLOT NUMERICAL VARIABLE
# -> useful to have a visual impact of the relations with the outcome 
# -> same shape.. seems that Charge is Mins are giving the same information
par(mfrow = c(2,4))

plot(df$Churn,df$Day_Mins,pch=16,ylab='Day_Mins',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Eve_Mins,pch=16,ylab='Eve_Mins',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Night_Mins,pch=16,ylab='Night_Mins',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Intl_Mins,pch=16,ylab='Intl_Mins',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Day_Charge,pch=16,ylab='Day_Charge',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Eve_Charge,pch=16,ylab='Eve_Charge',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Night_Charge,pch=16,ylab='Night_Charge',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Intl_Charge,pch=16,ylab='Intl_Charge',xlab='Churn', col=c("green","red"))
mtext("Relationship between Mins and Charge variables", side = 3, line = -2 , outer = TRUE)

plot(df$Churn,df$CustServ_Calls,pch=16,ylab='CustServ_Calls',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Day_Calls,pch=16,ylab='Day_Calls',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Eve_Calls,pch=16,ylab='Eve_Calls',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Night_Calls,pch=16,ylab='Night_Calls',xlab='Churn', col=c("green","red"))
plot(df$Churn,df$Intl_Calls,pch=16,ylab='Intl_Calls',xlab='Churn', col=c("green","red"))

CustServ_Calls = ggplot(df, aes(CustServ_Calls,fill =Churn)) + 
  geom_bar(position = "fill")+
  theme_bw() + 
  xlab("CustServ_Calls")+
  ggtitle("CustServ_Calls")+
  theme(plot.title = element_text(size = 30))+
  theme(axis.title = element_text(size = 12))
CustServ_Calls

Intl_Calls = ggplot(df, aes(Intl_Calls, fill = Churn)) + 
  geom_bar(position = "fill")+
  theme_bw() + 
  xlab("Intl_Calls") 
Intl_Calls

# -> We see a strange box-plot in CustServ_Calls
# -> that is due to the fact that has only 9 values
# -> but seems to be some relationships!
# -> same for Intl_Calls (just 20 values) but here there is no trend 

## PLOTTING CATEGORICAL
# -> Being m/f seems to be inifluent to churn, while the 
# -> other two variables seems to have an effect (opposite wrt churn)
Intl_Plan = ggplot(df, aes(Intl_Plan, fill = Churn)) +
  geom_bar(position = "fill") + 
  theme_bw() +  
  xlab("Intl_Plan")+
  ggtitle("Intl_Plan")+
  theme(plot.title = element_text(size = 30))

VMail_Plan = ggplot(df, aes(VMail_Plan, fill = Churn)) +
  geom_bar(position = "fill") + 
  theme_bw() +  
  xlab("VMail_Plan")+
  ggtitle("VMail_Plan")+
  theme(plot.title = element_text(size = 30))


Gender = ggplot(df, aes(Gender, fill = Churn)) +
  geom_bar(position ="fill") + 
  theme_bw() +  
  xlab("Gender") +
  ggtitle("Gender")+
  theme(plot.title = element_text(size = 30))
Gender

cowplot::plot_grid(Intl_Plan, VMail_Plan)

## PLOTTING THE REMAININ VARIABLES
# -> Very interesting! there is a relationship!
# -> possible roads:
# -> 1) take the variable as factor, which imply choosing a baseline,
# ->    which could be the state with more customers
# -> 2) Grouping each state in bigger regions (North, South, West, MidWest)
# -> My approach is to run a logistic model, with a step procedure
# -> to select the most important variables, encoding state
# -> as a dummy variable (k-1) states, choosing as a reference the 
# -> the most frequent state. I will try both the procedure 

State = ggplot(df, aes(State, fill = Churn)) +
  geom_bar(position = "fill") + 
  theme_bw() +  
  xlab("State")+
  ggtitle("State")+
  theme(plot.title = element_text(size = 30))+
  coord_flip()

State

# 1)
# -> set the reference as the State with more customers
# ll <-data.frame(table(df$State))
# max = ll[which.max(ll$Freq),]; print(max)
# df$State <- relevel(df$State,"WV"); print(levels(df$State)[1])

# 2)
## From the US CENSUS
NE.name <- c("Connecticut","Maine","Massachusetts","New Hampshire",
             "Rhode Island","Vermont","New Jersey","New York",
             "Pennsylvania")
NE.abrv <- c("CT","ME","MA","NH","RI","VT","NJ","NY","PA")
NE.ref <- c(NE.name,NE.abrv)

MW.name <- c("Indiana","Illinois","Michigan","Ohio","Wisconsin",
             "Iowa","Kansas","Minnesota","Missouri","Nebraska",
             "North Dakota","South Dakota")
MW.abrv <- c("IN","IL","MI","OH","WI","IA","KS","MN","MO","NE",
             "ND","SD")
MW.ref <- c(MW.name,MW.abrv)

S.name <- c("Delaware","District of Columbia","Florida","Georgia",
            "Maryland","North Carolina","South Carolina","Virginia",
            "West Virginia","Alabama","Kentucky","Mississippi",
            "Tennessee","Arkansas","Louisiana","Oklahoma","Texas")
S.abrv <- c("DE","DC","FL","GA","MD","NC","SC","VA","WV","AL",
            "KY","MS","TN","AR","LA","OK","TX")
S.ref <- c(S.name,S.abrv)

W.name <- c("Arizona","Colorado","Idaho","New Mexico","Montana",
            "Utah","Nevada","Wyoming","Alaska","California",
            "Hawaii","Oregon","Washington")
W.abrv <- c("AZ","CO","ID","NM","MT","UT","NV","WY","AK","CA",
            "HI","OR","WA")
W.ref <- c(W.name,W.abrv)

region.list <- list(
  Northeast=NE.ref,
  Midwest=MW.ref,
  South=S.ref,
  West=W.ref)

df$Regions = sapply(df$State, function(x) names(region.list)[grep(x,region.list)])
df$Regions = as.factor(df$Regions)

ll <-data.frame(table(df$Regions))
max = ll[which.max(ll$Freq),]; print(max)
df$Regions <- relevel(df$Regions,"South"); print(levels(df$Regions)[1])

Regions = ggplot(df, aes(Regions, fill = Churn)) +
  geom_bar(position = "fill") + 
  theme_bw() +  
  xlab("Regions")+
  ggtitle("Regions")+
  theme(plot.title = element_text(size = 30))

Regions

df$State = NULL

# -> Area_Code doesn't influence the churn/non churn 
Area_Code = ggplot(df, aes(Area_Code, fill = Churn)) +
  geom_bar(position = "fill") + 
  theme_bw() +  
  xlab("Area_Code")+
  ggtitle("Area_Code")+
  theme(plot.title = element_text(size = 30))

Area_Code

cowplot::plot_grid(State,Area_Code,Gender, nrow = 1)

## Need some feature engineering 
# -> Since VMAIL_messgage is a nested variable, meaning that 
# -> takes non zero-value only if the main dummy variable (VMAIL_plan)
# -> is true (1), we have a problem of an highly-skewed covariate,
# -> due to the high presence of zero (extremely zero-inflated covariate)
# -> Furthermore is highly collinear with VMAIL_Plan
# -> Even if it does not violate any assumption
# -> of the logistic model that i will create,
# -> a useful transformation in order to have a better interpretability 
# -> in the next steps
par(mfrow = c(1,1))
table(df$VMail_Message)

ggplot(data=df, aes( df$VMail_Message)) + 
  geom_histogram(breaks=seq(0, 50, by=2), 
                 col="red", 
                 fill="green", 
                 alpha = .2) + 
  labs(title="VMAIL_Message Histogram")

df$VMail_cat <- cut(df$VMail_Message, 
                    breaks=c(-Inf,0,25,Inf), 
                    labels=c("no","low","high"))

VMail_cat = ggplot(df, aes(VMail_cat, fill = Churn)) +
  geom_bar(position = "fill") + 
  theme_bw() +  
  xlab("Vmail_cat")+
  ggtitle("Vmail_cat")+
  theme(plot.title = element_text(size = 30))

VMail_cat

df$VMail_Plan = NULL
df$VMail_Message = NULL
## CHECKING MULTICOLLINEARITY BTW NUMERICAL VARIABLES 
# -> first of all we need to create a dataset composed by numerical variables,
# -> then useful R functions comes in handy 
num_var = which(sapply(df, is.numeric))
df_numerical = df[,num_var]; length(df_numerical)

pairs.panels(df_numerical,hist.col = "green", stars = T,ellipses = F, gap = 0,scale =T,jiggle = T)
cor.plot(df_numerical)
#   * Assumption * 
# -> there is perfect correlation between the total number of mins that customers
# -> use the phone and the amount of charge, which is logical
# -> My approach is to eliminate from the dataset all columns that are related with
# -> charging, because the informations that this two group of predictors are redundant

df = df[,-which(names(df) %in% c("Day_Charge","Eve_Charge","Night_Charge","Intl_Charge"))]
print(length(df))

######################
# * Model Building * #
######################

## VARIABLE SELECTION:
## STEPWISE ALGORITH 

## 1
# set the seed for reproducibility
set.seed(1234)

# run the full model 
full = glm(Churn ~. , family = "binomial", data = df)
summary(full)

# run stepwise procedure (AIC criterion)
step(full, direction = "both", trace = T)

# take as a variable the model selected by the step procedure 
aic = step(full, direction = "both", trace = 0)

# run stepwise procedure (BIC criterion)
bic = step(full ,direction = "both", trace = 0, k = log(nrow(df)))

# Export in latex
stargazer(aic, bic, title = "AIC vs BIC selection", align = T, font.size = "scriptsize")

# Now let's analyze better the output of summary exploiting a useful package
summary(bic)

# check coefficient 
summ(bic, confint = TRUE, digits = 4)

# For exponential family models,i'm interested in getting the exponentiated coefficients 
# rather than the linear estimates
summ(bic, exp = TRUE, confint = TRUE, digits = 4)
pvalue = summary(bic)$coefficients[,4]
coef = summary(bic)$coefficients[,1]
exp = round(exp(cbind(OR = coef(bic), confint(bic))),5)
tot = round(cbind(coef,exp, pvalue),5)
xtable(tot, digits = 3)

# in order to  have a better idea of how the uncertainty
# and magnitude of effect differs for these variables.
# Since the aim is of the project is to interpret the most
# important variables i also plot the confidence level with a = 0.10
plot_summs(bic,scale =T,  inner_ci_level = .90)

## HOSMER-LEMESHOW TEST:
# small values with large p-values indicate a good fit to the data while large values with
# p-values below 0.05 indicate a poor fit. 
# on the left the ranges which we divided the dataset
# of the fitted probabilities (equal size)
library(generalhoslem)
hl  = logitgof(df$Churn, fitted(bic), g=10, ord = F); print(hl)
cbind(hl$expected, hl$observed)

## GOODNESS OF FIT
# llh :      log-likelihood from the fitted model
# llhNull:   The log-likelihood from the intercept-only restricted model
# G2:        Minus two times the difference in the log-likelihoods
# McFadden:  McFadden's pseudo r-squared
# r2ML:      Maximum likelihood pseudo r-squared (Cox & Sheel)
# r2CU:      Cragg and Uhler's pseudo r-squared (Nagelkerke's pseudo r-squared)
library(pscl)
bic_model = pR2(bic)
full_model = pR2(full)
vec3 = cbind(bic_model,full_model); vec3
xtable(vec3, digits = 3)

# The goodness of fit test simply compares the residual deviance 
# to a chi-square distribution.
# This tells you that:
# HO: model fits well
# H1: model doesn't fit well 
# -> we fail to reject the null 
with(bic, pchisq(deviance, df.residual, lower.tail = F))

## COMPARING AGAINST THE NULL MODEL 
# Test whether our model (m) fits significantly better than a model with just an intercept (n) 
# (i.e., a null model)
# There is significant difference between our model and the null model.
# (as i expected)
with(bic, null.deviance - deviance)
with(bic, df.null - df.residual)
with(bic, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

## VARIABLE IMPORTANCE
# To assess the relative importance of individual predictors in the model,
# we can also look at the absolute value of the t-statistic for each model parameter.
# This technique is utilized by the varImp function in the caret package for general 
# and generalized linear models.
varimp = varImp(bic)
head(varimp, 10)

###########################
### CHECKING ASSUMPTION ### 
###########################

## MULTICOLLINEARITY 
# Even if it is not required it is good idea to remove higly collinear variables
# -> there is no multicollinearity 
vif = car::vif(bic)
xtable(vif, digits = 4)
## LINEARITY ASSUMPTION 
residualPlots(bic)
termplot(bic)
## INDIPENDENCE OF RESIDUALS
plot(bic$residuals)
lag.plot(bic$residuals, lags = 1)
