library(survival) # for survival analysis
library(survminer) # Drawing Survival Curves using 'ggplot2'
library(lubridate) # for data manipulation
setwd("C:/Users/nimee/Documents/MSA Spring 2020/8200-Predictive Analytics/FinalProject")
library(readxl)


#Trial -1

## Survival Analysis (Event-Death)-Non Recovered Cases
df = read_excel("DeathAndRecovery_Final.xlsx")
df = df[df$Survival_Time!=120,]
df$Survival_Time[1:10]
df$SurvivalStatus[1:10]

#Transform data to Survival Object
Survival_Data = Surv(df$Survival_Time, df$SurvivalStatus)
head(Survival_Data)
Survival_Data[160:170]
#KM plot for 1 group
plot(survfit(Survival_Data ~ 1),xlab = "Days",ylab = "Overall Survival Probability") 
out = survfit(Survival_Data ~ 1)
KMdata = cbind(out$time,out$n.event,out$n.censor,out$n.risk)
colnames(KMdata) = c("time","n.event","n.censor","n.risk")
View(KMdata)

## KM plot for two groups in Gender
plot(survfit(Survival_Data ~ df$gender),xlab = "Days",ylab = "Overall survival probability") 
ggsurvplot(survfit(Survival_Data ~ df$gender),data = df)

##Logrank test-statistical check to see if there is any systematic pattern btw gender & survaival probability
survdiff(Survival_Data ~ df$gender) 

## What is the 1 month survival rate? : % survial after 30 days
#i.e what is the prob that a person will survive at 30 days
summary(survfit(Survival_Data ~ 1), times = 30) 

#Cox regression model and hazard ratio - helps us to evaluate how a varaible will have an effect on the survival rate
out2 = coxph(Surv(Survival_Time, SurvivalStatus) ~ gender12, data = df)
out2

#Hazard Ratio Plot
ggforest(out2, data = df)


#KM plot for Age groups
df = df[df$AgeBins!=0,]
ggsurvplot(survfit(Survival_Data ~ df$AgeBins),data = df)

#logrank test-statistical check to see if there is any systematic pattern btw Age groups & survaival probability
survdiff(Survival_Data ~ df$AgeBins) 

## Recovery Analysis (Event-Recovered): Non Death Cases
df = read_excel("DeathAndRecovery_Final.xlsx")
df = df[df$Recovery_Time!=120,]
df$Recovery_Time[1:10]
df$RecoveryStatus[1:10]
#Transform data to Survival Object
Recovery_Data = Surv(df$Recovery_Time, df$RecoveryStatus)
head(Recovery_Data)
Recovery_Data[135:145]

#KM plot for 1 group
plot(survfit(Recovery_Data ~ 1),xlab = "Days",ylab = "Overall Recovery Probability") 
out = survfit(Recovery_Data ~ 1)
KMdata = cbind(out$time,out$n.event,out$n.censor,out$n.risk)
colnames(KMdata) = c("time","n.event","n.censor","n.risk")
View(KMdata)


# KM plot for two groups in Gender
plot(survfit(Recovery_Data ~ df$gender),xlab = "Days",ylab = "Overall Recovery Probability")
ggsurvplot(survfit(Recovery_Data ~ df$gender),data = df)

#logrank test-statistical check to see if there is any systematic pattern btw gender & recovery probability
survdiff(Recovery_Data ~ df$gender)

## What is the 1 month recovery rate? : % recovery after 30 days
#i.e what is the prob that a person will recover at 30 days
summary(survfit(Recovery_Data ~ 1), times = 30) 

#cox regression model and hazard ratio - helps us to evaluate how a varaible will have an effect on the survival rate
out2 = coxph(Surv(Recovery_Time, RecoveryStatus) ~ gender12, data = df)
out2

#Hazard Ratio Plot
ggforest(out2, data = df)


#KM plot for Age group
df = df[df$AgeBins!=0,]
plot(survfit(Recovery_Data ~ df$AgeBins),xlab = "Days",ylab = "Overall Recovery probability")
ggsurvplot(survfit(Recovery_Data ~ df$AgeBins),data = df)

#logrank test-statistical check to see if there is any systematic pattern btw gender & recovery probability
survdiff(Recovery_Data ~ df$AgeBins) 



#Trial -2 

## Survival Analysis (Event-Death) : Including Recoverd Cases as 120(infinity)
df = read_excel("DeathAndRecovery_Final.xlsx")
colnames(df)
head(df)

df$Survival_Time[1:10]
df$SurvivalStatus[1:10]

#Transform data to Survival Object
Survival_Data = Surv(df$Survival_Time, df$SurvivalStatus)
head(Survival_Data)
Survival_Data[130:180]

#KM plot for 1 group
plot(survfit(Survival_Data ~ 1),xlab = "Days",ylab = "Overall Survival Probability") 
out = survfit(Survival_Data ~ 1)
KMdata = cbind(out$time,out$n.event,out$n.censor,out$n.risk)
colnames(KMdata) = c("time","n.event","n.censor","n.risk")
View(KMdata)

#KM plot for two groups in Gender
plot(survfit(Survival_Data ~ df$gender),xlab = "Days",ylab = "Overall Survival Probability")
ggsurvplot(survfit(Survival_Data ~ df$gender),data = df)

#logrank test-statistical check to see if there is any systematic pattern btw gender & survival probability
survdiff(Survival_Data ~ df$gender) 

## What is the 1 month survival rate? : % survial after 30 days
#i.e what is the prob that a person will survive at 30 days
summary(survfit(Survival_Data ~ 1), times = 30)

#Cox regression model and hazard ratio - helps us to evaluate how a varaible will have an effect on the survival rate

out2 = coxph(Surv(Survival_Time, SurvivalStatus) ~ gender12, data = df)
out2

#Hazard Ratio Graph
ggforest(out2, data = df)


#KM plot for Age groups
df = df[df$AgeBins!=0,]
plot(survfit(Survival_Data ~ df$AgeBins),xlab = "Days",ylab = "Overall Survival probability")
ggsurvplot(survfit(Survival_Data ~ df$AgeBins),data = df)

#logrank test-statistical check to see if there is any systematic pattern btw gender & survival probability
survdiff(Survival_Data ~ df$AgeBins) 


## Recovery Analysis (Event-Recovered)Including Death Cases as 120(infinity)
df = read_excel("DeathAndRecovery_Final.xlsx")
df$Recovery_Time[1:10]
df$RecoveryStatus[1:10]
#Transform data to Survival Object
Recovery_Data = Surv(df$Recovery_Time, df$RecoveryStatus)
head(Recovery_Data)
Recovery_Data[140:150]

#KM plot for 1 group
plot(survfit(Recovery_Data ~ 1),xlab = "Days",ylab = "Overall Recovery Probability") 
out = survfit(Recovery_Data ~ 1)
KMdata = cbind(out$time,out$n.event,out$n.censor,out$n.risk)
colnames(KMdata) = c("time","n.event","n.censor","n.risk")
View(KMdata)

#KM plot for two groups in Gender
plot(survfit(Recovery_Data ~ df$gender),xlab = "Days",ylab = "Overall Recovery probability")
ggsurvplot(survfit(Recovery_Data ~ df$gender),data = df)

#logrank test-statistical check to see if there is any systematic pattern btw gender & recovery probability
survdiff(Recovery_Data ~ df$gender) 

## What is the 1 month survival rate? : % survial after 30 days
#i.e what is the prob that a person will survive at 30 days

summary(survfit(Recovery_Data ~ 1), times = 30)

#Cox regression model and hazard ratio - helps us to evaluate how a varaible will have an effect on the survival rate
out2 = coxph(Surv(Recovery_Time, RecoveryStatus) ~ gender12, data = df)
out2

#Hazard Ratio Graph
ggforest(out2, data = df)


#KM plot for Age groups
df = df[df$AgeBins!=0,]
plot(survfit(Recovery_Data ~ df$AgeBins),xlab = "Days",ylab = "Overall Recovery probability")
ggsurvplot(survfit(Recovery_Data ~ df$AgeBins),data = df)

#logrank test-statistical check to see if there is any systematic pattern btw gender & survival probability
survdiff(Recovery_Data ~ df$AgeBins) 

