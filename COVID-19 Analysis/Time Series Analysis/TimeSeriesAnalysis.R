rm(list = ls())
graphics.off()
require(astsa)
library(readxl)
setwd("C:/Users/nimee/Documents/MSA Spring 2020/8200-Predictive Analytics/FinalProject")

#No of Confirmed Cases

Confirmed_Data <- read_excel("NoofConfirmedTimeSeriesData.xlsx")
Confirmed_Data_TS<-ts(Confirmed_Data$NoOfConfirmedCases)
plot(Confirmed_Data_TS)
log_TS<-log(Confirmed_Data_TS)
plot(log_TS)
diff_log_TS<-diff(log_TS)
plot(diff_log_TS)

acf2(diff_log_TS)
arima(diff_log_TS, order = c(0,0,0))
sarima(diff_log_TS,0,0,0)

# SARIMA(0,0,0) looks like the best fit since it has the lowest AIC value = 2.24

# Predicting for the next 1 week
predict <- predict(arima(Confirmed_Data_TS, order = c(0,0,0)), n.ahead = 7)

ts.plot(Confirmed_Data_TS,predict$pred,col=1:2,xlim=c(1,60),ylab="NumberOfConfirmedCases")

# Prediction is tending to average too early because past data is available only for 60 days

#No of Deaths

Death_Data <- read_excel("NoofDeathsTimeSeriesData.xlsx")
Death_Data_TS<-ts(Death_Data$NoOfDeaths)
plot(Death_Data_TS)
log_TS<-log(Death_Data_TS)
plot(log_TS)
diff_log_TS<-diff(log_TS)
plot(diff_log_TS)

acf2(diff_log_TS)
arima(diff_log_TS, order = c(0,0,0))
sarima(diff_log_TS,0,0,0)

# ARIMA(0,0,0) looks like the best fit since it has the lowest AIC value = 2.24

# Predicting for the next 1 week
predict <- predict(arima(Death_Data_TS, order = c(0,0,0)), n.ahead = 7)

ts.plot(Death_Data_TS,predict$pred,col=1:2,xlim=c(1,40),ylab="NumberOfDeaths")

# Prediction is tending to average too early because past data is available only for 60 days


#No of Recovered Cases

Recovery_Data <- read_excel("NoofRecoveriesTimeSeriesData.xlsx")
Recovery_Data_TS<-ts(Recovery_Data$NoOfRecoveries)
plot(Recovery_Data_TS)
log_TS<-log(Recovery_Data_TS)
plot(log_TS)
diff_log_TS<-diff(log_TS)
plot(diff_log_TS)

acf2(diff_log_TS)
arima(diff_log_TS, order = c(1,0,1))
sarima(diff_log_TS,1,0,1)

# ARIMA(1,0,1) looks like the best fit since it has the lowest AIC value = 2.24

# Predicting for the next 1 week
predict <- predict(arima(Recovery_Data_TS, order = c(1,0,1)), n.ahead = 7)

ts.plot(Recovery_Data_TS,predict$pred,col=1:2,xlim=c(1,45),ylab="NumberOfRecoveredCases")

# Prediction is increasing initially as the number of recovered cases are more and then
# tending to average too early because past data is available only for 60 days
