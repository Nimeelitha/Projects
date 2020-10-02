library(lme4)
install.packages("lmerTest")
library(lmerTest)
install.packages("dta")
library(dta)
data = read.csv("C:/Users/keert/Desktop/Semester 2/Predictive/final/new.csv")
View(data)
names(data)

plot(data)
str(data)

#modelrand = lmer(death ~ country + (1|country), data=data)
#summary(modelrand)
#rand(model)


lm_country = lm(death~gender+visiting.Wuhan+from.Wuhan+factor(country)-1, 
                data=data)
summary(lm_country)

lmer_country_only = lmer(death ~ 1+(1|country)-1, data=data)
summary(lmer_country_only)
rand(lmer_country_only)

summary(lmer(death ~ age + visiting.Wuhan + from.Wuhan + (1|country)-1, data=data))
summary(model_country)
rand(model_country)

##Age level effects

lm_age = lm(death~gender+visiting.Wuhan+from.Wuhan+factor(age)-1, data=data)
summary(lm_age)
rand(lm_age)

lmer_age_only = lmer(death ~ 1+(1|age), data=data)
summary(lmer_age_only)
rand(lmer_age_only)

model_age = lmer(death ~ gender+ visiting.Wuhan + from.Wuhan + (1|age)-1+(1|country), data=data)
summary(model_age)
rand(model_age)


