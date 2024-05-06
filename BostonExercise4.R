# Load the required library
'install.packages("readr")
install.packages("tidyr")
install.packages("caret")
install.packages("broom")
install.packages("ggplot2")
install.packages("stargazer")
install.packages("dplyr")
install.packages("ISLR2")
install.packages("ggcorrplot")
install.packages("corrplot")'

library(ggcorrplot)
library(readr)
library(ISLR2)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(stargazer)
library(car)


data("Boston")  # Load the Boston dataset

# Create nox_dum variable
Boston$nox_dum <- ifelse(Boston$nox > 0.5, 1, 0)


# Summary statistics for nox_dum
summary(Boston$nox_dum)

# Correlation of nox_dum with other variables
cor(Boston[-which(names(Boston) == "nox")], Boston$nox_dum)

# Load the car package for vif function
library(car)

# Model with all other predictors
model <- lm(nox_dum ~ . - nox, data = Boston)  # Exclude crim as per project description

# Calculate VIF
vif(model)

# Load the MASS package for logistic regression
library(MASS)

# Initial logistic regression model
logit_model <- glm(nox_dum ~ zn + indus + chas + rm + age + dis + rad + tax + ptratio + lstat + medv, family = binomial, data = Boston)
summary(logit_model)

vif(logit_model)

stargazer(logit_model, type = "latex", title = "Logistic Regression Results", out = "model_output.txt")

# Regression model where we remove variables with high vif

logit_model_revised <- glm(nox_dum ~ zn + indus + chas + age + dis + rad + tax + ptratio + lstat, family = binomial, data = Boston)
summary(logit_model_revised)

vif(logit_model_revised)


logit_model_revised_final <- glm(nox_dum ~ zn + indus + chas + age + dis + rad + tax, family = binomial, data = Boston)
summary(logit_model_revised_final)
