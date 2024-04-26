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


data("Boston")  # Load the Boston dataset

# Create the nox_dum variable
Boston$nox_dum <- ifelse(Boston$nox > 0.5, 1, 0)

# View the first few rows to check the dummy variable
head(Boston)

# Summary statistics for nox_dum
summary(Boston$nox_dum)

# Assuming cor_matrix is already calculated
cor_matrix <- cor(Boston[, -which(names(Boston) == "nox_dum")], use = "pairwise.complete.obs")

# Plot the correlation matrix using ggcorrplot
ggcorrplot(cor_matrix, method = "square", 
           lab = TRUE,  # Show correlation coefficients
           type = "full",  # Display the lower triangle of the correlation matrix
           lab_size = 3,  # Adjust the size of the correlation coefficients
           colors = c("#6D9EC1", "white", "#E46726"),  # Colors: blue to white to red
           title = "Correlation Matrix of Boston Dataset Variables",
           ggtheme = theme_minimal())  # Use a minimal theme

# Fit logistic regression model
model <- glm(nox_dum ~ . - crim, data = Boston, family = binomial)

summary(model)
# Load stargazer for creating nice tables
library(stargazer)

# Display the model output using stargazer
stargazer(model, type = "text")


# Setting up the plotting area to accommodate multiple plots
par(mfrow=c(1, 2))  # Adjust the layout based on the number of variables

# List of variable names to plot
variable_names <- names(Boston)

# Loop through each variable in the dataset
for (var in variable_names) {
  # Create a boxplot for each variable
  boxplot(Boston[[var]], main=var)
}
