'# Load the required library
install.packages("readr")
install.packages("tidyr")
install.packages("caret")
install.packages("broom")'
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(stargazer)
# Set up the dataframe
df <- read_csv('exercise_1/Caschool.csv')

data_selected <- select(df, mathscr , everything())

print(colnames(data_selected))

# Show the first 5 rows of the dataframe
head(df)

# Make summary of the main statistics of the dataframe
summary(df)

# Estimate the linear regression model
X <- df$avginc
Y <- df$mathscr
regr <- lm(Y ~ X)

# Print a tidy summary of the model
stargazer(regr, type = "text")

confint(regr, level = 0.9)

summary(regr)

# Visualizing the relationship along with the regression line
ggplot(df, aes(x = avginc, y = mathscr)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Relationship between Average Income and Math Score",
       x = "Average Income",
       y = "Math Score") +
  theme_minimal()




