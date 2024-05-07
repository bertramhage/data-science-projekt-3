#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.iolib.summary2 import summary_col

# Load data
cars24 = pd.read_csv('/Users/jacobbrams/Library/Mobile Documents/com~apple~CloudDocs/UNI - 4.sem DTU/Data and Data Science/Projekt 3/data-science-projekt-3/cars_24_combined.csv')

# Drop the 'Unnamed: 0' column if it exists
cars24.drop(columns=[col for col in cars24.columns if 'Unnamed' in col], inplace=True, errors='ignore')

# Remove rows with missing values
cars24.dropna(inplace=True)

# Filtering for 'Maruti Baleno'
cars24_mb = cars24[cars24['Car Name'].str.contains('Maruti Baleno')]

# Standardization of numerical columns + year should be the age of the car
numeric_cols = cars24_mb.select_dtypes(include=[np.number])
for col in numeric_cols:
    cars24_mb[col] = (cars24_mb[col] - cars24_mb[col].mean()) / cars24_mb[col].std()

# Convert 'Drive' into a dummy variable
cars24_mb['Drive'] = cars24_mb['Drive'].map({'Manual': 0, 'Automatic': 1})

#Year should be the age of the car
cars24_mb['Year'] = 2024 - cars24_mb['Year']

# Define models
model1 = ols('Price ~ Distance + Year + Owner + Drive', data=cars24_mb).fit()
model2 = ols('Price ~ Distance * Year + Owner + Drive', data=cars24_mb).fit()
model3 = ols('Price ~ Distance + Year + Owner + Drive + Distance * Year', data=cars24_mb).fit()
model4 = ols('Price ~ Distance + Year + Owner + Drive + np.power(Year, 2)', data=cars24_mb).fit()

# Summarize models
results_table = summary_col([model1, model2, model3, model4], stars=True, float_format='%0.2f',
                            model_names=['Model 1: Linear', 'Model 2: Interaction', 'Model 3: Interaction + Linear', 'Model 4: Quadratic'],
                            info_dict={'R-squared': lambda x: "{:.2f}".format(x.rsquared),
                                       'No. observations': lambda x: "{0:d}".format(int(x.nobs))})

print(results_table)


# %%
#make plots for the regression models
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

#plot for model 1
sns.regplot(x='Distance', y='Price', data=cars24_mb, ax=ax[0, 0])
ax[0, 0].set_title('Model 1: Linear')
ax[0, 0].set_xlabel('Distance')
ax[0, 0].set_ylabel('Price')

#plot for model 2
sns.regplot(x='Distance', y='Price', data=cars24_mb, ax=ax[0, 1])
ax[0, 1].set_title('Model 2: Interaction')
ax[0, 1].set_xlabel('Distance')
ax[0, 1].set_ylabel('Price')

#plot for model 3
sns.regplot(x='Distance', y='Price', data=cars24_mb, ax=ax[1, 0])
ax[1, 0].set_title('Model 3: Interaction + Linear')
ax[1, 0].set_xlabel('Distance')
ax[1, 0].set_ylabel('Price')

#plot for model 4
sns.regplot(x='Distance', y='Price', data=cars24_mb, ax=ax[1, 1])
ax[1, 1].set_title('Model 4: Quadratic')
ax[1, 1].set_xlabel('Distance')
ax[1, 1].set_ylabel('Price')

plt.tight_layout()
plt.show()


#%%
#change the name of Drive from 0 and 1 to Manual and Automatic
cars24_mb['Drive'] = cars24_mb['Drive'].replace(0, 'Manual')
cars24_mb['Drive'] = cars24_mb['Drive'].replace(1, 'Automatic')

#make a t-test for the difference in price between manual and automatic cars
from scipy.stats import ttest_ind
manual = cars24_mb[cars24_mb['Drive'] == 'Manual']
automatic = cars24_mb[cars24_mb['Drive'] == 'Automatic']

ttest = ttest_ind(manual['Price'], automatic['Price'])

print(ttest)
#make a boxplot for the price of manual and automatic cars with a legend for the drive type orange and blue

sns.boxplot(x='Drive', y='Price', data=cars24_mb, linewidth=2.5, palette='Set2', showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
plt.title('Price for Manual and Automatic cars')
plt.xlabel('Drive')
plt.ylabel('Price')
plt.show()

# %%

# Assuming 'Distance' is km driven and 'Year' represents the car age calculated from the current year
cars24_mb['Year'] = 2024 - cars24_mb['Year']  # Replace 2024 with the current year if needed
X = cars24_mb[['Distance', 'Year']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = cars24_mb['Price']

model = sm.OLS(y, X).fit()
print(model.summary())

#%%
# Calculate elasticity of price with respect to km and make a table for the elasticity of price with respect to distance for each model
elasticity_km = model4.params['Distance'] * (cars24_mb['Distance'].mean() / cars24_mb['Price'].mean())
print(f"Elasticity of price with respect to km: {elasticity_km}")

# Calculate marginal effect of car age
marginal_effect_age = model1.params['Year']
print(f"Marginal effect of car age: {marginal_effect_age}")


#make a table with eleasticity of price given distance for each of the models  
elasticity = pd.DataFrame({'Model': ['Model 1', 'Model 2', 'Model 3', 'Model 4'],
                           'Elasticity of Price with respect to Distance': [model1.params['Distance'], model2.params['Distance'], model3.params['Distance'], model4.params['Distance']]})

print(elasticity)
# %%
# Model diagnostic plots
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Residuals vs Fitted
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax[0], line_kws={'color': 'red', 'lw': 1})
ax[0].set_title('Residuals vs Fitted')
ax[0].set_xlabel('Fitted values')
ax[0].set_ylabel('Residuals')
ax[0].axhline(y=0, color='black', linewidth=1)
ax[0].legend(['Residuals', 'Fitted values'])

# QQ Plot
sm.qqplot(model.resid, line='45', fit=True, ax=ax[1])
ax[1].set_title('Normal Q-Q')

plt.tight_layout()
plt.legend(['Residuals', 'Fitted values'])
plt.show()

# Influential Variables
standardized_coeffs = model.params / model.bse
print("Standardized coefficients:\n", standardized_coeffs)

# Discussion of Influential Variables
print("\nDiscussion: ")
print(f"Distance has a standardized coefficient of {standardized_coeffs['Distance']:.3f}, indicating {'stronger' if abs(standardized_coeffs['Distance']) > abs(standardized_coeffs['Year']) else 'weaker'} influence on price compared to car age.")
print(f"Car age has a standardized coefficient of {standardized_coeffs['Year']:.3f}, which shows its {'significant' if standardized_coeffs['Year'] >= 2 or standardized_coeffs['Year'] <= -2 else 'less significant'} impact on the price.")

# Discussing Uncertainties and Limitations
print("\nModel Uncertainties and Limitations:")
print("1. Potential multicollinearity between Distance and Year if older cars typically have higher km.")
print("2. Model assumes linear relationships; real-world relationships might be non-linear or require transformation.")
print("3. Presence of high leverage points or outliers could distort the model, as seen if Cook's distance is significant for any observation.")

#%%
#check for heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
_, p_value, _, _ = het_breuschpagan(model1.resid, X)
print(f"P-value for Breusch-Pagan test: {p_value}")



# %%
# Antag at 'cars24_mb' er dit dataframe, som allerede er indlæst og forberedt
average_distance = cars24_mb['Distance'].mean()
average_price = cars24_mb['Price'].mean()

# Beregn elasticiteten med koefficienten fra Model 1
beta_distance = -0.14  # Fra Model 1: Linear
elasticity_price_distance = beta_distance * (average_distance / average_price)

print("Gennemsnitlig Distance:", average_distance)
print("Gennemsnitlig Pris:", average_price)
print("Price Elasticity of Distance:", elasticity_price_distance)
# %%
def calculate_elasticity(model, data):
    beta_distance = model.params['Distance']
    average_distance = data['Distance'].mean()
    average_price = data['Price'].mean()
    elasticity = beta_distance * (average_distance / average_price)
    return elasticity

# Assuming cars24_mb is your dataset after filtering for 'Maruti Baleno' or whatever your criteria was
elasticity_model1 = calculate_elasticity(model1, cars24_mb)
elasticity_model2 = calculate_elasticity(model2, cars24_mb)
elasticity_model3 = calculate_elasticity(model3, cars24_mb)
elasticity_model4 = calculate_elasticity(model4, cars24_mb)

# Create a DataFrame to display the elasticities for each model
elasticities = {
    "Model": ["Model 1", "Model 2", "Model 3", "Model 4"],
    "Elasticity of Price w.r.t Km": [
        elasticity_model1, 
        elasticity_model2, 
        elasticity_model3, 
        elasticity_model4
    ]
}

elasticity_table = pd.DataFrame(elasticities)
print(elasticity_table)
# %%

# %%
