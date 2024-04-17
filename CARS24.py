#%%x
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load data
cars24 = pd.read_csv('/Users/jacobbrams/Library/Mobile Documents/com~apple~CloudDocs/UNI - 4.sem DTU/Data and Data Science/Projekt 3/data-science-projekt-3/cars_24_combined.csv')

#show data
cars24.head()
print(cars24.head())
#%%

#count which car names are most common
count = cars24['Car Name'].value_counts()
print(count)

#remove rows with missing values
cars24 = cars24.dropna()

#filter only containning mariuti baleno
cars24_mb = cars24[cars24['Car Name'].str.contains('Maruti Baleno')]
count_ms_baleno = cars24_mb['Car Name'].value_counts()
print(count_ms_baleno)
#%%x

#show the biggest difference in distance
cars24_mb_distance = cars24_mb['Distance'].max() - cars24_mb['Distance'].min()
print("Max difference in distance: ", cars24_mb_distance, "km")

#show the biggest price
cars24_mb_price = cars24_mb['Price'].max()
print("Highest price: ", cars24_mb_price, "EUR")
#%%x

#make summary statistics
cars24_mb.describe()
'''print(cars24_mb.describe())'''

start_date = cars24_mb['Year'].min()
end_date = cars24_mb['Year'].max()
print("Start date: ", start_date)
print("End date: ", end_date)
#%%x
#average distance, min and max distance
avg_distance = cars24_mb['Distance'].mean()
min_distance = cars24_mb['Distance'].min()
max_distance = cars24_mb['Distance'].max()
print("Average distance: ", avg_distance, "km")
print("Min distance: ", min_distance, "km")
print("Max distance: ", max_distance, "km")

#average price, min and max price
avg_price = cars24_mb['Price'].mean()
min_price = cars24_mb['Price'].min()
max_price = cars24_mb['Price'].max()
print("Average price: ", avg_price, "EUR")
print("Min price: ", min_price, "EUR")
print("Max price: ", max_price, "EUR")
#%%x

#Correlation calculation
cars24_mb_corr = cars24_mb.corr()
cars24_mb_corr

#Correlation matrix for Maruti Baleno
sns.heatmap(cars24_mb_corr, annot=True)
plt.show()
#save the plot
'''plt.savefig('/Users/jacobbrams/Library/Mobile Documents/com~apple~CloudDocs/UNI - 4.sem DTU/Data and Data Science/Projekt 3/Part II/Correlation_Matrix.png', dpi=300, bbox_inches='tight', pad_inches=0.5)'''
#scatter plot for price and distance
plt.scatter(cars24_mb['Price'], cars24_mb['Distance'])
plt.xlabel('Price')
plt.ylabel('Distance')
plt.title('Price vs Distance')
plt.show()
#save the plot
'''plt.savefig('/Users/jacobbrams/Library/Mobile Documents/com~apple~CloudDocs/UNI - 4.sem DTU/Data and Data Science/Projekt 3/Part II/Price_vs_Distance.png', dpi=300, bbox_inches='tight', pad_inches=0.5)'''
#%%x
# estimate a regression model for price as a function of distance, year and owner
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.iolib.summary2 import summary_col

#make th varibable Drive as a dummy variable
cars24_mb['Drive'] = cars24_mb['Drive'].astype('category')
cars24_mb['Drive'] = cars24_mb['Drive'].cat.codes

model1 = ols('Price ~ Distance + Year + Owner + Drive', data=cars24_mb).fit()
model2 = ols('Price ~ Distance * Year + Owner + Drive', data=cars24_mb).fit()
model3 = ols('Price ~ Distance + Year + Owner + Drive + Distance * Year', data=cars24_mb).fit()
model4 = ols('Price ~ Distance + Year + Owner + Drive + np.power(Distance, 2)', data=cars24_mb).fit()

# Summarize models using summary_col for a nicer output
results_table = summary_col([model1, model2, model3,model4], stars=True, float_format='%0.2f',
                            model_names=['Model 1: Linear', 'Model 2: Interaction', 'Model 3: Interaction + Linear', 'Model 4: Quadratic'],
                            info_dict={'R-squared': lambda x: "{:.2f}".format(x.rsquared),
                                       'No. observations': lambda x: "{0:d}".format(int(x.nobs))})

print(results_table)

'''#save the table in latex
latex_output = results_table.as_latex()

file_path = '/Users/jacobbrams/Library/Mobile Documents/com~apple~CloudDocs/UNI - 4.sem DTU/Data and Data Science/Projekt 3/Part II/Regression_Models.tex'
with open(file_path, 'w') as file:
    file.write(latex_output)'''
#calculate the elasticity of price with respect to distance

#Elasiticity of price with respect to distance
print('Elasticity of price with respect to distance')
print('Model 1 Elasiticity Price - Distance: ', model1.params['Distance'] * cars24_mb['Distance'].mean() / cars24_mb['Price'].mean())
print('Model 2 Elasiticity Price - Distance: ', model2.params['Distance'] * cars24_mb['Distance'].mean() / cars24_mb['Price'].mean())
print('Model 3 Elasiticity Price - Distance: ', model3.params['Distance'] * cars24_mb['Distance'].mean() / cars24_mb['Price'].mean())
print('Model 4 Elasiticity Price - Distance: ', model4.params['Distance'] * cars24_mb['Distance'].mean() / cars24_mb['Price'].mean())

#Check for autocorrelation
from statsmodels.stats.stattools import durbin_watson
durbin_watson(model1.resid)
durbin_watson(model2.resid)
durbin_watson(model3.resid)
durbin_watson(model4.resid)
print('Durbin-Watson test for autocorrelation')
print('Model 1 autocorrelation: ', durbin_watson(model1.resid))
print('Model 2 autocorrelation: ', durbin_watson(model2.resid))
print('Model 3 autocorrelation: ', durbin_watson(model3.resid))
print('Model 4 autocorrelation: ', durbin_watson(model4.resid))



'''#Check for heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
het_breuschpagan(model1.resid, model1.model.exog)
het_breuschpagan(model2.resid, model2.model.exog)
het_breuschpagan(model3.resid, model3.model.exog)
het_breuschpagan(model4.resid, model4.model.exog)

print('Breusch-Pagan test for heteroscedasticity')
print('Model 1: ', het_breuschpagan(model1.resid, model1.model.exog))
print('Model 2: ', het_breuschpagan(model2.resid, model2.model.exog))
print('Model 3: ', het_breuschpagan(model3.resid, model3.model.exog))
print('Model 4: ', het_breuschpagan(model4.resid, model4.model.exog))

#Check for normality
from scipy.stats import shapiro
shapiro(model1.resid)
shapiro(model2.resid)
shapiro(model3.resid)
shapiro(model4.resid)
print('Shapiro-Wilk test for normality')
print('Model 1: ', shapiro(model1.resid))
print('Model 2: ', shapiro(model2.resid))
print('Model 3: ', shapiro(model3.resid))
print('Model 4: ', shapiro(model4.resid))

'''


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




# %%
