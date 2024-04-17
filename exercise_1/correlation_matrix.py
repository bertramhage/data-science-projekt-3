import pandas as pd
import seaborn as sns

data = pd.read_csv('exercise_1/Caschool.csv')

data_cont = data.drop(columns=['distcod', 'county', 'district', 'grspan'])

correlation_matrix = data_cont.corr()
#print(correlation_matrix)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()