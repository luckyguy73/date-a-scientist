import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Create your df here:
df = pd.read_csv('profiles.csv')

print(df.describe())
print(df.head())
print(df['religion'].value_counts())

df.dropna(subset=['religion', 'diet', 'ethnicity', 'drinks', 'smokes', 'drugs'], inplace=True)

# df.religion = df.religion.fillna('other')
# df.diet = df.diet.fillna('other')
# df.ethnicity = df.ethnicity.fillna('other')

def split_col(data, index=0):
  output = str(data).split()
  return str(output[index]).strip(',')

df['diet_cat'] = df['diet'].apply(lambda x: split_col(x, -1))
df['religion_cat'] = df['religion'].apply(lambda x: split_col(x, 0))
df['eth_cat'] = df['ethnicity'].apply(lambda x: split_col(x, 0))

print(df['religion_cat'].value_counts())
print(df['diet_cat'].value_counts())
print(df['eth_cat'].value_counts())
print(df['drinks'].value_counts())
print(df['drugs'].value_counts())
print(df['smokes'].value_counts())

religion_mapping = {'other': 0, 'atheism': 1, 'agnosticism': 2,  'islam': 3, 'hinduism': 4, \
  'buddhism': 5, 'judaism': 6, 'catholicism': 7, 'christianity': 8}
diet_mapping = {'other': 0, 'halal': 1, 'kosher': 2, 'vegan': 3, 'vegetarian': 4, 'anything': 5}
eth_mapping = {'other': 0, 'native': 1, 'pacific': 2, 'middle': 3, 'indian': 4, 'black': 5, \
  'hispanic': 6, 'asian': 7, 'white': 8}
drinks_mapping = {'not at all': 0, 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4, \
  'desperately': 5}
drugs_mapping = {'never': 0, 'sometimes': 1, 'often': 2}
smokes_mapping = {'no': 0, 'sometimes': 1, 'when drinking': 2, 'trying to quit': 3, 'yes': 4}

df['religion_num'] = df.religion_cat.map(religion_mapping)
df['diet_num'] = df.diet_cat.map(diet_mapping)
df['eth_num'] = df.eth_cat.map(eth_mapping)
df['drinks_num'] = df.drinks.map(drinks_mapping)
df['drugs_num'] = df.drugs.map(drugs_mapping)
df['smokes_num'] = df.smokes.map(smokes_mapping)

print(df['religion_num'].value_counts())
print(df['diet_num'].value_counts())
print(df['eth_num'].value_counts())
print(df['drinks_num'].value_counts())
print(df['drugs_num'].value_counts())
print(df['smokes_num'].value_counts())



x = df[['diet_num', 'eth_num', 'drinks_num', 'drugs_num', 'smokes_num']]
y = df[['religion_num']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

print(lm.coef_)

y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Religion")
plt.ylabel("Predicted religion")
plt.title("Actual Religion vs Predicted Religion")

plt.show()