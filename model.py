import pandas as pd

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('House_Rent_Dataset.csv')
print(data)

data["Year"] = pd.to_datetime(data['Posted On']).dt.year
data['Month'] = pd.to_datetime(data['Posted On']).dt.month
data['Day'] = pd.to_datetime(data['Posted On']).dt.day

data.drop(['Posted On','Year'],axis=1,inplace=True)
data.head()

data[['Floor_Level','Total_Floor']] = data['Floor'].str.split(r'\s*out of\s*',expand=True)
data.head()

data['Floor_Level'] = data['Floor_Level'].replace('Ground',0)
data['Floor_Level'] = data['Floor_Level'].replace('Upper Basement',0)
data['Floor_Level'] = data['Floor_Level'].replace('Lower Basement',0)

data.fillna(method="ffill", inplace=True)

data['Floor_Level'] = data['Floor_Level'].astype(int)
data['Total_Floor'] = data['Total_Floor'].astype(int)

print(data[data['Floor_Level'] > data['Total_Floor']])

to_drop = data[data['Floor_Level'] > data['Total_Floor']].index

data.drop(to_drop,inplace = True)

data.drop('Floor',axis=1,inplace=True)

data.reset_index(inplace=True)

data.drop('index', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

columns = ['Area Type','Area Locality','City','Furnishing Status','Tenant Preferred','Point of Contact']
for i in columns:
    encoder = LabelEncoder()
    data[i] = encoder.fit_transform(data[i])

data.drop(['Tenant Preferred','Point of Contact'],axis=1,inplace=True)

import matplotlib.pyplot as plt

column_names = ['Rent','Size']
plt.figure(figsize=(6,5))
i = 1
for col in column_names:
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.subplot(1,2,i)
    i+=1
    plt.boxplot(data[col])
    plt.title(col)

import numpy as np

Q1 = data['Size'].quantile(0.25)
Q3 = data['Size'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
upper_index = np.where(data['Size']>=upper)[0]
lower_index = np.where(data['Size']<=lower)[0]
data.drop(index = upper_index,inplace=True)
data.drop(index = lower_index,inplace=True)
data.reset_index(inplace=True)

Q1 = data['Rent'].quantile(0.25)
Q3 = data['Rent'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
upper_index = np.where(data['Rent']>=upper)[0]
lower_index = np.where(data['Rent']<=lower)[0]
data.drop(index = upper_index,inplace=True)
data.drop(index = lower_index,inplace=True)
data.reset_index(inplace=True)

data.drop(['level_0','index'], axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler

x = data.drop('Rent',axis=1)
y = data['Rent']

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=5)

# Scaling X
x_scaler = StandardScaler()
xtrain = x_scaler.fit_transform(xtrain)
xtest = x_scaler.transform(xtest)

# Scaling y
y_scaler = StandardScaler()
ytrain_scaled = y_scaler.fit_transform(ytrain.values.reshape(-1, 1)).flatten()

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor()
gb_model.fit(xtrain,ytrain)

y_pred_scaled5 = gb_model.predict(xtest)
y_pred5 = y_scaler.inverse_transform(y_pred_scaled5.reshape(-1, 1)).flatten()
print(y_pred_scaled5)

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mean_absolute_error(ytest,y_pred_scaled5)

mean_squared_error(ytest,y_pred_scaled5)

r2_score(ytest,y_pred_scaled5)


import pickle
pickle.dump(gb_model,open('model.pkl','wb'))