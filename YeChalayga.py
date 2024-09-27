import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk(r'C:\Users\aksha\OneDrive\Desktop\ml waaala'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

cp = pd.read_csv(r"C:\Users\aksha\OneDrive\Desktop\ml waaala\indiancrop_dataset.csv")
cp.head()
cp.info()
cp.duplicated().sum()
cp.isnull().sum()

cp.describe()
cp.value_counts()
print(cp.columns)

cp['CROP'].unique()
cp_dir = {
    'Rice':1,
    'Maize':2,
    'ChickPea':3,
    'KidneyBeans':4,
    'PigeonPeas':5,
    'MothBeans':6 ,
    'MungBean':7 ,
    'Blackgram':8,
    'Lentil':9,
    'Pomegranate':10,
    'Banana': 11,
    'Mango': 12,
    'Grapes': 13, 
    'Watermelon':14, 
    'Muskmelon':15,
    'Apple':16,
    'Orange':17,
    'Papaya':18,
    'Coconut':19, 
    'Cotton':20,
    'Jute':21,
    'Coffee':22
}

cp['CROP'] = cp['CROP'].map(cp_dir)
cp['CROP'].unique()

x = cp.drop('CROP', axis = 1)
y = cp['CROP']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state=42)
x_train.shape

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
ohe = OneHotEncoder(drop = 'first')
scaler = StandardScaler()
print(x_train.head(1))
preprocessor = ColumnTransformer(
    transformers = [
        ('encoder', ohe, [7]),
        ('stndrlstn', scaler,[0,1,2,3,4,5,6,8])
    ],
    remainder = 'passthrough'
)

x_train_dummy = preprocessor.fit_transform(x_train)
x_test_dummy = preprocessor.transform(x_test)

# from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


models = {
    # 'LR': LinearRegression(),
    'Decision Tree': DecisionTreeClassifier(),
}

for name, model in models.items():
    model.fit(x_train_dummy,y_train)
    y_pred = model.predict(x_test_dummy)
    score = accuracy_score(y_test,y_pred)
    print(f"{name} model with accuracy. {score}")

dct = DecisionTreeClassifier()
dct.fit(x_train_dummy, y_train)
y_pred = dct.predict(x_test_dummy)
accuracy_score(y_test, y_pred)

def prediction(N_SOIL,P_SOIL,K_SOIL,TEMPERATURE,HUMIDITY,ph,RAINFALL,STATE,CROP_PRICE):
    features = np.array([[N_SOIL,P_SOIL,K_SOIL,TEMPERATURE,HUMIDITY,ph,RAINFALL,STATE,CROP_PRICE]])
    scaled = preprocessor.transform(features)
    prediction_val = dct.predict(scaled).reshape(1,-1)
    return prediction_val[0]

N_SOIL = 0
P_SOIL = 23
K_SOIL = 15
TEMPERATURE = 22.56664172
HUMIDITY = 93.37488907
ph = 7.598729065
RAINFALL = 109.8585753
STATE = "Uttar Pradesh"
CROP_PRICE = 500

result = prediction(N_SOIL,P_SOIL,K_SOIL,TEMPERATURE,HUMIDITY,ph,RAINFALL,STATE,CROP_PRICE)
value_to_find = result
keys_with_value = [key for key, value in cp_dir.items() if value == value_to_find]
print(keys_with_value)

import pickle
pickle.dump(dct,open('w.pkl','wb'))
pickle.dump(preprocessor,open('a.pkl','wb'))