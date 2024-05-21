import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'/Users/yichen/Desktop/Aitagem/Admission_Predict.csv')
df = df.rename(columns={'Serial No.': 'Serial_No'})
df = df.rename(columns={'GRE Score': 'GRE_Score'})
df = df.rename(columns={'TOEFL Score': 'TOEFL_Score'})
df = df.rename(columns={'University Rating': 'University_Rating'})
df = df.rename(columns={'Chance of Admit ': 'Chance_Of_Admit'})
df = df.rename(columns={'LOR ': 'LOR'})
df = df.drop(['Serial_No'], axis=1)

# SPLIT DATA FOR MODEL SELECTION
X = df.drop(['Chance_Of_Admit'], axis=1)
y = df['Chance_Of_Admit']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)

# TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)
# Save the model
joblib.dump(model, 'Final_Model.pkl')