# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2. Load the house dataset with features (house size and number of rooms) and target values (house price and number of occupants), then splits the data into       training and testing sets.
 
3.Scales the input features using StandardScaler and create an SGDRegressor with specified parameters.  

4. Train the model and calculate the Mean Squared Error.

5.Display the predicted and actual values.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:LAKSHMI PAVITHRA M 
RegisterNumber:212225220055

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("house.csv")
#print(data.columns)
data.columns = data.columns.str.strip()
# Features (inputs)
X = data[['Size', 'Bedrooms']]

# Targets (outputs)
y_price = data['Price']
y_occ = data['Occupants']

# Scaling (important for SGD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

# Train models
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

# Input
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

# Scale input
new_data = scaler.transform([[size, bed]])

# Prediction
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))

  
*/
```

## Output:

<img width="1439" height="747" alt="image" src="https://github.com/user-attachments/assets/833bb6b5-7d28-442b-852c-4da1d8afb154" />


<img width="1358" height="490" alt="image" src="https://github.com/user-attachments/assets/bd741462-2947-40e5-b323-1c82629beb14" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
