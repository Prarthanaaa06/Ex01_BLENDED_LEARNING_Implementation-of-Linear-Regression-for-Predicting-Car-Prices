# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Start**

2. **Import Libraries**

   * Import required libraries:
     `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `statsmodels`

3. **Load Dataset**

   * Read the dataset file `CarPrice_Assignment.csv` into a DataFrame.

4. **Select Features and Target**

   * Choose independent variables (features):
     `enginesize`, `horsepower`, `citympg`, `highwaympg`
   * Choose dependent variable (target):
     `price`

5. **Split Dataset**

   * Divide the dataset into:

     * Training set (80%)
     * Testing set (20%)
   * Use `train_test_split()` with `random_state=42`

6. **Feature Scaling**

   * Initialize `StandardScaler`
   * Fit scaler on training data
   * Transform both training and testing feature sets

7. **Train Model**

   * Initialize `LinearRegression` model
   * Fit the model using scaled training data (`X_train_scaled`, `Y_train`)

8. **Make Predictions**

   * Predict target values using test data (`X_test_scaled`)
   * Store predictions in `Y_pred`

9. **Display Model Coefficients**

   * Print:

     * Coefficients for each feature
     * Intercept value

10. **Evaluate Model Performance**

* Calculate and print:

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
  * R-squared score

11. **Check Linearity**

* Plot Actual vs Predicted values using scatter plot
* Draw a reference diagonal line

12. **Check Independence of Errors**

* Compute residuals:
  `residuals = Y_test - Y_pred`
* Calculate Durbin-Watson statistic
* Interpret value (≈2 indicates no autocorrelation)

13. **Check Homoscedasticity**

* Plot residuals vs predicted values using `residplot`
* Observe spread of residuals

14. **Check Normality of Residuals**

* Plot histogram with KDE of residuals
* Generate Q-Q plot using `statsmodels`

15. **End**


## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Prarthana D
RegisterNumber:  212225230213
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('CarPrice_Assignment.csv')

#Select features and target
X = df[['enginesize','horsepower','citympg','highwaympg']]
Y = df['price']

#Split target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

#Feature scaling becz it will be easier when evtg is in same format
scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Fit the X and Y in a straight line ie we are training the model
model = LinearRegression()
model.fit(X_train_scaled,Y_train)

#Predict the outcome by giving new set of data
Y_pred = model.predict(X_test_scaled)

#intersept ir beta knot values and coefficiets are being displayed
#model coefficient and metrics
print("Name: Prarthana D")
print("Reg. No: 25014010")
print("MODEL COEFFICIENTS: ")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:} : {coef:}")
print(f"{'Intercept':} : {model.intercept_:}")

#Performance metrics
print("MODEL PERFORMANCE:")
mse = mean_squared_error(Y_test,Y_pred)
print(f"MSE : {mse}")
print(f"MAE : {mean_absolute_error(Y_test,Y_pred)}")
print(f"RMSE : {np.sqrt(mse)}")
print(f"R-Squared : {r2_score(Y_test,Y_pred)}")

# 1. Linearity Check
#to check if evtg is in same format
plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred, alpha =0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()], 'r--')
plt.title("Lineraity Check : Actual VS Predicted Prices")
plt.xlabel("Acatual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals= Y_test - Y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\n Durbin-Watson Statistic: {dw_test:.2f}",
     "\n (values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10,5))
sns.residplot(x=Y_pred, y=residuals,lowess=True, line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals VS Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

#4. Normality of residuals
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals, kde= True, ax=ax1)
ax1.set_title("Resitduals Distribution")
sm.qqplot(residuals, line='45',fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")

```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

<img width="845" height="188" alt="image" src="https://github.com/user-attachments/assets/4fbf6ca3-3592-4523-8e49-be9caf99ba6e" />

<img width="769" height="135" alt="image" src="https://github.com/user-attachments/assets/00c8c910-4c14-49c1-a12f-8ce72f2a2b91" />

<img width="1229" height="598" alt="image" src="https://github.com/user-attachments/assets/fa9f9892-a687-4673-a722-30929dbd8b2c" />

<img width="1291" height="601" alt="image" src="https://github.com/user-attachments/assets/56509106-9667-406b-8338-3875e53a3167" />

<img width="1317" height="602" alt="image" src="https://github.com/user-attachments/assets/e360b705-02a2-4b8f-a44e-232efc29363f" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
