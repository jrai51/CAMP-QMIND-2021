from io import SEEK_CUR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


df_train = pd.read_csv('titanic_train.csv') # reading train data into "df_train"
df_test = pd.read_csv('titanic_test.csv') # reading test data into "df_test"

X = df_train.drop(columns=["PassengerId","Name","Cabin", "Fare", "Ticket"]) #Clearing out data that has no apparent corellation on survival
y= df_train['Survived']

#supervised learning for binary classification problmes
#sklearn?

#Random Forest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))