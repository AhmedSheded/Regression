import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

dataset = pd.read_csv('/home/sheded/PycharmProjects/SAT degrees/satf.csv')

X = dataset.iloc[:, :1]
y = dataset.iloc[:, -1]

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fitting simple linear regression to the training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))

y_pred = regressor.predict(X_test)

print('y pred is ', y_pred[:5])
print("y test is ", y_test[:5])

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(median_absolute_error(y_test, y_pred))

# visualising the training set results

plt.scatter(X_train, y_train, color= 'r')
plt.scatter(X_test, y_test, color='g')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.title('SAT degrees')
plt.xlabel('high_GPA')
plt.ylabel('univ_GPA')
plt.show()