"""THIS CODE IS DONE ON PYCHARM
MANY FUNCTIONS HAVE BEEN COMMENTED
REMOVE THE # TO SEE OUTPUT OF EACH FUNCTION WHICH HAVE BEEN COMMENTED"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
var = pd.read_csv("Salary.csv")
# READING THE DATA
#print(var)

# CHECKING FOR NULL
#print(var.isnull().sum())  # no NULL found

# ANALYSING THE DATA
x = var.iloc[:, :-1]
y = var.iloc[:, -1:]
#sns.scatterplot(x) # just to checkout
#sns.scatterplot(y) # just to checkout
#sns.scatterplot(data = var, x = "YearsExperience", y = "Salary") # its a linear regression
#plt.show()

# DIVIDING INTO TRAINING AND TESTING
x1,x2,y1,y2 = train_test_split(x,y, test_size= 0.2, random_state= 2)
#print(x2) # printing to check what are getting tested and what are getting used for training

# Creating a Linear Regression Model and Fitting the Model to Data
model = LinearRegression()
print(model.fit(x1, y1))

exp = float(input("Enter experience: "))  # taking users input
print("Your salary will be around: ", model.predict([[exp]]))
print("\n")

# CHECKING THE PREDICTION
print("This is the prediction check.")
yp = model.predict(x2)
print(yp[:5])
print(y2[:5])
print("\n")

# CHECKING ACCURACY
print("This is the accuracy: ")
print(model.score(x2, y2)*100) # it's coming around 97 %



