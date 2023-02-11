# Fraud-Detection
This Project done under CodeClause Internship in August 2022



# Train a Perceptron Model without Feature Scaling
Here is the code for training a model without feature scaling. First and foremost, let’s load the dataset and create the dataset comprising of features and labels. In this post, the IRIS dataset has been used. In the below code, X is created as training data whose features are sepal length and petal length.


from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
Y = iris.target

Next step is to create the training and test split. The sklearn.model_selection module provides class train_test_split which couldbe used for creating the training / test split. Note that stratification is not used. 


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

Next step is to create an instance of Perceptron classifier and train the model using X_train and Y_train dataset / label. The code below uses Perceptron class of sklearn.linear_model module.

from sklearn.linear_model import Perceptron
 
prcptrn = Perceptron(eta0=0.1, random_state=1)
prcptrn.fit(X_train, Y_train)

Next step is to measure the model accuracy. This can be measured using the class accuracy_score of sklearn.metrics module or calling score method on the Perceptron instance. 


from sklearn.metrics import accuracy_score
Y_predict = prcptrn.predict(X_test)
print("Accuracy Score %.3f" %accuracy_score(Y_test, Y_predict))

The accuracy score comes out to be 0.578 


# Train a Perceptron Model with Feature Scaling
One does the feature scaling with the help of the following code. This step is followed just after creating training and test split.


from sklearn.preprocessing import StandardScaler
 
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

The above code represents StandardScaler class of sklearn.preprocessing module. The fit method of StandardScaler is used to estimate sample mean and standard deviation for each feature using training data. The transform method is then used to estimate the standardized value of features using those estimated parameters (mean & standard deviation).

The next step is to train a Perceptron model and measure the accuracy:


prcptrnFS = Perceptron(eta0=0.1, random_state=1)
prcptrnFS.fit(X_train_std, Y_train)
 
Y_predict_std = prcptrnFS.predict(X_test_std)
 
from sklearn.metrics import accuracy_score
print("Accuracy Score %0.3f" % accuracy_score(Y_test, Y_predict_std))
The accuracy score comes out to be 0.978.

You can note that the accuracy score increased by almost 40%.

# Sources 
https://vitalflux.com/python-improve-model-performance-using-feature-scaling/
https://en.wikipedia.org/wiki/Feature_scaling
https://towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35
https://medium.com/@draj0718/logistic-regression-with-standardscaler-from-the-scratch-ec01def674e8
