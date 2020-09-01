#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing the dataset
dataset=pd.read_excel('D:\predictive_maintenance\Boiler_Data\Book1.4.xlsx')

#Separating the Input and output from the datasets
X=dataset.iloc[:,[1,2]].values
y=dataset.iloc[:, 3].values


#splitting dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=0)

#Feature Scaling(Data preprocessing)
from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()
X_train=std_scaler.fit_transform(X_train)
X_test=std_scaler.fit_transform(X_test)

#De normalizing to original data
x_train_denorm =std_scaler.inverse_transform(X_train)

x_test_denorm=std_scaler.inverse_transform(X_test)


#Fitting Logistic regression to the Training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)

#predicting the test set results
y_pred = classifier.predict(X_test)
y_pred1 = classifier.predict([[180,14]])

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#visualizing the training set graphical results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X_set1, y_set1 = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('white', 'green')))

plt.xticks()
plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i,j in enumerate((np.unique(y_set))):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('blue', 'green'))(i), label = j)



plt.title('Support Vector Classifier (Training Set)')

plt.xlabel('temperature')

plt.ylabel('pressure')

plt.legend()

plt.show()


#Test set

X1, X2 = np.meshgrid(np.arange(start = X_set1[:, 0].min() - 1, stop = X_set1[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set1[:, 1].min() - 1, stop = X_set1[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('white', 'green')))

plt.xticks()
plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i,j in enumerate((np.unique(y_set1))):

    plt.scatter(X_set1[y_set1 == j, 0], X_set1[y_set1 == j, 1],

                c = ListedColormap(('blue', 'green'))(i), label = j)



plt.title('Support Vector Classifier (Test Set)')

plt.xlabel('temperature')

plt.ylabel('pressure')

plt.legend()

plt.show()

from sklearn.metrics import accuracy_score
z=accuracy_score(y_test,y_pred)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
acccuracy = metrics.accuracy_score(y_test, y_pred)

