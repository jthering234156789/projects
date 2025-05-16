import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset/data
df = pandas.read_csv("Crop_recommendation.csv")

#grab only the first 6 columns
x = df.iloc[:, :7]

#cast as type string just to be safe
y = df.iloc[:, 7].astype(str)

#split data with 20 percent of it being testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#create paramater for grid search
max_depth = np.linspace(5, 20, 15).astype(int)
paramaters = {'max_depth':max_depth}

rfc = RandomForestClassifier()
grsF = GridSearchCV(rfc, paramaters)

#after running these line, and seeing the best paramater we get 17.
#grsF.fit(X_train, y_train)
#print(grsF.score(X_test, y_test))
#print(grsF.best_params_)


#create paramater for grid search
svc = SVC(kernel = "linear")
bag = BaggingClassifier(svc)
baggingParamaters = {'n_estimators':np.linspace(10, 50, 15).astype(int)}
grsB = GridSearchCV(bag, baggingParamaters)

#after running these line, and seeing the best paramater we get 35.
#grsB.fit(X_train, y_train)
#print(grsB.score(X_test, y_test))
#print(grsB.best_params_)

#create 2 different classifiers with found best paramaters.
newRFC = RandomForestClassifier(max_depth = 17, oob_score=True)
newBAG = BaggingClassifier(SVC(kernel='linear'), n_estimators= 35, oob_score=True)
#fit the classifiers, make the confusion matrices and save the plot.
newRFC.fit(X_train, y_train)
rfcMatix = confusion_matrix(y_test, newRFC.predict(X_test))
rfcDisp = ConfusionMatrixDisplay(rfcMatix)
rfcDisp.plot()
plt.savefig("rfcConfMatrix.png")
newBAG.fit(X_train, y_train)
bagMatix = confusion_matrix(y_test, newBAG.predict(X_test))
bagDisp = ConfusionMatrixDisplay(bagMatix)
bagDisp.plot()
plt.savefig("bagConfMatrix.png")

#print the OOB score and normal score for each method.
print(newRFC.oob_score_)
print(newRFC.score(X_test, y_test))
print(newBAG.oob_score_)
print(newBAG.score(X_test, y_test))
