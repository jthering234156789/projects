import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

flowerDataSet = pd.read_csv("IRIS.csv")
notSetosaTrain = flowerDataSet.query("species != 'Iris-setosa'").iloc[0::2]
notSetosaTest = flowerDataSet.query("species != 'Iris-setosa'").iloc[1::2]

x = notSetosaTrain[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = notSetosaTrain["species"]

xtest = notSetosaTest[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
ytest = notSetosaTest["species"]


params = {"C" : np.linspace(.1, 15, 20)}
classification = svm.SVC(kernel = 'linear')
gridSearch = GridSearchCV(classification, param_grid=params)
gridSearch.fit(x, y)
score_df = pd.DataFrame(gridSearch.cv_results_)
print(score_df[['param_C', 'mean_test_score', 'rank_test_score']])
classification = svm.SVC(C = gridSearch.best_params_["C"])
classification.fit(x, y)
print(classification.score(xtest, ytest))

confM = confusion_matrix(ytest, classification.predict(xtest))
disp = ConfusionMatrixDisplay(confM)

disp.plot()
plt.savefig("confMatrix.png")
plt.show()