import numpy as np
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

#take in out file
drugDataSet = pd.read_csv("files/drug200.csv")

#set non-number columns to numbers
drugDataSet['sexToInt'] = 1.0
drugDataSet['BPtoFloat'] = 0.0
drugDataSet['CholestFloat'] = .5
for i in range(len(drugDataSet["Sex"])):
    if drugDataSet['Sex'][i] == "M":
        drugDataSet.iloc[i, drugDataSet.columns.get_loc('sexToInt')] = 0
    if drugDataSet['BP'][i] == "NORMAL":
        drugDataSet.iloc[i, drugDataSet.columns.get_loc('BPtoFloat')] = .5
    elif drugDataSet['BP'][i] == "HIGH":
        drugDataSet.iloc[i, drugDataSet.columns.get_loc('BPtoFloat')] = 1
    if drugDataSet['Cholesterol'][i] == "High":
        drugDataSet.iloc[i, drugDataSet.columns.get_loc('CholestFloat')] = 1

#seperate between train and test sets
Train = drugDataSet.iloc[:150]
Test = drugDataSet.iloc[150:]

#normalize values
x = Train[["Age","sexToInt","BPtoFloat","CholestFloat","Na_to_K"]]
x -= np.average(x, axis = 0)
x /= np.std(x, axis = 0)+1
y = Train["Drug"]

xtest = Test[["Age","sexToInt","BPtoFloat","CholestFloat","Na_to_K"]]
xtest -= np.average(xtest, axis = 0)
xtest /= np.std(xtest, axis = 0)+1
ytest = Test["Drug"]

#set params dict for
paramsC = {"C" : np.linspace(.1, 40, 30, True)}
paramsG = {"gamma" : np.linspace(.1, 40, 30, True)}

#find best value for C
classificationC = svm.SVC(kernel='rbf')
gridSearchC = GridSearchCV(classificationC, param_grid=paramsC)
gridSearchC.fit(x, y)
score_dfC = pd.DataFrame(gridSearchC.cv_results_)
#print(score_dfC[['param_C', 'mean_test_score', 'rank_test_score']])

#find best value for gamma with optimal C
classificationG = svm.SVC(kernel='rbf')
gridSearchG = GridSearchCV(classificationG, param_grid=paramsG)
gridSearchG.fit(x, y)
score_dfG = pd.DataFrame(gridSearchG.cv_results_)
#print(score_dfG[['param_gamma', 'mean_test_score', 'rank_test_score']])

#make svc with optimal values
classification = svm.SVC(C = gridSearchC.best_params_["C"], gamma=gridSearchG.best_params_["gamma"])
classification.fit(x, y)
#print(classification.score(xtest, ytest))
confM = confusion_matrix(ytest, classification.predict(xtest))

#testing the svc using the first entry in x
input = np.array([23, 1.0, 1.0, 1.0, 25.35]).reshape(1, -1)
output = classification.predict(input)
print(f"for the input {input}, the svc says the output should be {output}.")
print(f"the actual output should have been {y.iloc[0]}")


print("displaying confusion matrix")
ypred = classification.predict(xtest)
cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()