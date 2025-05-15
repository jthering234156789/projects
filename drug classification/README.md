# Drug Classification Project
Classifying different types of drugs people take based on factors about their body(age, sex, blood pressure, chol., NA to K).
## Method
A support vector machine will be used with a radial basis function to predict a given drug.
## Data
All columns that are not numbers are turned into a number.
Sex: if male, its set to 0, otherwise 1

BP: 0 for low, .5 for medium, 1 for high

Chol.: 0 for low, 1 for high.
## results
based on our 50 test entries, we are 94% accurate with our predictions.

![image](https://github.com/jthering234156789/projects/blob/main/drug%20classification/confMatrix.png)
