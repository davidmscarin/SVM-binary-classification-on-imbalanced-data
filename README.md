# SVM algorithm documentation

## Auxiliary Functions

In the beginning of the python file you'll find a few defined functions before the algorithm's class structure.
They are functions implemented to help in the evaluation, execution and preparation during the various phases of the procedure's pipeline. 
    * The houdlout_estimation and plot_decision_bound incorporated with sklearn methods, automate sklearn's SVC execution (which will be used as a reference to compare our implementation's performance). 
    * The following four functions make the scaffold to our algorithms procedure. This include creating the target matrix to store the classification values, normalization of output and, most importantly, the pipeline itself, which splits the train and test sets, feeds the SVM training and returns the prediction as well as the test set targets for further assessment of the prediction.
    * There's also a score function that measures precision and a plot function to visualize the SVM's prediction.

## SVM Algorithm

The implementation of the SVM algorithm we selected for our assignment was the code respective to Stephen Marsland's book *Machine Learning: An Algorithmic Pesrpective* version of the SVM existing in the author's website, found on the book's prologue. This implementation has two types of kernel functions: Polynomial and Radial Basis Function (three considering the Linear apart from the Polynomial). It uses the *cvxopt* library to solve the quadratic programming problem of convex in order to find the support vectors (the data points that satisfy the defined constraints).
This version of the SVM algorithm (as you may observe in the authors execution examples in the already mentioned website) supports binary and multiclass classification, even though there are major changes needed in the input and output matrices normalization the which are automated by our auxiliary functions.

Note: the majority of commented code in class structure was left out by the author which we didn't remove nor change. We still added comments to identify each phase of the procedure.

## Our modification to the SVM

As we've already mentioned, there are inbuilt functions to normalize the algorithm's input and output matrices. These, as written above, wouldn't be needed for binary classification but in order to apply the proposed changes to the algorithm we made it universal to binary or multiclass, adapting binary to the workings of multiclass.

The alteration we set out to do is the option to apply C values to each class instead of a global C value.
To enable this feature, which is turned off by default, pass the ordered C array as the value of the *c_* parameter in the *sequence* function and also *wc* as True. The latter is very importante, otherwise the weighting won't apply.

## Input and Output

Note that the execution of the algorithm must be called through the *sequence* function that handles all the process from there.
The input expected is the desired dataframe with the last column as the class attribute in form of a numpy array.
The output will return the predict and test target arrays respectively. These are returned unformated, and one must call the *getPut* function to normalize the matrices to the standards.