from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from data_scaling import scale
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


linear_svc = SVC(kernel="linear")
poly_svc = SVC(kernel="poly")
rbf_svc = SVC(kernel="rbf")


def holdout_estimation(X,y, col1, col2, models=[linear_svc, poly_svc, rbf_svc],ts=0.3,seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=ts,random_state=seed)
    # scale(X_train, X_test)
    for model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print(f"{model.kernel} kernel\nAccuracy: {accuracy_score(y_test,y_pred):.3f}")
        plot_decisionBound(model,X_train,y_train, col1, col2)


def cv_estimation(model,X,y,k=10):
    val_scores = cross_val_score(model, X, y, cv=k)
    return val_scores

def plot_decisionBound(model,X,y, col1, col2):
    disp = DecisionBoundaryDisplay.from_estimator(
    model, X, response_method="predict",
    alpha=0.5)
    disp.ax_.scatter(X[col1], X[col2], c=y, edgecolor="k")
    plt.show()

def check_params():
    standard_classifier = SVC()
    params = standard_classifier.get_params()
    print(params)

def tuning(X, y, col1, col2, ts = 0.3, seed = 0):
    
    svc_params = [
    {'kernel':['poly'],
    'degree': [1, 2, 3, 4],
    'C': [0.001, 0.1, 1, 10, 100, 1000]},
    {'kernel':['rbf'],
    'gamma': ['auto', 'scale'],
    'C': [0.001, 0.1, 1, 10, 100, 1000]}
    ]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=ts,random_state=seed)

    grid_search = GridSearchCV(SVC(), svc_params)
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    prediction = best_estimator.predict(X_test)
    print("Chosen parameters:", grid_search.best_params_)
    print(f"Accuracy: {accuracy_score(prediction, y_test):.3f}")
    plot_decisionBound(best_estimator, X_train, y_train, col1, col2)

    
