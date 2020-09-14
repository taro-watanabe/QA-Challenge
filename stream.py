# Importing Modules/Libraries

import streamlit as st
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from mpl_toolkits.mplot3d import Axes3D


st.title("QA Pipeline Challenge")

# Obtaining Shared Dataset

def obtain_list(port, limit):  ### Obtaining the whole dataset
    if type(port) != str:
        raise Exception('port must be an url expresseed with a string, page unspecified.')

    if type(limit) != int or limit < 0:
        raise Exception('limit has to be a positive integer.')

    data = []

    for i in range(limit):
        data.append(requests.get(port + '?page={}'.format(i)).json())

    return data

# Conversion to dataframe & Train/Test Split

# Options for choosing less amount of data for faster processing. (Choosing the "first" ~%, thus have vulnerbilities.)
ratio = st.selectbox("Choose how much ratio of the dataset to extract. Smaller ratio means faster process, Bigger ratio may take a TON of time.", ("1%", "10%", "50%", "100%"))

if ratio == "1%":
    r = 0.01
if ratio == "10%":
    r = 0.1
if ratio == "50%":
    r = 0.5
if ratio == "100%":
    r = 1

dataset = sum(obtain_list('http://localhost:5000/api/v1/data', int(100000*r)), [])
df = pd.DataFrame(dataset)
X = df.drop(["promoted", "id"], axis=1)
y = df["promoted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Mode Selector

mode = st.selectbox("Select the topic you desire", ("Task 1: Classification Algorithm", "Task 2: Regression Models w/ endogenous variable"))
if mode == "Task 1: Classification Algorithm":

    st.write("""## Task 1: Classification Algorithm""")

    submode = st.selectbox("Select the mode you desire", ("Live Interactive Manual Algorithm & Parameter Editing", "Pipeline Auto-Comparison (Hyperparameters Defaulted)"))

    if submode == "Live Interactive Manual Algorithm & Parameter Editing":
        st.write("""
        ### Notice: \n
        This is a Web-app interactive version (both reduced and added) from the full ipynb file. For thought processes and explanations, please refer to the corresponding ipynb file.""")
        st.write("""## Live Manual Algorithm & Parameter Editing 
        ### Please Select the Classification Algorithm, and their parameters""")

        alg_name = st.selectbox("Select Classifier to use", ("Ridge Classification", "Logistic Regression","Support Vector Classification", "K-Nearest Neighbors Classfication", "Decision Tree Classification"))

        def add_parameter(alg_name):
            param = dict()
            if alg_name == "Ridge Classification":
                param["alpha"] = st.slider("Choose an alpha parameter. Default:1.0", 0.0000001, 20.0)

                fit_int = st.selectbox("Calculate the intercept? Default: No", ("Yes", "No"))
                if fit_int == "Yes":
                    param["fit_int"] = True
                else:
                    param["fit_int"] = False

                normalize = st.selectbox("Normalize? Default: No", ("Yes", "No"))
                if normalize == "Yes":
                    param["normalize"] = True
                else:
                    param["normalize"] = False

            if alg_name == "Logistic Regression":

                param["solver"] = st.selectbox("Which Solver to Use? Default: lbfgs", ("newton-cg", "lbfgs", "liblinear", "sag", "saga"))

                if param["solver"] == "newton-cg" or param["solver"] == "sag" or param["solver"] == "lbfgs":
                    param["penalty"] = "l2"
                if param["solver"] == "saga":
                    param["penalty"] = st.selectbox("Which penalty function to use? Default: l2", ("l1", "l2", "none"))
                if param["solver"] == "liblinear":
                    param["penalty"] = st.selectbox("Which penalty function to use? Default: l2", ("l1", "l2", "elasticnet"))

                if param["solver"] == "liblinear" and param["penalty"] == "l2":
                    dual = st.selectbox("Dual On? Default: No" , ("Yes", "No"))
                    if dual == "Yes":
                        param["dual"] = True
                    else:
                        param["dual"] = False
                else:
                    param["dual"] = False



                param["C"] = st.slider("Which C do you want to choose? Default: 1.0", 0.000001, 20.0)

                fit_int = st.selectbox("Calculate the intercept? Default: Yes", ("Yes", "No"))
                if fit_int == "No":
                    param["fit_int"] = False
                else:
                    param["fit_int"] = True

            if alg_name == "Support Vector Classification":

                param["kernel"] = st.selectbox("Which Kernel? Default: rbf", ("linear", "poly", "rbf", "sigmoid", "precomputed"))
                param["C"] = st.slider("C-Value? Default: 1.0", 0.0000001, 20.0)

                if param["kernel"] == "poly":
                    param["degree"] = st.slider("Degree Of Polynomial Default: 3",1,20)
                else:
                    param["degree"] = 3

                if param["kernel"] == "rbf" or param["kernel"] == "poly" or param["kernel"] == "sigmoid":
                    gammaask = st.selectbox("Choose a gamma coefficient Default: scale", ("scale", "auto", "Specify"))
                    if gammaask != "Specify":
                        param["gamma"] = gammaask
                    else:
                        param["gamma"] = st.slider("Specify a gamma", 0.0,20.0)
                else:
                    param["gamma"] = "scale"

                if param["kernel"] == "poly" or param["kernel"] == "sigmoid":
                    coefask = st.selectbox("Do you wanto to specify your coef0?", ("Yes", "No"))
                    if coefask == "Yes":
                        param["coef0"] = st.slider("Select a Coef0", -20.0,20.0)
                    if coefask == "No":
                        param["coef0"] = 0.0
                else:
                    param["coef0"] = 0.0

            if alg_name == "K-Nearest Neighbors Classfication":

                param["n_neighbors"] = st.slider("Choose the number of neighbors. Default: 5", 1, 20)
                param["weights"] = st.selectbox("Choose the weights", ("uniform", "distance"))
                param["algorithm"] = st.selectbox("Choose the Algorithm", ("auto", "ball_tree", "kd_tree", "brute"))
                if param["algorithm"] == "ball_tree" or param["algorithm"] == "kd_tree":
                    leafask = st.selectbox("Do you want to specify the leafsize?", ("Yes", "No"))
                    if leafask == "Yes":
                        param["leaf_size"] = st.slider("Select a leafsize", 1, 200)
                    else:
                        param["leaf_size"] = 30
                else:
                    param["leaf_size"] = None
                pask = st.selectbox("Do you want to specify the power parameter?", ("Yes", "No"))
                if pask == "Yes":
                    param["p"] = st.slider("Select a power parameter", 1,20)
                else:
                    param["p"] = 2

            if alg_name == "Decision Tree Classification":

                param["criterion"] = st.selectbox("Choose a criterion. Default: gini", ("gini", "entropy"))
                param["splitter"] = st.selectbox("Choose a spilitter. Default: best", ("best", "random"))
            return param

        param = add_parameter(alg_name)



        def obtain_class(alg_name, param):
            if alg_name == "Ridge Classification":
                classify = RidgeClassifier(alpha=param["alpha"], fit_intercept=param["fit_int"], normalize=param["normalize"], solver="auto", random_state=1)
            if alg_name == "Logistic Regression":
                classify = LogisticRegression(penalty=param["penalty"], dual=param["dual"], C=param["C"], fit_intercept=param["fit_int"], solver=param["solver"], random_state=1)
            if alg_name == "Support Vector Classification":
                classify = SVC(C=param["C"], kernel=param["kernel"], degree=param["degree"], gamma=param["gamma"], coef0=param["coef0"], random_state=1)
            if alg_name == "K-Nearest Neighbors Classfication":
                classify = KNeighborsClassifier(n_neighbors=param["n_neighbors"], weights=param["weights"], algorithm=param["algorithm"], leaf_size=param["leaf_size"], p=param["p"])
            if alg_name == "Decision Tree Classification":
                classify = DecisionTreeClassifier(criterion=param["criterion"], splitter=param["splitter"])

            return classify

        classify = obtain_class(alg_name, param)

        classify.fit(X_train,y_train)
        y_pred = classify.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write("""### Summary""")
        st.write(f"Classifier: {alg_name}")
        st.write(f"Accuracy:{acc}")






    if submode == "Pipeline Auto-Comparison (Hyperparameters Defaulted)":

        pipeline_Ridge = Pipeline([('scale_1', StandardScaler()), ('class_Ridge', RidgeClassifier(random_state=1))])

        pipeline_LR = Pipeline([('scale_2', StandardScaler()), ('class_LR', LogisticRegression(random_state=1))])

        pipeline_SVC = Pipeline([('scale_3', StandardScaler()), ('class_SVC', SVC(random_state=1))])

        pipeline_KNN = Pipeline([('scale_1', StandardScaler()), ('class_KNN', KNeighborsClassifier())])

        pipeline_DT = Pipeline([('scale_1', StandardScaler()), ('class_DT', DecisionTreeClassifier(random_state=1))])

        pipelines = [pipeline_Ridge, pipeline_LR, pipeline_SVC, pipeline_KNN, pipeline_DT]

        pipes = {0: 'Ridge Classification', 1: 'Logistic Regression', 2: 'Support Vector Classifier',
                 3: 'K-Nearest Neighbors Classification', 4: 'Decision Tree Classification'}

        def comparepipes():
            for item in pipelines:
                item.fit(X_train, y_train)

            for i, j in enumerate(pipelines):
                print('{} has an accuracy of:a {}%.'.format(pipes[i], j.score(X_test, y_test) * 100))


    x1 = X['competence']
    x2 = X['network_ability']

    fig = plt.figure()
    plt.scatter(x1, x2, s=0.1, c=y, alpha=0.9, cmap="viridis")
    plt.xlabel("Competence")
    plt.ylabel("Network Abilities")
    plt.colorbar()

    st.pyplot()


if mode == "Task 2: Regression Models w/ endogenous variable":
    st.write("""## Task 2: Regression Models w/ endogenous variable""")

    pipeline_PA = Pipeline([('scale_1', StandardScaler()), ('class_PA', PassiveAggressiveRegressor(random_state=1))])

    pipeline_Gauss = Pipeline([('scale_2', StandardScaler()), ('class_Gauss', GaussianProcessRegressor(random_state=1))])

    pipeline_Gamma = Pipeline([('scale_3', StandardScaler()), ('class_Gamma', ARDRegression())])

    pipeline_Bayes = Pipeline([('scale_1', StandardScaler()), ('class_Bayes', BayesianRidge())])

    pipeline_KNN = Pipeline([('scale_1', StandardScaler()), ('class_KNN', KNeighborsRegressor())])

    pipelines = [pipeline_PA, pipeline_Gauss, pipeline_Gamma, pipeline_Bayes, pipeline_KNN]

    pipes = {0: 'Passive Agressive Regression', 1: 'Gaussian Process Regressor', 2: 'Gamma Regressor',
             3: 'Bayesian Ridge Regressor', 4: 'K-Nearest Neighbors Classification'}


    def comparepipes():
        for item in pipelines:
            item.fit(X_train, y_train)

    x1 = X['competence']
    x2 = X['network_ability']
    x3 = y


    comparepipes()

    fig = plt.figure()
    plt.scatter(x1, x2, s=0.1, c=y, alpha=0.9, cmap="viridis")
    plt.xlabel("Competence")
    plt.ylabel("Network Abilities")
    plt.colorbar()

    fig2 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X["competence"], X["network_ability"], y, c=y, cmap='viridis')



    def plottinplottin():

        counter = 0
        colors = ["red", "green", "blue", "orange", "brown"]
        for item in pipelines:
            X_predicted_network = item.predict(X_test)
            ax.plot3D(X_test["competence"], X_test["network_ability"], X_predicted_network, color=colors[counter])
            counter += 1


    plottinplottin()



    st.pyplot()
