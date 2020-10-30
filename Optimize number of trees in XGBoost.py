#!/usr/bin/env python
# coding: utf-8

# # How to optimize number of trees in XGBoost

# In[1]:


def optimisetree(): 
    print()
    print(format('How to optimise number of trees in XGBoost','*^82'))    

    import warnings
    warnings.filterwarnings("ignore")
    
    # load libraries
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    import matplotlib.pyplot as plt    
    
    plt.style.use('ggplot')    
    
    # load the wine datasets
    dataset = datasets.load_wine()
    X = dataset.data; y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # grid search
    model = XGBClassifier()
    n_estimators = range(50, 400, 50)
    param_grid = dict(n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, y)
    
    # summarize results
    print()
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print()
    for mean, stdev, param in zip(means, stds, params):
	     print("%f (%f) with: %r" % (mean, stdev, param))
    
    # plot
    pyplot.errorbar(n_estimators, means, yerr=stds)
    pyplot.title("XGBoost n_estimators vs Log Loss")
    pyplot.xlabel('n_estimators')
    pyplot.ylabel('Log Loss')
    pyplot.savefig('n_estimators.png')
    
optimisetree()


# In[ ]:




