# Automl Platform
Sklearn powered Automl Platform 


### Install dpendency
0. Kindly make sure python 3.11 higher is installed , and install poetry  `pip install poetry`
1. Clone the repo `git clone https://github.com/vyturr/rule_enginee.git`
2. cd <to_repo_directory>
3. `poetry shell` 
4. `Poetry install`

### Remove python bytecode __pycache__ other cached files  

`pyclean -d jupyter package ruff -v .`

### Linting 

`ruff check -v .`

### Fixing linting errors

`ruff check --fix -v .`

### Source code formatting

`ruff format -v .`


# List of important ML library 
1. [mlinsights](https://github.com/sdpython/mlinsights) 
2. [mlextend](https://github.com/rasbt/mlxtend/tree/master)
3. [Category Encoders](https://github.com/scikit-learn-contrib/category_encoders) 
4. [Feature Engine](https://github.com/feature-engine/feature_engine) 
5. [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas/tree/master) 
6. [scikit-lego](https://github.com/koaning/scikit-lego)
7. [sklearn-pandas-transformers](https://gitlab.com/thibaultB/transformers)


# for pipeline implementation see
1. [Customize Transformer blogs1](https://ploomber.io/blog/sklearn-custom/)
2. [Customize Transformer blogs2](https://www.andrewvillazon.com/custom-scikit-learn-transformers/)
3. [Customize Estimators Doc](https://learn-scikit.oneoffcoder.com/index.html)

# Transformed Classifier Discussion
1. https://github.com/scikit-learn/scikit-learn/issues/20952