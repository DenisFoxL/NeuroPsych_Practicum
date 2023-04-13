import numpy as np
import scipy.stats as sts
import pandas as pd
from sklearn.linear_model import LinearRegression

def corr_mat(data):
    corr_matrix = data.corr(method='pearson')
    p_matrix = pd.DataFrame(columns=corr_matrix.columns, index=corr_matrix.columns)
    for i in corr_matrix.columns:
        for j in corr_matrix.index:
            if i==j:
                p_matrix.loc[i,j]=1
            else:
                non_missing=data[[i,j]].dropna()
                _, p = sts.pearsonr(non_missing[i], non_missing[j])
                p_matrix.loc[i,j] = p
    return corr_matrix, p_matrix

def linear_reg(data,x,y):
    """
    Function for creating a Linear Regression Model
    :param data: The DataFrame containing the data
    :param x: A list of predicting variables column names
    :param y: A string of the predicted variable column name
    :return: The fitted model
    """
    # drop rows with missing values
    data = data[x+[y]].dropna()

    # extracting predictors
    X=data[x]

    # extracting prediced variable
    y=data[y]

    # initialize LinearRegression model
    model=LinearRegression()

    #fit model
    model.fit(X,y)

    return model
