import numpy as np
import pandas as pd
import scipy.stats as sts
from sklearn.linear_model import LinearRegression

def Correlation_mat(data):
    corr_matrix=data.corr()
    p_values=np.zeros_like(corr_matrix)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_coef, p_value = sts.pearsonr(data[col1], data[col2])
            p_values[i,j]=p_value
    #
    # for i in range(len(data.columns)):
    #     for j in range(len(data.columns)):
    #         corr, p_value = sts.pearsonr(data.iloc[:, i], data.iloc[:, j])
    #         p_values[i, j] = p_value
    return corr_mat,p_values

def linear_reg(data,x,y,covariates):
    data=data.dropna()
    if covariates!=[]:
        X = data[[y]+covariates].values
    else:
        X=data[y].values
    x = data[x].values
    model=LinearRegression()
    model.fit(X.reshape(-1,1), x)
    return model
