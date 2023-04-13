import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Functions.Organizing_funcs as of
import Functions.Statistic_funcs as sf
import seaborn as sns

def scatter(data,cols):
    coordinates=data[cols]
    x=coordinates[cols[0]].values
    y = coordinates[cols[1]].values
    plt.scatter(x,y)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title(f'Scatterplot of {cols[0]} and {cols[1]}')
    plt.show()




def Linear_reg_plot(data,model):
    # plot the predicted values against the actual values
    plt.scatter(data['SDS,SDS_Total_Raw'], data['CBCl_OC_Raw'])
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression Results')

    # add the regression line to the plot
    xmin, xmax = plt.xlim()
    coef = model.coef_
    intercept = model.intercept_
    x_vals = np.array([xmin, xmax])
    y_vals = intercept + coef[0] * x_vals + coef[1] * x_vals
    plt.plot(x_vals, y_vals, '--')

    plt.show()

def plot_corr_mat(corr_mat,p_values):
    """
    Function that plots a correlation matrix and marks significant values
    :param corr_mat: DataFrame of the correlation matrix
    :param p_values: DataFrame of the P-value matrix
    :return: none
    """
    #create mask for showing just the bottom triangle
    mask=np.tril(np.ones(corr_mat.shape)).astype(np.bool)

    #plot the matrix
    ax=sns.heatmap(corr_mat, mask=~mask, cmap="coolwarm", center=0, square=True, annot=True, fmt=".2f")

    #loops for adding the significant values to the plot
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if p_values.iloc[i,j] < 0.05:
                if p_values.iloc[i,j]<0.01:
                    if p_values.iloc[i,j]<0.01:
                        ax.text(j, i + 0.3, '***', color='black', fontsize=10)
                    else:
                        ax.text(j, i + 0.3, '**', color='black', fontsize=10)
                else:
                    ax.text(j, i + 0.3, '*', color='black', fontsize=10)
    plt.show()


if __name__=='__main__':
    path= r'/Data/data-2023-03-20.csv'
    data=of.clean(path)
    data=of.calculate_score(data,'CBCL','CBCl_OC_Raw')
    main_cols=['Basic_Demos,Age', 'Basic_Demos,Sex','CBCL,CBCL_AD', 'SDS,SDS_Total_Raw','CBCl_OC_Raw']
    corr_data=data.loc[:,main_cols]
    plot_corr_mat(*sf.corr_mat(corr_data))