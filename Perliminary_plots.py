import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Organizing_funcs as of
import Statistic_funcs as sf
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

def correlation_mat_plot(data):
    # corr_mat,p_values=sf.Correlation_mat(data)
    # sns.set(style='white')
    # mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    # sns.heatmap(corr_mat, mask=mask, annot=True, cmap='coolwarm', center=0,
    #             square=True, linewidths=.5, cbar_kws={'shrink': .5})
    # for i in range(len(df.columns)):
    #     for j in range(len(df.columns)):
    #         if i < j and p_values[i, j] < 0.05:
    #             plt.text(j + 0.5, i + 0.5, '*', horizontalalignment='center',
    #                      verticalalignment='center', fontsize=18)
    # plt.show()
    corr_mat=data.corr()
    sns.heatmap(corr_mat, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
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
if __name__=='__main__':
    path=r'C:\Users\denis\Downloads\Homework\Neuropsychology\HBN\Data\data-2023-03-20.csv'
    data=of.clean(path)
    data=of.calculate_score(data,'CBCL','CBCl_OC_Raw')
    main_cols=['Basic_Demos,Age', 'Basic_Demos,Sex','CBCL,CBCL_AD', 'SDS,SDS_Total_Raw','CBCl_OC_Raw']
    corr_data=data.loc[:,main_cols]
    model=sf.linear_reg(data,'SDS,SDS_Total_Raw','CBCl_OC_Raw',['CBCL,CBCL_AD'])
    Linear_reg_plot(data,sf.linear_reg(data,'SDS,SDS_Total_Raw','CBCl_OC_Raw',['CBCL,CBCL_AD']))