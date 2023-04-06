import pandas as pd
import numpy as np

def clean(path):
    data=pd.read_csv(path)
    data=data.drop(index=0).reset_index(drop=True)
    data['Identifiers']=data['Identifiers'].str.replace(',assessment','')
    data=data.replace('.',np.nan)
    data['Identifiers']=data['Identifiers'].astype(str)
    data['Basic_Demos,Sex']=data['Basic_Demos,Sex'].astype(int)
    data.iloc[:, 3:] = data.iloc[:, 3:].astype('float64')
    return data

def calculate_score(data,q,name):
    relevant_cols=[]
    for col in [c for c in data.columns if q in c]:
        parts=col.split('_')
        last_part=parts[-1]
        if last_part.isdigit():
            relevant_cols.append(col)
        else:
            continue
    data[name]=data[relevant_cols].astype('float64').sum(axis=1,skipna=True)
    return data


if __name__ == '__main__':
    pass