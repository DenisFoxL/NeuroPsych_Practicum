import pandas as pd
import numpy as np


def clean(path):
    """
    Function for removing unnecessary information
    :param path: The path to the CSV file to be cleaned.
    :return: The cleaned DataFrame.
    """
    # Read in the CSV file and assign it to the 'data' variable.
    data = pd.read_csv(path)

    # Drop the first row of the DataFrame and reset the index to start at 0.
    data = data.drop(index=0).reset_index(drop=True)

    # Remove the string ",assessment" from the 'Identifiers' column.
    data['Identifiers'] = data['Identifiers'].str.replace(',assessment', '')

    # Replace all period characters in the DataFrame with NaN values.
    data = data.replace('.', np.nan)

    # Convert the types of the columns to an appropriate data type.
    data['Identifiers'] = data['Identifiers'].astype(str)
    data['Basic_Demos,Sex'] = data['Basic_Demos,Sex'].astype(int)
    data.iloc[:, 3:] = data.iloc[:, 3:].astype('float64')

    return data


def calculate_score(data, q, name):
    """
    Calculates the sum of values from relevant columns in a DataFrame and adds the result to a new column.

    :param data: The DataFrame to calculate the score on.
    :param q: A string that serves as a filter to identify relevant columns for the score calculation.
    :param name: A string that will be used to name the new column with the calculated score.
    :return: The DataFrame with the new calculated score column.
    """

    # Create an empty list to store relevant column names.
    relevant_cols = []

    # Loop over all column names in the DataFrame that contain the filter string 'q'.
    for col in [c for c in data.columns if q in c]:
        parts = col.split('_')
        last_part = parts[-1]

        # Check if the last part of the column name is a number.
        # If it is, add the column name to the list of relevant columns.
        if last_part.isdigit():
            relevant_cols.append(col)
        else:
            continue

    # Calculate the sum of values in the relevant columns for each row and add the result to a new column with the name 'name'.
    data[name] = data[relevant_cols].astype('float64').sum(axis=1, skipna=True)

    # Return the DataFrame with the new calculated score column.
    return data


if __name__ == '__main__':
    pass
