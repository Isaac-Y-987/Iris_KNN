"""Series of simple exercises to demonstrate how to use pandas.
"""


import pandas as pd


# Get a bunch of data
data = pd.read_csv("bezdek_iris.csv")

# Get a desired column
desired_column = data["petal-length"]

# Filter for certain columns
filtered_columns = data[["sepal-length", "petal-length"]]

# Get a desired row
desired_row = data.iloc[0]

# Filter for rows that satisfy a condition
filtered_rows = data[data["label"] == "Iris-setosa"]

# Get a random sample of rows.  Order is not preserved.
random_rows = data.sample(frac=0.5, random_state=0)     # random_state is just a seed


# Combine tables' rows
table_merged = pd.concat([data, random_rows], ignore_index=True)

# Apply a function to each row
table_stringified = data.apply(lambda row: pd.Series(str(element) for element in row), axis=1)

# Create a new column
table_with_new_column = data.copy()
table_with_new_column["weight"] = "light"

# Create a new column whose value depends on other columns
table_with_dependent_column = data.copy()
table_with_dependent_column["name_uppercase"] = table_with_dependent_column.apply(lambda row: row["label"].upper(), axis=1)

# Sort table rows based on a certain column
table_sorted = data.sort_values("sepal-width", ignore_index=True)
