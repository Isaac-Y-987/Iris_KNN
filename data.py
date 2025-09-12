import pandas as pd


#
# USER-SET PARAMETERS
#
seed = 0
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 1 - train_ratio - val_ratio


#
# PREPARE DATASETS
#
data = pd.read_csv("bezdek_iris.csv")

# Create separate dataframes for each class
data_setosa = data[data['label'] == 'Iris-setosa']
data_versicolor = data[data['label'] == 'Iris-versicolor']
data_virginica = data[data['label'] == 'Iris-virginica']
# Shuffle each dataframe
data_setosa = data_setosa.sample(frac=1, random_state=seed)
data_versicolor = data_versicolor.sample(frac=1, random_state=seed)
data_virginica = data_virginica.sample(frac=1, random_state=seed)
# Combine into training, validation, and testing dataframes
train_num = round(50 * train_ratio)     # Number of training samples per class
val_num = round(50 * val_ratio)
test_num = 50 - train_num - val_num
data_train = pd.DataFrame(columns=["sepal-length", "sepal-width", "petal-length", "petal-width", "label"])
data_val = pd.DataFrame(columns=["sepal-length", "sepal-width", "petal-length", "petal-width", "label"])
data_test = pd.DataFrame(columns=["sepal-length", "sepal-width", "petal-length", "petal-width", "label"])
for ii in range(train_num):
    data_train = pd.concat([data_train, pd.DataFrame([data_setosa.iloc[ii]])], ignore_index=True)
    data_train = pd.concat([data_train, pd.DataFrame([data_versicolor.iloc[ii]])], ignore_index=True)
    data_train = pd.concat([data_train, pd.DataFrame([data_virginica.iloc[ii]])], ignore_index=True)
for ii in range(train_num+1, train_num+val_num):
    data_val = pd.concat([data_val, pd.DataFrame([data_setosa.iloc[ii]])], ignore_index=True)
    data_val = pd.concat([data_val, pd.DataFrame([data_versicolor.iloc[ii]])], ignore_index=True)
    data_val = pd.concat([data_val, pd.DataFrame([data_virginica.iloc[ii]])], ignore_index=True)
for ii in range(train_num+val_num+1, 50):
    data_test = pd.concat([data_test, pd.DataFrame([data_setosa.iloc[ii]])], ignore_index=True)
    data_test = pd.concat([data_test, pd.DataFrame([data_versicolor.iloc[ii]])], ignore_index=True)
    data_test = pd.concat([data_test, pd.DataFrame([data_virginica.iloc[ii]])], ignore_index=True)
# Delete dataframes that aren't intended for use
del data, data_setosa, data_versicolor, data_virginica
# One last shuffle of all dataframes
data_train = data_train.sample(frac=1, random_state=seed)
data_val = data_val.sample(frac=1, random_state=seed)
data_test = data_test.sample(frac=1, random_state=seed)
