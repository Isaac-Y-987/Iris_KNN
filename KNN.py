import pandas as pd
import math
class KNN_classifier:
    def __init__(self, k, training_data: pd.DataFrame):
        self.k = k
        self.training_data = training_data
    def guess(self, test_sample: pd.Series):
        self.training_data["distance"] = self.training_data.apply(lambda row: math.sqrt((row["sepal-length"] - test_sample["sepal-length"])**2 +
                                                                                        (row["sepal-width"] - test_sample["sepal-width"])**2 +
                                                                                        (row["petal-length"] - test_sample["petal-length"])**2 +
                                                                                        (row["petal-width"] - test_sample["petal-width"])**2), axis=1)
        table_sorted = self.training_data.sort_values("distance", ascending=True, ignore_index=True)
        top_three = table_sorted.loc[0:self.k-1, "label"].to_list()
        return max(set(top_three), key=top_three.count)
        #TODO add a way to break ties by increasing the K value
    def accuracy(self, test_data_set: pd.DataFrame):
        accuracy_counter = 0
        for rr,row in test_data_set.iterrows():
            guess = self.guess(row)
            if row["label"] == guess:
                accuracy_counter+=1
        accuracy = accuracy_counter/len(test_data_set)
        return accuracy
