import os
import numpy as np
import dnashape_merge

def obtain_dsc_feature_for_a_list_of_sequences(count):
    arr = dnashape_merge.result
    dsc_feature = arr[count]
    return dsc_feature

# dsc_features = obtain_dsc_feature_for_a_list_of_sequences(1)
# print(dsc_features.shape)