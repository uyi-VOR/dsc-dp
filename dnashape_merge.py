import os
import numpy as np

def dsc_merge_files_in_folder(folder_path, output_file):
    merged_data = []

    file_list = os.listdir(folder_path)
    file_list.sort()

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            content = file.read().strip()
            float_list = [float(num) for num in content.split(",")]
            merged_data.append(float_list)

    num_floats = len(merged_data[0])
    merged_array = [[0.0] * len(file_list) for _ in range(num_floats)]

    for i in range(len(file_list)):
        for j in range(num_floats):
            merged_array[j][i] = merged_data[i][j]


    with open(output_file, "w") as output:
        for row in merged_array:
            output.write(",".join(str(num) for num in row))
            output.write("\n")
    return merged_array


def convert_to_numpy_array(lst, n):
    arr = np.array(lst)

    arr = arr.reshape((int(n/270), 270, 13, 1))

    return arr



folder_path = "/mnt/rosetta/cm/20230619-DeepCoding-main_dsc_c4_network/ha_cds/4"
output_file = "/mnt/rosetta/cm/20230619-DeepCoding-main_dsc_c4_network/ha_cds/merged_dsc_4.txt"

lst = dsc_merge_files_in_folder(folder_path, output_file)

L = len(lst)

result = convert_to_numpy_array(lst, L)

# print(result[2].shape)
