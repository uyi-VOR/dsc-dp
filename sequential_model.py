import numpy as np

def obtain_c4_feature_for_a_list_of_sequences(seqs):
  c4_features = []
  for seq in seqs:
      this_kmer_feature = obtain_c4_feature_for_one_sequence(seq)
      c4_features.append(this_kmer_feature)
  return np.hstack((c4_features))

def obtain_c4_feature_for_one_sequence(seq):
  data = np.zeros((len(seq),4),dtype=np.uint8)
  for i in range(len(seq)):
        if seq[i] == 'A':
            data[i] = [1,0,0,0]
        if seq[i] == 'C':
            data[i] = [0,1,0,0]
        if seq[i] == 'G':
            data[i] = [0,0,1,0]
        if seq[i] == 'T':
            data[i] = [0,0,0,1]
  return data

# seqs = ['ATGAGGTCCGAA']  # 序列列表
# c4_features = obtain_c4_feature_for_a_list_of_sequences(seqs)
# print(c4_features.shape)