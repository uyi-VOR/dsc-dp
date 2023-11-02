import training_test_network
import utils
import time
import numpy as np
import pandas as pd

def cross_validation(type):

    batch_sz = 100
    epoch_num = 3
    learning_rt = 1e-3

    tfrecords_ls = ['/mnt/rosetta/cm/20230619-DeepCoding-main_dsc_c4_network'+ '/data_'+ type + '1.tfrecords', '/mnt/rosetta/cm/20230619-DeepCoding-main_dsc_c4_network'+ '/data_'+ type + '2.tfrecords', '/mnt/rosetta/cm/20230619-DeepCoding-main_dsc_c4_network'+ '/data_'+ type + '3.tfrecords']

    fprs = []
    tprs = []
    aucs = []
    time_elaspes = []

    # for i in range(1):
    for i in range(len(tfrecords_ls)):
        # test_tfrecords_ls = [tfrecords_ls[i]]
        # tfrecords_ls.pop(i)
        # train_tfrecords_ls = tfrecords_ls
        test_tfrecords_ls = tfrecords_ls[i:i+1]
        train_tfrecords_ls = tfrecords_ls[:i] + tfrecords_ls[i+1:]
        time_elaspe, model_file, uuid_str = training_test_network.train_model(train_tfrecords_ls, test_tfrecords_ls, type, n_epoch=epoch_num,
                                                               learning_rate=learning_rt, batch_size= batch_sz)
        time.sleep(10)
        # print("******************************")
        # print(uuid_str)
        scores, labels = training_test_network.test_model(model_file, test_tfrecords_ls, batch_sz)
        print("******************************")
        print(scores)
        print(labels)
        fpr, tpr, auc, f1 = utils.eval_perf(labels, scores)
        print("F1 score:", f1)
        utils.save_result('results/' + type + '/labels' + uuid_str + '.csv', labels)
        utils.save_result('results/' + type + '/scores' + uuid_str + '.csv', scores)
        # time_elaspe = np.float32(time_elaspe)
        time_elaspe = np.array(time_elaspe)
        utils.save_result('results/' + type + '/results' + uuid_str + '.csv', [tpr, 1 - fpr, auc, time_elaspe])


        # print(fpr,tpr,auc,time_elaspes)
        fprs.append(fpr)
        tprs.append(tprs)
        aucs.append(auc)
        time_elaspes.append(time_elaspe)

    return fprs, tprs, aucs, time_elaspes
fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('ha')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('human')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('an')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('pl')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('mouse')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('yeast')

type = 'ha'
model_file = 'model/ha/model_d6ce7577326245f5b0bfe6cdf7e7aa83.ckpt'
uuid_str = 'd6ce7577326245f5b0bfe6cdf7e7aa83'
scores, labels = training_test_network.test_model(model_file, ['data_ha4.tfrecords'], 1)
print(scores)
#print(ty(scores))
print(labels)
#print(type(labels))
print("----------")
fpr, tpr, auc, f1 = utils.eval_perf(labels, scores)
print(fpr)
print(tpr)
print(auc)
print("F1 score:", f1)
data=pd.DataFrame({'tpr':list(tpr),'fpr':list(fpr)})
data.to_csv('results/' + type + '/results' + uuid_str + '-1.csv')

















