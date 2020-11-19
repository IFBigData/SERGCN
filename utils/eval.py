import numpy as np
from utils.logger import logger

def cal_AP(scores_list,labels_list):
    list_len = len(scores_list)
    assert(list_len == len(labels_list)), 'score and label lengths are not same'
    dtype = [('score',float), ('label',int)]
    values = []
    for i in range(list_len):
        values.append((scores_list[i],labels_list[i]))
    np_values = np.array(values, dtype=dtype)
    np_values = np.sort(np_values, order='score')
    np_values = np_values[::-1]
    class_num = sum(labels_list)
    max_pre = np.zeros(class_num)
    pos = 0
    for i in range(list_len):
        if (np_values[i][1] == 1):
            max_pre[pos] = (pos + 1) * 1.0 / (i + 1)
            pos = pos + 1
    for i in range(class_num-2, -1, -1):
        if (max_pre[i] < max_pre[i + 1]):
            max_pre[i] = max_pre[i + 1]
    return sum(max_pre) / (len(max_pre) + 1e-6)

def normnp(scores_np):
    shape_x = scores_np.shape
    for i in range(shape_x[0]):
        scores_np[i,:] = scores_np[i,:] / sum(scores_np[i,:])
    return scores_np

def compute_map(confs, labels):
    # confs: confidence of each class, shape is [num_samples, num_classed]
    # labels: label for each sample, shape is [num_samples, 1]
    csn = normnp(confs)
    num_class = confs.shape[-1]
    per_class_ap = []
    for i in range(num_class):
        class_scores = list(csn[:, i])
        class_labels = [l == i for l in labels]

        per_class_ap.append(cal_AP(class_scores, class_labels))
    logger.info(per_class_ap)
    return np.mean(per_class_ap)


if __name__ == '__main__':
    conf = np.array([0.9, 0.1, 0.8, 0.4])
    pred_cls = np.array([0, 1, 2, 0])
    target_cls = np.array([0, 0, 2, 1])
    print(compute_map(conf, pred_cls, target_cls))