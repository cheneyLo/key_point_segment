import sys
from Beauty_OGS import train_and_valid
import os


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


if __name__ == "__main__":

    trainpath = [
        r'./data/train_point_cycle/train_lby-20-02_cycle',
    ]
    negsamplepath = [
                ]

    validpath = [
        r'./data/train_point_cycle/valid_lby-20-02_cycle',
    ]


    fine_tune_model_dir = r"./weight"
    save_model_dir = r"./weight"

    Num_Gpus = 1
    data_group_num = 40
    cache_memory_num = None
    try:
        validset = os.path.join(os.path.split(trainpath)[0], "{}_validset.txt".format(os.path.split(trainpath)[-1]))
    except:
        validset = os.path.join(os.path.split(trainpath[0])[0], "{}_validset.txt".format(os.path.split(trainpath[0])[-1]))

    batch_size = 1
    workers = 1
    epochs = 20 * data_group_num
    expand = 1
    need_negative_ratio = 0.0
    open_vis = True

    tasklist = [['mandibular_part', 0]]

    train_and_valid(tasklist, fine_tune_model_dir, save_model_dir, trainpath, validpath, negsamplepath,
                                   cache_memory_num, validset, data_group_num, epochs, expand, batch_size, workers,
                                   Num_Gpus, need_negative_ratio, open_vis)



