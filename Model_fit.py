import sys
import gc
import os
import numpy as np
import random
import math
import torch
from torch.optim import lr_scheduler
import json
import datetime
import SimpleITK as sitk
import copy
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def save_array_nii(mask, savepath, nbits=np.int8):
    savefoldpath = os.path.split(savepath)[0]
    if not os.path.exists(savefoldpath):
        os.makedirs(savefoldpath)

    mask = mask.astype(nbits)
    mask_sitk = sitk.GetImageFromArray(mask)
    sitk.WriteImage(mask_sitk, savepath)

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(JsonEncoder, self).default(obj)

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, sqrt(2. / n))
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class gray_agumentation_tensor(object):
    def __init__(self):
        pass

    def random(self, epoch):
        random.seed(random.randint(0, 99999999))
        self.parms = []
        for i in range(epoch):

            status = random.choice([1, 2, 3, 4])

            n1 = random.uniform(0.1, 1.0)
            if random.choice([False, True]):
                n1 = 1.0 / n1

            n1L = random.uniform(0.01, n1)

            n2 = random.uniform(0.1, 1.0)
            if random.choice([False, True]):
                n2 = 1.0 / n2

            n3 = random.uniform(0.01, 1.0)
            num = random.randint(5, 40)

            self.parms.append((n1, n1L, n2, n3, num, status))


    def getnolinearitem(self, img_norm, para):
        status = para[5]
        if status == 1:
            nolinearitem = torch.pow(img_norm, para[0])

        elif status == 2 :
            nolinearitem = 1.0 - torch.pow(1.0 - torch.pow(img_norm, para[0]), para[2])

        elif status == 3 :
            nolinearitem = 2 * torch.pow(img_norm, para[0]) / (1.0 + torch.pow(img_norm, para[1]))

        elif status == 4 :
            vmin = math.exp(-para[4]*para[3])/(math.exp(-para[4]*para[3])+math.exp(para[4]*para[3]))
            vmax = math.exp(para[4]-para[4]*para[3])/(math.exp(para[4]-para[4]*para[3])+math.exp(para[4]*para[3]-para[4]))
            nolinearitem = torch.exp(para[4]*img_norm-para[4]*para[3])/(torch.exp(para[4]*img_norm-para[4]*para[3]) + torch.exp(-para[4]*img_norm+para[4]*para[3]))
            nolinearitem = (nolinearitem - vmin) / (vmax - vmin)

        return nolinearitem

    def run(self, src, mask=None, complexity=15, noise_var=0.05):

        vmin = src.min()
        vmax = src.max()
        src = (src-vmin)/(vmax-vmin)

        It = random.randint(1, complexity)
        alpha = np.random.uniform(0.01, 1.0, It)
        self.random(It)

        alpha_sum = alpha.sum()

        newsrc = torch.zeros_like(src)
        for i in range(0, It):
            newsrc = newsrc + alpha[i] * self.getnolinearitem(src, self.parms[i])

        # 与原图叠加
        newsrc = newsrc / alpha_sum
        alpha = random.uniform(0, 0.005)
        newsrc = newsrc * alpha + (1.0 - alpha) * src
        if random.choice([True, False]):
            newsrc *= (1.0 + torch.rand(newsrc.shape).cuda() * random.uniform(0.0, noise_var))

        newsrc = (newsrc - newsrc.min()) / (newsrc.max()-newsrc.min())
        newsrc = newsrc * (vmax-vmin) + vmin


        # for x in locals().keys():
        #    del locals()[x]

        del It
        del alpha
        del self.parms
        gc.collect()

        return newsrc







class fit_config():
    def __init__(self):
        self.epochs = 50
        self.l_r = 1e-3
        self.batch_size = 1
        self.num_gpu = 1
        self.optimizer = None
        self.criterion = None
        self.criterion_weight = None
        self.monitor = None
        self.model_path = None
        self.train_mode = 'scrach'
        self.val_split = 0.1
        self.val_idx = None
        self.re_model = None
        self.pre_model = None

def weight_perturbation_in_lowbits(optimizer):
    # 半精度扰动
    for group in optimizer.param_groups:
        for p in group['params']:
            p.data = p.data.half().float()


def check_keys(model, weight_state_dict):
    ckpt_keys = set(weight_state_dict.keys())
    if hasattr(model, 'module'):
        model_keys = set(model.module.state_dict().keys())
    else:
        model_keys = set(model.state_dict().keys())

    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def load_pre_weight(Model, weightpath):
    if os.path.exists(weightpath):
        state = torch.load(weightpath)
        if hasattr(state, "state_dict"):
            state = {key.replace('module.', ''): value for key, value in state.state_dict().items()}
        else:
            state = {key.replace('module.', ''): value for key, value in state.items()}

        if check_keys(Model, state):
            try:
                Model.load_state_dict(state)
            except:
                try:
                    Model.module.load_state_dict(state)
                except:
                    raise (Exception("Error happend in {}".format(weightpath)))
        else:
            raise Exception('Key not Matched')

    else:
        raise(Exception("{} not exist".format(weightpath)))

    return Model


def seg_model_fit(train_config):
    random.seed(2018)
    ## train parameters
    optimizer = train_config.optimizer
    criterion = train_config.criterion
    Epochs = train_config.epochs
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    taskname = train_config.taskname
    fine_tune_model_dir = train_config.fine_tune_model_dir
    model_save_dir = train_config.model_save_dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model_name = train_config.model_name

    Model = train_config.model
    traindataset = train_config.traindataset
    validdataset = train_config.validdataset
    gray_agumentation = gray_agumentation_tensor()


    sys.setrecursionlimit(10000)
    print("Model Training!! ")
    # 导入预训练权重或者fine tune权重
    try:
        Model = load_pre_weight(Model, model_save_dir + "/" + model_name.replace(".pkl", "_val.pkl"))
        print('Load saved weight successed!')
    except:
        try:
            Model = load_pre_weight(Model, fine_tune_model_dir + "/" + model_name.replace(".pkl", "_val.pkl"))
            print('Load pre-weight successed!')
        except:
            print("init weight")
            pass

    Num_Train = len(traindataset)
    TrainDataloader = traindataset
    ValidDataloader = validdataset

    # 定义训练记录参数
    Output_channel = train_config.Num_Group
    epochs_channel_loss = np.zeros([Epochs, np.maximum(1, Output_channel), 2], dtype='float32')
    epochs_loss = np.zeros([Epochs, 2], dtype='float32')
    mean_channel_dice = np.zeros([np.maximum(1, Output_channel)], dtype='float32')
    mean_channel_weight = np.zeros([np.maximum(1, Output_channel)], dtype='float32')

    record_epoch = -1
    max_dice_valid = 0
    max_dice_train = 0
    train_pdice_record = {}
    train_ndice_record = {}
    valid_dice_record = {}
    cases_dice = {}
    try:
        targetnames = TrainDataloader.get_Target_Name()
    except:
        targetnames = TrainDataloader.dataset.get_Target_Name()

    for epoch in range(Epochs):
        #optimizer._reset_lr_to_swa()
        print('Starting epoch %d/%d.' % (epoch + 1, Epochs))
        Model.train()

        epoch_loss = 0
        # newsize = np.array([8 * random.randint(6, 39), TrainDataloader.Input_Size[1], TrainDataloader.Input_Size[2]])
        # TrainDataloader.set_Input_Size(newsize)
        # print(newsize)
        mean_channel_dice = mean_channel_dice * 0
        mean_channel_weight = mean_channel_weight * 0
        for a, (imgs, true_masks, casenames) in enumerate(TrainDataloader):
            train_config.TrainMoniter.processbar(epoch+1, Epochs, TrainDataloader.get_procee(), len(TrainDataloader), opts=taskname)
            # 网络前通道调整
            #  导入cuda
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            output = Model(imgs)
            loss, pdice = criterion(output, true_masks, float(epoch)/Epochs)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            mean_channel_dice += np.array(pdice)
            mean_channel_weight += (np.array(pdice) > 0)
            epochs_channel_loss[epoch, :, 0] += [1 - cd for cd in pdice]

            for i in range(0, len(pdice)):
                train_pdice_record[targetnames[i]] = pdice[i]
            train_config.TrainMoniter.plot_losses(train_pdice_record)

            print("the channel dices are {}".format([cd for cd in pdice]))

            # # # 查看中间结果
            if a % 2 == 0:
                if len(true_masks.shape) == 5:
                    label = true_masks[:, :, 1:-1].sum(dim=[0, 2]).cpu().numpy()
                    if output.shape[1] > 1:
                        result = output[:, :, 1:-1].detach().sum(dim=[0, 2]).cpu().numpy()[1::]
                    else:
                        result = output.detach().sum(dim=[0, 2]).cpu().numpy()
                    result_view = {"train_" + "_".join(targetnames): [result, label]}
                    train_config.TrainMoniter.show_image3D(result_view, "train")
                elif len(true_masks.shape) == 4:
                    label = true_masks.sum(dim=[0]).cpu().numpy()
                    if output.shape[1] > 1:
                        result = output[:,1::,:].detach().sum(dim=[0]).cpu().numpy()
                    else:
                        result = output.detach().sum(dim=[0]).cpu().numpy()
                    result_view = {"train_" + "_".join(targetnames): [result, label]}
                    train_config.TrainMoniter.show_image3D(result_view, "train")

        epoch_loss = epoch_loss / (a + 1)
        mean_channel_weight[mean_channel_weight == 0] = 1.0
        mean_channel_dice = mean_channel_dice / mean_channel_weight
        train_dice = mean_channel_dice[mean_channel_dice>0].mean()
        epochs_channel_loss[epoch, :, 0] = epochs_channel_loss[epoch, :, 0] / (a + 1)
        epochs_loss[epoch, 0] = epoch_loss

        if train_dice > max_dice_train:
            max_dice_train = train_dice
            best_epoch_train = epoch
            try:
                torch.save(Model.state_dict(), os.path.join(model_save_dir, model_name.replace(".pkl", "_train.pkl")), _use_new_zipfile_serialization=False)
            except:
                torch.save(Model.state_dict(), os.path.join(model_save_dir, model_name.replace(".pkl", "_train.pkl")))

        ###############################################################################################################
        ###############################################################################################################
        ## validation
        print('start eval')
        Model.eval()
        val_loss = 0
        mean_channel_dice = mean_channel_dice * 0
        mean_channel_weight = mean_channel_weight * 0
        for a, (imgs, true_masks, _) in enumerate(ValidDataloader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            with torch.no_grad():
                 output = Model(imgs)
                 val_loss_, pdice = criterion.dice(output, true_masks)

            val_loss += val_loss_.item()
            epochs_channel_loss[epoch, :, 1] += [1 - cd for cd in pdice]
            mean_channel_dice += np.array(pdice)
            mean_channel_weight += (np.array(pdice) > 0)

            # # 查看中间结果
            if len(true_masks.shape) == 5:
                label = true_masks[:, :, 1:-1].sum(dim=[0, 2]).cpu().numpy()
                if output.shape[1] > 1:
                    result = output[:, :, 1:-1].detach().sum(dim=[0, 2]).cpu().numpy()[1::]
                else:
                    result = output.detach().sum(dim=[0, 2]).cpu().numpy()
                result_view = {"V_T_" + "_".join(targetnames): [result, label]}
                train_config.TrainMoniter.show_image3D(result_view, "valid")
            elif len(true_masks.shape) == 4:
                label = true_masks.sum(dim=[0]).cpu().numpy()
                if output.shape[1] > 1:
                    result = output[:, 1::, :].detach().sum(dim=[0]).cpu().numpy()
                else:
                    result = output.detach().sum(dim=[0]).cpu().numpy()
                result_view = {"V_T_" + "_".join(targetnames): [result, label]}
                train_config.TrainMoniter.show_image3D(result_view, "valid")

        val_loss = val_loss/(a + 1)
        mean_channel_weight[mean_channel_weight==0] = 1.0
        mean_channel_dice = mean_channel_dice / mean_channel_weight
        tempdice = mean_channel_dice  #[2::]
        val_dice = tempdice.mean()
        epochs_channel_loss[epoch, :, 1] = epochs_channel_loss[epoch, :, 1] / (a + 1)
        epochs_loss[epoch, 1] = val_loss
        for i in range(0, epochs_channel_loss.shape[1]):
            valid_dice_record["valid_" + targetnames[i]] = mean_channel_dice[i]
        train_config.TrainMoniter.plot_losses(valid_dice_record)
        #ValidDataloader.release_memory()
        print("Validation finished ! --- channel dice: {}".format(mean_channel_dice))
        print("Validation finished ! --- validation loss: {}, validation dice: {}".format(val_loss, val_dice))
        if val_dice > max_dice_valid:
            max_dice_valid = val_dice
            best_epoch_valid = epoch
            try:
                torch.save(Model.state_dict(), os.path.join(model_save_dir, model_name.replace(".pkl", "_val.pkl")), _use_new_zipfile_serialization=False)
            except:
                torch.save(Model.state_dict(), os.path.join(model_save_dir, model_name.replace(".pkl", "_val.pkl")))

        torch.cuda.empty_cache()
        scheduler.step()
    print("the best valid model is saved in epoch {}".format(best_epoch_valid))
    print("the best train model is saved in epoch {}".format(best_epoch_train))

    # 释放资源
    try:
        TrainDataloader.release_memory()
        ValidDataloader.release_memory()
    except:
        TrainDataloader.dataset.release_memory()
        ValidDataloader.dataset.release_memory()
    gc.collect()
    torch.cuda.empty_cache()