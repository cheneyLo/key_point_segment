import sys
from TransContextResUnet3D import TransContextResUNet
import torch.optim as optim
import torch.nn as nn
from lookahead import Lookahead
from TrainMoniter import Visualizer
from Model_fit import seg_model_fit, weights_init, fit_config
from balanced_dataset_beauty import *
from loss import MyDice


nasal_bone0_list = ['nasal_bone']
nasal_bone0_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel', 'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']
nasal_bone0_size = np.array((48, 200, 256))
nasal_bone0_wwwl = [[800, 2000], [-50, 50]]


mandibular_part0_list = ['Infradentale', 'Supramentale', 'Pogonion', 'Gnathion', 'Menton', 'Mentale_L', 'Mentale_R']
mandibular_part0_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel', 'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']
mandibular_part0_size = np.array((16, 200, 256))
mandibular_part0_wwwl = [[300, 2000], [-50, 50]]


mandibular_part1_list = ['MentalTubercle_L', 'MentalTubercle_R','SigmoidNotch_L', 'SigmoidNotch_R', 'Gonion_L', 'Gonion_R',  'Genion', 'LineaObliquaMandibulae_L', 'LineaObliquaMandibulae_R']
mandibular_part1_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel',  'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']
mandibular_part1_size = np.array((16, 400, 496))
mandibular_part1_wwwl = [[300, 2000], [-50, 50]]

Upperteeth_part0_list = ['Supradentale', 'Subspinale', 'Mastoideale_L', 'Mastoideale_R', 'Opisthion', 'Endobasion', 'Zygomaxillare_L', 'Zygomaxillare_R']
Upperteeth_part0_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel',  'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']
Upperteeth_part0_size = np.array((16, 400, 496))
Upperteeth_part0_wwwl = [[300, 2000], [-50, 50]]

Upperteeth_part1_list = ['AnteriorNasalSpine', 'PosteriorNasalSpine', 'Coronion_L', 'Coronion_R',
                         'CondylionLaterale_L', 'CondylionLaterale_R', 'CondylionMediale_L', 'CondylionMediale_R']
Upperteeth_part1_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel',  'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']
Upperteeth_part1_size = np.array((16, 400, 496))
Upperteeth_part1_wwwl = [[300, 2000], [-50, 50]]

Upperteeth_part2_list = ['Inion', 'Jugale_L', 'Jugale_R', 'Porion_L', 'Porion_R','Zygion_L', 'Zygion_R', 'ForamenInfraorbitale_L', 'ForamenInfraorbital']
Upperteeth_part2_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel',  'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']

Upperteeth_part2_size = np.array((16, 400, 496))
Upperteeth_part2_wwwl = [[300, 2000], [-50, 50]]

Eye_part0_list = ['Orbitale_L', 'Orbitale_R', 'Asterion_L', 'Asterion_R','Ektokonchion_L', 'Ektokonchion_R', 'Rhinion',
                  'FrontomalareOrbitale_L', 'FrontomalareOrbitale_R', 'Krotaphion_L', 'Krotaphion_R']
Eye_part0_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel',  'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']

Eye_part0_size = np.array((16, 400, 496))
Eye_part0_wwwl = [[300, 2000], [-50, 50]]

Eye_part1_list = ['Sella', 'Dakryon_L', 'Dakryon_R', 'FrontomalareTemporale_L', 'FrontomalareTemporale_R',
                      'Glabella_L', 'Glabella_R', 'Nasion', 'ForamenSupraorbitale_L', 'ForamenSupraorbitale_R']
Eye_part1_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel',  'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']

Eye_part1_size = np.array((16, 400, 496))
Eye_part1_wwwl = [[300, 2000], [-50, 50]]

Forehead_part0_list = ['Ophryon_L', 'Ophryon_R', 'Euryon_L', 'Euryon_R',  'Opisthocranion',  'Frontotemporale_L',
                       'Frontotemporale_R', 'Coronale_L', 'Coronale_R', 'Lambda', 'Metopion', 'Vertex']
Forehead_part0_negative_oar_list = ['Kidney_L', 'Kidney_R', 'Spleen', 'Liver', 'Stomach', 'Pancreas', 'Gallbladder', 'SmallBowel',  'Colon', 'Bladder', 'Femoral_R', 'Femoral_L',  'Rectum', 'Adrenal', 'Duodenum', 'PelvicBone', 'Heart', 'Lung_L', 'Lung_R']

Forehead_part0_size = np.array((16, 400, 496))
Forehead_part0_wwwl = [[300, 2000], [-50, 50]]


exclude_oar_list = ['Mentale_L', 'Mentale_R', 'MentalTubercle_L', 'MentalTubercle_R', 'SigmoidNotch_L', 'SigmoidNotch_R', 'Gonion_L', 'Gonion_R',    'LineaObliquaMandibulae_L', 'LineaObliquaMandibulae_R']





def build_model(model, Num_Group, Num_Gpus, depth, initial_features, img_size, d_r=0):
    print('Creating and compiling model...')
    print('-' * 30)
    # Model = DetectNet(in_channel=1, out_channel=Num_Group + 1)

    if Num_Group > 1 :
        Model = model(in_channel=1, out_channel=Num_Group + 1, depth=depth, initial_features=initial_features, img_size=img_size, d_r=d_r)
    else:
        Model = model(in_channel=1, out_channel=Num_Group, depth=depth, initial_features=initial_features, img_size=img_size, d_r=d_r)

    Model.apply(weights_init)

    if torch.cuda.is_available():
        Model = Model.cuda()
        if Num_Gpus >= 2:
            Model = nn.DataParallel(Model, device_ids=range(Num_Gpus))

    return Model


def get_oar_num(oar_list, exclude_oar=[], combine_left_right=False):
    if not combine_left_right:
        return len(oar_list)
    else:
        n = 0
        for oarname in oar_list:
            if oarname not in exclude_oar:
                if '_L' in oarname and oarname.replace('_L', "_R") in oar_list:
                    n += 1

        return len(oar_list) - n

def get_oar_id(oarnames, oar_list, exclude_oar, combine_left_right=False):
    if isinstance(oarnames, str):
        oarname = oarnames
        if oarname not in oar_list:
            return None
        id = 0
        for oar in oar_list:
            if oar in exclude_oar:
                if oarname != oar:
                    id += 1
                    continue
                else:
                    break
            else:
                if oarname != oar :
                    if '_R' not in oar and combine_left_right:
                        id += 1
                    elif not combine_left_right:
                        id += 1
                else:
                    break
    elif isinstance(oarnames, list):
        id = []
        for oarname in oarnames:
            id.append(get_oar_id(oarname, oar_list, combine_left_right=combine_left_right))
    else:
        raise Exception('not support format {}'.format(oarnames))

    return id


def Trainer(Train_paras):
    #### get the image and OAR masks data
    fine_tune_model_dir = Train_paras['fine_tune_model_dir']
    model_save_dir = Train_paras['model_save_dir']
    model_prefix = Train_paras['prefix']
    Input_Size = Train_paras['Input_Size']
    filenamedict = Train_paras['filenamedict']
    Num_OAR = Train_paras['Num_OAR']
    window = Train_paras['window']
    combine_left_right = Train_paras['combine_left_right']
    model = Train_paras['model']
    trainpath, validpath, negsamplepath = Train_paras['trainpath'], Train_paras['validpath'], Train_paras['negsamplepath']
    batch_size = Train_paras['batch_size']
    workers = Train_paras['workers']
    downsample = Train_paras['downsample']
    thickness_range = Train_paras['thickness_range']
    cache_memory_num = Train_paras['cache_memory_num']
    parallel = Train_paras['parallel']
    val_idx = Train_paras['val_idx']
    expand = Train_paras['expand']
    epochs = Train_paras['epochs']
    data_group_num = Train_paras['data_group_num']
    whole_slice_oars = Train_paras['whole_slice_oars']
    need_negative_ratio = Train_paras['need_negative_ratio']
    exclude_oar = Train_paras['exclude_oar']

    ## prepare the dataset
    [TrainDataset, ValidDataset] = Generate_Train_Valid_Datasets(trainpath=trainpath, validpath=validpath, neg_sample_path=negsamplepath,
                                                                 filenamedict=filenamedict, exclude_oar=exclude_oar,
                                                                 batch_size=batch_size, workers=workers,
                                                                 augmentation=None,
                                                                 Input_Size=Input_Size, window=window,
                                                                 downsample=downsample, thickness_range=thickness_range,
                                                                 combine_left_right=combine_left_right,
                                                                 data_group_num=data_group_num,
                                                                 cache_memory_num=cache_memory_num, shuffle=True,
                                                                 parallel=parallel, val_idx=val_idx, val_split=0.01,
                                                                 expand=expand, need_negative_ratio=need_negative_ratio)

    ## set the train parameters
    train_config = fit_config()
    train_config.fine_tune_model_dir = fine_tune_model_dir
    train_config.model_save_dir = model_save_dir
    train_config.batch_size = batch_size
    train_config.workers = workers
    train_config.taskname = model_prefix
    train_config.l_r = 1e-2
    train_config.epochs = epochs
    train_config.model = model
    train_config.Num_Group = Num_OAR
    train_config.traindataset = TrainDataset
    train_config.validdataset = ValidDataset
    train_config.optimizer = Lookahead(optim.Adam(model.parameters(), lr=train_config.l_r), k=5, alpha=0.5)
    #train_config.optimizer = optim.Adam(model.parameters(), lr=train_config.l_r)
    train_config.criterion = MyDice(types=Num_OAR, whole_slice_oars=whole_slice_oars)
    train_config.TrainMoniter = Visualizer(display_env=model_prefix, open=Train_paras['open_vis'])
    train_config.train_mode = 'fine_tune'
    try:
        train_config.model_name = '{}_{}.pkl'.format(model_prefix, model.module.name)
    except:
        train_config.model_name = '{}_{}.pkl'.format(model_prefix, model.name)

    seg_model_fit(train_config)
    print("Model train and validation is finished!")




def train_and_valid(tasklist, fine_tune_model_dir, save_model_dir, trainpath, validpath, negsamplepath ,cache_memory_num, validset, data_group_num, epochs, expand, batch_size, workers, Num_Gpus, need_negative_ratio=0, open_vis=False):
    Train_paras = dict()
    Train_paras['model_save_dir'] = save_model_dir
    Train_paras['fine_tune_model_dir'] = fine_tune_model_dir
    Train_paras['epochs'] = epochs
    Train_paras['data_group_num'] = data_group_num
    Train_paras['expand'] = expand
    Train_paras['combine_left_right'] = True
    Train_paras['parallel'] = True
    Train_paras['trainpath'] = trainpath
    Train_paras['validpath'] = validpath
    Train_paras['negsamplepath'] = negsamplepath
    Train_paras['cache_memory_num'] = cache_memory_num
    Train_paras['val_idx'] = validset
    Train_paras['expand'] = expand
    Train_paras['batch_size'] = batch_size * Num_Gpus
    Train_paras['workers'] = workers
    Train_paras['open_vis'] = open_vis
    Train_paras['need_negative_ratio'] = need_negative_ratio
    Train_paras['thickness_range'] = [0.1, 1.5]
    exclude_oar = exclude_oar_list
    Train_paras['exclude_oar'] = exclude_oar
    d_r = 0.1

    for task in tasklist:
        prefix = task[0]
        taskid = task[1]

        try:
            Train_paras['prefix'] = "{}{}".format(prefix, taskid)
            Train_paras['Input_Size'] = eval("{}{}_size".format(prefix, taskid))
            oar_list = eval("{}{}_list".format(prefix, taskid))
            Num_OAR = get_oar_num(oar_list, exclude_oar, combine_left_right=Train_paras['combine_left_right'])
            Train_paras['Num_OAR'] = Num_OAR
            Train_paras['filenamedict'] = {
                "src": 'image.nii.gz',     # , 'headbone.nii.gz'
                "mask": ["body.nii.gz"],
                "target": ["{}.nii.gz".format(id) for id in oar_list],
                "negative": eval("{}{}_negative_oar_list".format(prefix, taskid)) }
            Train_paras['window'] = eval("{}{}_wwwl".format(prefix, taskid))
            Train_paras['model'] = build_model(TransContextResUNet, Num_OAR, Num_Gpus, depth=3,  initial_features=32, img_size=Train_paras['Input_Size'], d_r=d_r)

            Train_paras['downsample'] = 1
            Train_paras['whole_slice_oars'] = []

        except:
            raise Exception("Error prefix")

        print("{}-whole oarlist:{}".format(prefix, Train_paras['whole_slice_oars']))
        print("input size {}".format(Train_paras['Input_Size']))
        print("input window {}".format(Train_paras['window']))
        Trainer(Train_paras)