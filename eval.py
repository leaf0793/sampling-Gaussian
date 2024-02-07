import numpy as np
from models import *

# from datasets import 
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from datetime import datetime, date
import os
from utils.train_util import *
# from tqdm import tqdm
from tqdm import tqdm, trange


os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'


def main_worker(gpu, cfg):
    if len(cfg.use_cuda)>1:
        dist.init_process_group(backend = "gloo", world_size=cfg.gpu_num, init_method=cfg.dist_url, rank=gpu)
        torch.cuda.set_device(gpu)

    # from models.teacher.modules_save import teacher_main
    # from models.student.modules import student_main
    # from models.naive.modules import naive_main
    # from models.rk.modules import rk_main
    # from models.student1.stackhourglass import PSMNet
    if cfg.offcial:
        from models.official_model.stackhourglass import PSMNet
        from models.warpers import Loss_warper
    else:
        from models.official_model_try1.stackhourglass import PSMNet
        from models.warpers_regression import Loss_warper
        
    from models.official_model_try2.stackhourglass import PSMNet
    from models.warpers2 import Loss_warper
    
    model = PSMNet(192).cuda(gpu)
    if cfg.loadmodel:
        model, _ = load_model(model, None, cfg, gpu)
        
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3,
    #                         betas=(0.9, 0.999), weight_decay=1e-2, )
    # model, optimizer = load_model(model, optimizer, cfg, gpu)

    model = Loss_warper(model).cuda(gpu)
    if len(cfg.use_cuda)>1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=False)

    TrainImgLoader_disp, TestImgLoader_disp = DATASET_disp(cfg)
    del TrainImgLoader_disp
    start_time = datetime.now()


    # ! -------------------eval-------------------

    loss_all = []
    start_time = datetime.now()
    for _, data_batch in enumerate(tqdm(TestImgLoader_disp)):
        with torch.no_grad():
            model.eval()
            disp_loss, _ = test(model, data_batch, gpu)
            loss_all.append(disp_loss)
    loss_all = np.mean(loss_all, 0)
    loss_all = Error_broadcast(loss_all,len(cfg.use_cuda.split(',')))
    if main_process(gpu):
        message = 'Epoch: {}/{}. Epoch time: {}. Eval Disp EPE loss: {:.3f}, D1 loss: {:.3f}'.format(
            1, cfg.max_epoch, str(datetime.now()-start_time)[:-4],
            loss_all[0], loss_all[1])
        print(message)

    # break

if __name__ == '__main__':
    import argparse
    from utils.common import init_cfg, get_cfg
    parser = argparse.ArgumentParser(description='PSMNet')
    cfg = init_cfg(parser.parse_args())
    
    cfg.server_name = 'LARGE'
    # cfg.use_cuda = '0,1,2,3'
    cfg.use_cuda = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_cuda
    cfg.pad_size= (512, 256)
    
    cfg.want_size = (480, 640)
    # cfg.want_size = (320, 480)
    cfg.loadmodel = None
    
    cfg.num_class_ratio = 1
    # cfg.finetune=None
    
    cfg.finetune = 'kitti'
    # cfg.loadmodel = ""
    # cfg.finetune = None
    # cfg.finetune = 'driving'
    # cfg.loadmodel = 'zoo/test_104_0.58898.tar'
    # cfg.loadmodel = 'official_zoo/pretrainedsceneflow_100_00.tar'
    # cfg.loadmodel = 'zoo_rk_new/rk_scene_210_2.45909.tar'
    # cfg.loadmodel = '/home/pan/Works/code/multitask/zoo/test_7.tar'
    # cfg.loadmodel = 'zoo/test_71_0.61438.tar'
    # cfg.finetune = None
    # cfg.loadmodel = 'zoo/head_newplace_46_0.69635.tar'
    # cfg.loadmodel = 'zoo_zoo2/backbone/second_try_newloss/0928_0015_0.8207.tar'
    # cfg.loadmodel = 'zoo_zoo/backbone/slim_1.0/0723_20_0.7482.tar'
    # cfg.loadmodel = 'zoo_best/test_noheadloss_naive_driving_56_0.62600.tar'
    # cfg.loadmodel = 'zoo_zoo2/backbone/first_try_newloss/0925_0020_0.7639.tar'
    cfg.loadmodel = 'zoo_zoo2/try2/from_amax_kldiv_kitti/1107_0900_0.5789.tar'
    # cfg.start_epoch = 0 if cfg.loadmodel is None else int(
    #     cfg.loadmodel.split('_')[1][:-4])+1
    # cfg.save_prefix = "./zoo_naive_kitti12/{}".format("naive_only_kitti12")
    cfg.slim_pecentage = 1.0
    # file_dir = "./zoo_zoo/backbone/{}".format(f"slim_{cfg.slim_pecentage}")
    file_dir = "./zoo_zoo2/backbone/{}".format(f"my_sgd")
    cfg.offcial = False
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    from datetime import datetime
    current = datetime.now()
    cfg.save_prefix = "{}/{:02}{:02}".format(file_dir, current.month, current.day)
    cfg = get_cfg(cfg)
    cfg.disp_batch = 1
    cfg.max_epoch = 20
    if len(cfg.use_cuda) > 1:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, nprocs=cfg.gpu_num,
                 args=(cfg,))
    else:
        main_worker(int(cfg.use_cuda), cfg)

