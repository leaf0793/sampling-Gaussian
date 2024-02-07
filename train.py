import numpy as np
from models import *

# from datasets import 
import torch
import torch.nn as nn
import torch.distributed as dist
# from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from datetime import datetime
import os
# os.environ['KL_div'] = '1'
from utils.train_util import *
from tqdm import tqdm
# from tqdm import tqdm, trange


os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'


def main_worker(gpu, cfg):
    if len(cfg.use_cuda)>1:
        dist.init_process_group(backend = "gloo", world_size=cfg.gpu_num, init_method=cfg.dist_url, rank=gpu)
    # if(main_process(gpu)):
    #     writer = SummaryWriter('runs/exp/{}'.format(date.today()))
    # from models.teacher.modules_save import teacher_main
    # from models.student.modules import student_main
    # from models.naive.modules import naive_main
    # from models.rk.modules import rk_main
    # from models.student1.stackhourglass import PSMNet
    #!
    from models.official_model_try2.stackhourglass import PSMNet
    from models.warpers2 import Loss_warper
    #! my model
    # from models.official_model_try1.stackhourglass import PSMNet
    # from models.warpers_regression import Loss_warper
    #! official model 
    # from models.warpers import Loss_warper
    # from models.official_model.stackhourglass import PSMNet
    if len(cfg.use_cuda)>1:
        torch.cuda.set_device(gpu)
    model = PSMNet(192).cuda(gpu)
    if cfg.loadmodel:
        model, _ = load_model(model, None, cfg, gpu)
        cfg.start_epoch = 0 if 'kitti' not in cfg.loadmodel else cfg.start_epoch
    model = Loss_warper(model).cuda(gpu)
    if len(cfg.use_cuda)>1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3,
                            betas=(0.9, 0.999), weight_decay=1e-2, )
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=10, verbose=True, threshold=3e-3)
    # model, optimizer = load_model(model, optimizer, cfg, gpu)
        # adjust_learning_rate(optimizer=optimizer, epoch=epoch)
    TrainImgLoader_disp, TestImgLoader_disp = DATASET_disp(cfg)
    small_test_loss = 100
    # for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
    for epoch in (pbar := tqdm(range(cfg.start_epoch, cfg.max_epoch+1))):
        adjust_learning_rate(cfg=cfg, optimizer=optimizer, epoch=epoch)
        # TrainImgLoader_disp.sampler.set_epoch(epoch)
        epoch_loss = []
        start_time = datetime.now()
        if 1:
            for batch_idx, data_batch in enumerate(TrainImgLoader_disp):
                # if (batch_idx > len(TrainImgLoader_disp)*(epoch-10)/40 and epoch < 50) or \
                #         (batch_idx > (len(TrainImgLoader_disp)/10) and epoch <= 10):
                #     break
                
                # ! step 1: train disp

                loss, loss_disp, loss_head = train(model, data_batch, gpu, optimizer)
                # ! end 
                epoch_loss.append(float(loss))
                if main_process(gpu):
                    # len(TrainImgLoader_disp)/10
                    base = len(TrainImgLoader_disp)//10 if len(TrainImgLoader_disp)>10 else len(TrainImgLoader_disp)-1
                    if  batch_idx % base == 0:
                        message = 'Epoch: {}/{}. Iteration: {}/{}. LR:{:.1e},  Epoch time: {}. Disp loss: {:.3f}. '.format(
                            epoch, cfg.max_epoch, batch_idx, len(TrainImgLoader_disp),                    
                            float(optimizer.param_groups[0]['lr']), str(datetime.now()-start_time)[:-4],
                            loss)
                        pbar.set_description(f"{message}")

        # ! -------------------eval-------------------
        if eval_epoch(epoch, cfg=cfg) or epoch==cfg.max_epoch:
            loss_all = []
            start_time = datetime.now()
            for _, data_batch in enumerate(TestImgLoader_disp):
                with torch.no_grad():
                    model.eval()
                    disp_loss, _ = test(model, data_batch, gpu)
                    loss_all.append(disp_loss)
            loss_all = np.mean(loss_all, 0)
            loss_all = Error_broadcast(loss_all,len(cfg.use_cuda.split(',')))
            if main_process(gpu):
                # writer.add_scalar('full test/Loss', loss_all, epoch)
                message = 'Epoch: {}/{}. Epoch time: {}. Eval Disp loss: {:.3f}. D1 loss: {:.4f}'.format(
                    epoch, cfg.max_epoch, str(datetime.now()-start_time)[:-4],
                    loss_all[0], loss_all[1])
                with open(f"{cfg.file_dir}/log.txt","a+") as F:
                    F.write(f"{message}\n")
                print(message)
                if cfg.finetune != 'kitti' or epoch%100==0:
                    save_model_dict(epoch, model.state_dict(),
                                        optimizer.state_dict(), loss_all[0],cfg)

if __name__ == '__main__':
    import argparse
    from utils.common import init_cfg, get_cfg
    parser = argparse.ArgumentParser(description='PSMNet')
    cfg = init_cfg(parser.parse_args())
    
    cfg.server_name = 'LARGE'
    cfg.use_cuda = '2'
    # cfg.use_cuda = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_cuda
    cfg.use_cuda = '0'
    cfg.pad_size= (512, 256)
    
    cfg.want_size = (480, 640)
    # cfg.want_size = (320, 480)
    cfg.loadmodel = None
    
    cfg.finetune=None
    # cfg.finetune = 'kitti'
    # cfg.loadmodel = 'zoo_zoo2/try2/only_head/1030_0018_0.7100.tar'
    cfg.max_epoch = 20
    if 0:
        # cfg.loadmodel = 'zoo_zoo2/try2/from_amax_kldiv/1105_0016_0.7196.tar'
        cfg.loadmodel = 'zoo_zoo2/try2/from_amax_sigma_2gpu_1.3/1124_0020_0.6257.tar'
        # cfg.loadmodel = 'zoo_zoo2/try2/from_amax_sigma_2gpu_1.2/1122_0019_0.6314.tar'
        cfg.finetune = 'kitti'

    cfg.slim_pecentage = 1.0
    file_dir = "./zoo_zoo2/try2/{}".format("from_amax_sigma_2gpu_1.3_strictness_discreate")
    # file_dir = "./zoo_zoo2/try2/{}".format("from_amax_sigma_2gpu_1.3_kitti2")
    # file_dir = "./zoo_zoo2/try2/{}".format("from_amax_sigma_2gpu_1.3_KLdiv")
    # file_dir = "./zoo_zoo2/try2/{}".format("from_amax_sigma_2gpu_1.3_test2")
    # file_dir = "zoo_zoo/backbone/{}".format("official")
    # file_dir = "./zoo_zoo2/try2/{}".format("from_amax_sigma_2gpu_1.4")
    # file_dir = "./zoo_zoo2/try2/{}".format("from_amax_sigma_2gpu_1.0_expend")
    # file_dir = "./zoo_zoo2/try2/{}".format("from_amax_sigma_2gpu_1.0")
    cfg.file_dir = file_dir
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    from datetime import datetime
    current = datetime.now()
    cfg.save_prefix = "{}/{:02}{:02}".format(file_dir, current.month, current.day)
    cfg = get_cfg(cfg)
    # cfg.max_epoch = 20
    if cfg.finetune == 'kitti':
        cfg.max_epoch = 1001
    else:
        cfg.max_epoch = 20
    cfg.disp_batch = 4

    with open(f"{cfg.file_dir}/log.txt","a+") as F:
        tmp = vars(cfg)
        for keys,values in tmp.items():
            F.write(f"{keys}: {values}\n")
        del tmp


    if len(cfg.use_cuda) > 1:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, nprocs=cfg.gpu_num,
                 args=(cfg,))
    else:
        main_worker(int(cfg.use_cuda), cfg)

