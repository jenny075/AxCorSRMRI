import time
import json

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.profiler
from torch.utils.tensorboard import SummaryWriter
import sys
import DatasetCreation
from Models import New_D_doubleconv,FilterLow,InceptionV3

from DatasetCreation import *
import LossFunctions
import os
from TrainingFunctions import *
import matplotlib
matplotlib.use('Agg')

from ESRT.model import esrt

import General_config as config
import argparse
global_args = None




def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_set', default=None, type=str)
    parser.add_argument('--path_to_results', default=None, type=str)
    parser.add_argument('--path_to_trained_model', default=None, type=str)
    parser.add_argument('--amount_of_files', default=None, type=int)
    parser.add_argument('--amount_of_slices', default=3, type=int)
    parser.add_argument('--channels', default=16, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--valid_batch_size', default=None, type=int)
    parser.add_argument('--total_samples', default=None, type=int)
    parser.add_argument('--title', default=None, type=str)
    parser.add_argument('--data_type', default="nifti", type=str)
    parser.add_argument('--use_db', default=False, type=bool)
    parser.add_argument('--gpu_device', default=None,type=str)
    parser.add_argument('--loss', default='rel', type=str)
    parser.add_argument('--use_ssim_loss', default=False, type=bool)
    parser.add_argument('--cross_validation_fold', default=None, type=int)
    parser.add_argument('--debug_mode', default=False, type=bool)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--scheduler', default='const', type=str,choices=['const', 'Plateau', 'Multiplicative'])
    parser.add_argument('--lr_g', default=None, type=float)
    parser.add_argument('--lr_d', default=None, type=float)
    parser.add_argument('--max_workers_train', default=2, type=int)
    parser.add_argument('--max_workers_valid', default=2, type=int)
    parser.add_argument('--val_epoch', default=None, type=int)
    parser.add_argument('--multiprocess', default=False, type=bool)
    parser.add_argument('--random_patches', default=False, type=bool)
    parser.add_argument('--no_train',  default=False, type=bool)
    parser.add_argument('--no_valid',  default=False, type=bool)
    parser.add_argument('--adversarial_weight_I', default=None, type=float)
    parser.add_argument('--adversarial_weight_E', default=None, type=float)
    parser.add_argument('--set_seed', default=24, type=int)
    parser.add_argument('--sig_flag', default=False, type=bool)
    parser.add_argument('--d_optimizer_step_size', default=None, type=int)
    parser.add_argument('--g_optimizer_step_size', default=None, type=int)
    parser.add_argument('--test_mode', default=False, type=bool)
    parser.add_argument('--image_save_freq_batch', default=100, type=int)
    parser.add_argument('--image_save_freq_epoch', default=100, type=int)
    parser.add_argument('--augmentation_state', default=False, type=bool)
    parser.add_argument('--cheakpoint_epoch', default=[], type=int,nargs='*')
    parser.add_argument('--transfer_learning', default=False, type=bool)
    return parser


def Data_Inittializaion (args):

    config.scheduler = args.scheduler
    config.transfer_learning = args.transfer_learning
    config.augmentation_state = args.augmentation_state
    if args.path_to_trained_model is not None:
        # config.resume = True
        config.path_to_trained_model = args.path_to_trained_model

        if not  args.transfer_learning:
            config.resume = True
        else:
            config.resume = False
            print("Loading pretrained model from dir-{} ".format(config.path_to_trained_model))

    if args.path_to_set is not None:
        config.path_to_set = args.path_to_set

    if args.path_to_results is not None:
        config.path = args.path_to_results

    if args.amount_of_files is not None:
        config.amount_of_files = args.amount_of_files
    config.cheakpoint_epoch = args.cheakpoint_epoch
    print(config.cheakpoint_epoch)
    if args.amount_of_slices is not None:
        config.amount_of_slices = args.amount_of_slices

    if args.total_samples is not None:
        config.total = args.total_samples


    config.data_type = args.data_type

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.valid_batch_size is not None:
        config.val_batch_fsimo = args.valid_batch_size

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.lr_g is not None:
        config.g_model_lr = args.lr_g

    if args.lr_d is not None:
        config.d_model_lr = args.lr_d

    if args.d_optimizer_step_size is not None:

        config.d_optimizer_step_size = args.d_optimizer_step_size

    if args.g_optimizer_step_size is not None:

        config.g_optimizer_step_size = args.g_optimizer_step_size

    if args.val_epoch is not None:
        config.val_epoch = args.val_epoch
    else:
        config.val_epoch = 10
    if args.debug_mode:
        config.debug_mode = args.debug_mode

    if args.patch_size:
        config.patch_size = args.patch_size
    config.image_save_freq_batch = args.image_save_freq_batch
    config.image_save_freq_epoch = args.image_save_freq_epoch
    if args.set_seed:
        config.set_seed = args.set_seed
        random.seed(args.set_seed)
        np.random.seed(args.set_seed)
        torch.manual_seed(args.set_seed)


    if args.adversarial_weight_I is not None:
        config.adversarial_weight_I = args.adversarial_weight_I

    if args.adversarial_weight_E is not None:
        config.adversarial_weight_E = args.adversarial_weight_E

    if args.title is not None:
        if args.cross_validation_fold is not None:
            config.cross_validation_fold = args.cross_validation_fold
            config.title_ = str(args.title) + '_fold' +str(config.cross_validation_fold) + '_'+ str(config.time_str)
        else:
            config.title_ = str(args.title) + str(config.time_str)
    config.loss = args.loss
    config.sig_flag = args.sig_flag
    if args.gpu_device is not None:
        config.multi_gpu = [int(elem) for elem in args.gpu_device.split(',')]
        if len(config.multi_gpu) > 1:
            config.use_multi_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    print("config.multi_gpu - ",config.multi_gpu)
    # Create a folder of super-resolution experiment results
    print("gpu available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print('gpu count:', str(torch.cuda.device_count()))
    start_time = time.time()

    if not config.resume:

        print("config.resume - ",config.resume)
        print("Building and creating ESRT model from scratch")
        result_dir = config.path +  config.title_
        # file = open('config_var', 'wb')
        # pickle.dump(config, file)
        # file.close()
        #

        if not config.debug_mode:
            os.makedirs(result_dir, exist_ok=True)
            #utils.json_dump(config.data_json, result_dir)
            # save_pickle(config, 'config_var', result_dir)
            # save_pickle(args, 'args_var', result_dir)
        else:
            print("DEBUG MODE")

            # with open(config, 'w') as f:
            #     f.writelines()
        # Create training process log file

    else:
        if config.train_separate:
            result_dir = config.path + config.title_
            if not config.debug_mode:
                os.makedirs(result_dir, exist_ok=True)
                #utils.json_dump(config.data_json, result_dir)
                path_to_model = config.path_to_trained_model
                print("Loading pretrained model from dir-{} ".format(path_to_model))
            else:
                print("DEBUG MODE")
        else:
            if ".pth" in config.path_to_trained_model:
                result_dir = config.path_to_trained_model.split("Saved")[0]
            else:
                result_dir = config.path_to_trained_model
            print("Loading pretrained model from dir-{} ".format(result_dir))

    writer = SummaryWriter(result_dir + '/' + 'tensor_logs')

    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(result_dir + '/' + 'prof_logs'),
    #     record_shapes=True,
    #     with_stack=True)
    prof = None

    print("Load train dataset and valid dataset...")

    if not config.resume or config.train_separate:




        train_list, valid_list, test_list, flag_milti_dataset, list_train_volume, train_file_to_idx, \
            list_test_volume, test_file_to_idx, list_val_volume, val_file_to_idx = \
            split_files_list_from_db(config.path_to_set, config.amount_of_files, config.total,
                                     config.amount_of_slices,max_slices=None \
                                     ,  patch_size=config.patch_size,
                                     fold=config.cross_validation_fold,
                                     state=config.augmentation_state)

        with open(result_dir + '/' + "train_list.json", "w") as fp:
            json.dump(train_list, fp)

        with open(result_dir + '/' + "valid_list.json", "w") as fp:
            json.dump(valid_list, fp)

        with open(result_dir + '/' + "test_list.json", "w") as fp:
            json.dump(test_list, fp)


        with open(result_dir + '/' + "train_file_to_idx.json", "w") as fp:
            json.dump(train_file_to_idx, fp)

        with open(result_dir + '/' + "val_file_to_idx.json", "w") as fp:
            json.dump(val_file_to_idx, fp)

        with open(result_dir + '/' + "test_file_to_idx.json", "w") as fp:
            json.dump(test_file_to_idx, fp)

    else:

        with open(result_dir + '/' + "train_list.json", 'r') as f:
            train_list = json.load(f)

        with open(result_dir + '/' + "valid_list.json", "r") as f:
            valid_list = json.load(f)

        with open(result_dir + '/' + "test_list.json", "r") as f:
            test_list = json.load(f)

        if len([config.path_to_set]) > 1:
            flag_milti_dataset = True
        else:
            flag_milti_dataset = False


        with open(result_dir + '/' + "train_file_to_idx.json", 'r') as f:
            train_file_to_idx = json.load(f)

        with open(result_dir + '/' + "val_file_to_idx.json", "r") as f:
            val_file_to_idx = json.load(f)

        with open(result_dir + '/' + "test_file_to_idx.json", "r") as f:
            test_file_to_idx = json.load(f)

        list_train_volume = DatasetCreation.create_volume_list(train_file_to_idx, config.patch_size, train=True)
        list_val_volume = DatasetCreation.create_volume_list(val_file_to_idx, config.patch_size, train=False)
        list_test_volume = DatasetCreation.create_volume_list(test_file_to_idx, config.patch_size, train=False)

    valid_batch = config.val_batch_fsimo
    print("Load train dataset and valid dataset...")
    dataset_vl_lr = DatasetCreation.CustomDataset_Test(slices = config.amount_of_slices, file_list = sorted(valid_list[0]),lr=True,file_type =  config.data_type,
                batch = valid_batch, patch_size=config.patch_size,use_db=args.use_db,
                                                           list_volumes=list_val_volume[0], file_to_idx=val_file_to_idx )
    dl_valid_lr = torch.utils.data.DataLoader(dataset=dataset_vl_lr, batch_size=valid_batch,
                                           num_workers=args.max_workers_valid,prefetch_factor=4)
    dataset_vl_hr = DatasetCreation.CustomDataset_Test(slices = config.amount_of_slices, file_list = sorted(valid_list[1]),lr=False,file_type =  config.data_type,
                batch = valid_batch, patch_size=config.patch_size,use_db=args.use_db,
                                                                  list_volumes=list_val_volume[1], file_to_idx=val_file_to_idx)
    dl_valid_hr = torch.utils.data.DataLoader(dataset=dataset_vl_hr, batch_size=valid_batch,
                                           num_workers=args.max_workers_valid,prefetch_factor=4)


    dataset_test_lr = DatasetCreation.CustomDataset_Test(slices = config.amount_of_slices, file_list = sorted(test_list[0]),lr=True,file_type =  config.data_type,
                batch = valid_batch,patch_size=config.patch_size,use_db=args.use_db,
                                                           list_volumes=list_test_volume[0], file_to_idx=test_file_to_idx )
    dl_test_lr = torch.utils.data.DataLoader(dataset=dataset_test_lr, batch_size=valid_batch,
                                           num_workers=args.max_workers_valid,prefetch_factor=4)
    dataset_test_hr = DatasetCreation.CustomDataset_Test(slices = config.amount_of_slices, file_list = sorted(test_list[1]),lr=False,file_type =  config.data_type
                ,batch = valid_batch, patch_size=config.patch_size,use_db=args.use_db,
                                                                  list_volumes=list_test_volume[1], file_to_idx=test_file_to_idx)
    dl_test_hr = torch.utils.data.DataLoader(dataset=dataset_test_hr, batch_size=valid_batch,
                                           num_workers=args.max_workers_valid,prefetch_factor=4)


    dataset_train = DatasetCreation.CustomDataset_Train(slices=config.amount_of_slices, file_list=train_list,
                                                             file_type=config.data_type
                                                             , batch=config.batch_size,
                                                             patch_size=config.patch_size,
                                                              use_db=args.use_db,
                                                             list_volumes=list_train_volume,
                                                             file_to_idx=train_file_to_idx,
                                                             random_patch=args.random_patches)
    dl_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config.batch_size,
                                           num_workers=args.max_workers_train ,
                                           worker_init_fn=worker_init_fn, drop_last=True, shuffle=True,
                                           prefetch_factor=4)

    print("Load train dataset and valid dataset successfully.")
    print ("Finish data peparation and parameters Initializaion")
    return dl_train , dl_valid_lr,dl_valid_hr,dl_test_lr,dl_test_hr,result_dir,writer,config

# def Initialize_Model_Param(args):
#
#
#     #Initlialize generator and discriminators




def training_validation_test(dl_train , dl_valid_lr,dl_valid_hr,dl_test_lr,dl_test_hr,result_dir,writer,args):

    print("Build model...")
    generator = esrt.ESRT(upscale=1, sig_flag=config.sig_flag)
    discriminator_I = New_D_doubleconv()
    discriminator_E = New_D_doubleconv()

    if config.use_multi_gpu:
        generator = nn.DataParallel(generator, config.multi_gpu).to(config.device)
        discriminator_I = nn.DataParallel(discriminator_I, config.multi_gpu).to(config.device)
        discriminator_E = nn.DataParallel(discriminator_E, config.multi_gpu).to(config.device)
    print("Build model successfully.")
    print("Define all optimizer functions...")
    d_optimizer_I = optim.Adam(discriminator_I.parameters(), config.d_model_lr, config.d_model_betas)
    d_optimizer_E = optim.Adam(discriminator_E.parameters(), config.d_model_lr, config.d_model_betas)
    g_optimizer = optim.Adam(generator.parameters(), config.g_model_lr, config.g_model_betas)
    print("Define all optimizer functions successfully.")
    print("Define all optimizer scheduler functions...")
    if config.scheduler == 'const':
        d_scheduler_I = lr_scheduler.StepLR(d_optimizer_I, config.d_optimizer_step_size, config.d_optimizer_gamma)
        d_scheduler_E = lr_scheduler.StepLR(d_optimizer_E, config.d_optimizer_step_size, config.d_optimizer_gamma)
        g_scheduler = lr_scheduler.StepLR(g_optimizer, config.g_optimizer_step_size, config.g_optimizer_gamma)

    elif config.scheduler == 'Plateau':
        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, factor=0.1, patience=5, verbose=True)
        d_scheduler_E = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer_E, factor=0.1, patience=5, verbose=True)
        d_scheduler_I = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer_I, factor=0.1, patience=5, verbose=True)

    elif config.scheduler == 'Multiplicative':
        config.lambda1 = lambda epoch: 0.99
        g_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(g_optimizer, config.lambda1)
        d_scheduler_E = torch.optim.lr_scheduler.MultiplicativeLR(d_optimizer_E, config.lambda1)
        d_scheduler_I = torch.optim.lr_scheduler.MultiplicativeLR(d_optimizer_I, config.lambda1)
    print("Define all optimizer scheduler functions successfully.")
    print("Define all loss functions...")
    color_criterion = nn.L1Loss().to(config.device)
    if config.loss == 'L1':
        adversarial_criterion = nn.BCEWithLogitsLoss().to(config.device)
    else:
        adversarial_criterion = nn.MSELoss().to(config.device)
    print("Define all loss functions successfully.")
    print("Check whether the training weight is restored...")

    flag_resume = False
    start_epoch = 0
    ratio_list1, ratio_list2 = 1,0
    max_size_lr = [0,0]
    max_size_hr = [0,0]
    best_fid = 1e6
    best_kid = 1e6
    fid = 0
    kid = 0

    best_msid_original = 1e6
    best_msid_original_avg = 1e6
    count = 0
    if config.resume:
        if ".pth" in config.path_to_trained_model:
            checkpoint = torch.load(config.path_to_trained_model)
        else:
            checkpoint = torch.load(config.path_to_trained_model + 'Saved/Periodic_save/periodic.pth')
            # checkpoint = torch.load(config.path_to_trained_model + 'Saved/PSNR/best.pth')
            # checkpoint = torch.load(config.path_to_trained_model + '/last_save.pth')

        discriminator_I.load_state_dict(checkpoint["Discriminator_I"], strict=config.strict)
        discriminator_E.load_state_dict(checkpoint["Discriminator_E"], strict=config.strict)
        generator.load_state_dict(checkpoint["Generator"], strict=config.strict)
        d_scheduler_I.load_state_dict(checkpoint["D_scheduler_I"])
        d_scheduler_E.load_state_dict(checkpoint["D_scheduler_E"])
        g_scheduler.load_state_dict(checkpoint["G_scheduler"])
        d_optimizer_I.load_state_dict(checkpoint["D_opt_I"])
        d_optimizer_E.load_state_dict(checkpoint["D_opt_E"])
        g_optimizer.load_state_dict(checkpoint["G_opt"])
        start_epoch = checkpoint["epoch"]
        ratio_list1 = checkpoint["ratio_list1"]
        ratio_list2 = checkpoint["ratio_list2"]
        flag_resume = True
        if "best_msid_original" in checkpoint:
            best_msid_original = checkpoint["best_msid_original"]

        if "best_msid_original_avg" in checkpoint:
            best_msid_original_avg = checkpoint["best_msid_original_avg"]

        if "count" in checkpoint:
            count = checkpoint["count"]
        max_size_lr = checkpoint["max_size_lr"]
        max_size_hr = checkpoint["max_size_hr"]
    print("Check whether the training weight is restored successfully.")
    scaler = amp.GradScaler()
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    InceptionV3_model = InceptionV3([block_idx]).to(config.device)
    InceptionV3_model = InceptionV3_model
    print("Start train the model.")
    start_all_train = time.time()
    step = 0
    for epoch in range(start_epoch, config.epochs):
        start_train = time.time()

        train(discriminator_I,discriminator_E,
              generator,
              dl_train,
              color_criterion,
              adversarial_criterion,
              d_optimizer_I,d_optimizer_E,
              g_optimizer,d_scheduler_I,d_scheduler_E, g_scheduler,
              args.scheduler,
              args.loss,
              epoch,
              scaler,
              writer, config)
        end_train = time.time()
        print('Epoch train time-', time.strftime("%H:%M:%S", time.gmtime(end_train - start_train)))

        if epoch % config.val_epoch == 0:
            start_val = time.time()
            fid, kid, step, max_size_lr, max_size_hr = validate(generator, InceptionV3_model, dl_valid_lr,
                                                                         dl_valid_hr,
                                                                         epoch, writer, step,
                                                                         result_dir,
                                                                         config.patch_size,
                                                                         max_size_lr, max_size_hr,config)
            end_val = time.time()
            print('Epoch val time-', time.strftime("%H:%M:%S", time.gmtime(end_val - start_val)))

        end_epoch = time.time()

        print('Total epoch time - ', time.strftime("%H:%M:%S", time.gmtime(end_epoch - start_train)))
        #FID
        is_best_fid = fid < best_fid
        best_fid = min(fid, best_fid)

        #KID
        is_best_kid = kid < best_kid
        best_kid = min(kid, best_kid)
        if not config.debug_mode :
            print("periodic save ")
            if not os.path.isdir(result_dir +'/Saved'+ '/Periodic_save'):
                os.makedirs(result_dir +'/Saved' +'/Periodic_save', exist_ok=True)
                torch.save({'Discriminator_I': discriminator_I.state_dict(),
                            'Discriminator_E': discriminator_E.state_dict(),
                            'Generator': generator.state_dict(),
                            'D_scheduler_I': d_scheduler_I.state_dict(),
                            'D_scheduler_E': d_scheduler_E.state_dict(),
                            'G_scheduler': g_scheduler.state_dict(),
                            'D_opt_I': d_optimizer_I.state_dict(),
                            'D_opt_E': d_optimizer_E.state_dict(),
                            'G_opt': g_optimizer.state_dict(),
                            'epoch': epoch+1,
                            'ratio_list1': ratio_list1,
                            'ratio_list2': ratio_list2,
                            "best_msid_original" : best_msid_original,
                            "best_fid": best_fid,
                            "best_kid": best_kid,
                            'count':count,
                            'max_size_lr' :max_size_lr,
                            'max_size_hr':max_size_hr}, os.path.join(result_dir,'Saved','Periodic_save', f"periodic.pth"))

        if not config.debug_mode and epoch in config.cheakpoint_epoch:
            print("checkpoint save {}".format(epoch))
            if not os.path.isdir(result_dir +'/Saved'+ '/check_points'):
                os.makedirs(result_dir +'/Saved' +'/check_points', exist_ok=True)
            torch.save({
                'Discriminator_I': discriminator_I.state_dict(),
                'Discriminator_E': discriminator_E.state_dict(),
                'Generator': generator.state_dict(),
                'D_scheduler_I': d_scheduler_I.state_dict(),
                'D_scheduler_E': d_scheduler_E.state_dict(),
                'G_scheduler': g_scheduler.state_dict(),
                'D_opt_I': d_optimizer_I.state_dict(),
                'D_opt_E': d_optimizer_E.state_dict(),
                'G_opt': g_optimizer.state_dict(),
                'epoch': epoch+1,
                'ratio_list1': ratio_list1,
                'ratio_list2': ratio_list2,
                "best_msid_original" : best_msid_original,
                "best_fid": best_fid,
                "best_kid": best_kid,
                'count':count,
                'max_size_lr' :max_size_lr,
                'max_size_hr':max_size_hr}, os.path.join(result_dir,'Saved', 'check_points',str(epoch)+".pth"))

        if not config.debug_mode and is_best_fid:
            if not os.path.isdir(result_dir + '/Saved' + '/FID'):
                os.makedirs(result_dir + '/Saved' + '/FID', exist_ok=True)
            # torch.save(discriminator.state_dict(), os.path.join(result_dir,'Saved','PSNR',"d-best.pth"))
            # torch.save(generator.state_dict(), os.path.join(result_dir,'Saved','PSNR',f"g-best.pth"))
            torch.save({
                'Discriminator_I': discriminator_I.state_dict(),
                'Discriminator_E': discriminator_E.state_dict(),
                'Generator': generator.state_dict(),
                'D_scheduler_I': d_scheduler_I.state_dict(),
                'D_scheduler_E': d_scheduler_E.state_dict(),
                'G_scheduler': g_scheduler.state_dict(),
                'D_opt_I': d_optimizer_I.state_dict(),
                'D_opt_E': d_optimizer_E.state_dict(),
                'G_opt': g_optimizer.state_dict(),
                'epoch': epoch + 1,
                'ratio_list1': ratio_list1,
                'ratio_list2': ratio_list2,
                "best_fid_original": best_fid,
                'count': count,
                'max_size_lr': max_size_lr,
                'max_size_hr': max_size_hr}, os.path.join(result_dir, 'Saved', 'FID', f"best.pth"))

        if not config.debug_mode and is_best_kid:
            if not os.path.isdir(result_dir + '/Saved' + '/KID'):
                os.makedirs(result_dir + '/Saved' + '/KID', exist_ok=True)
            # torch.save(discriminator.state_dict(), os.path.join(result_dir,'Saved','PSNR',"d-best.pth"))
            # torch.save(generator.state_dict(), os.path.join(result_dir,'Saved','PSNR',f"g-best.pth"))
            torch.save({
                'Discriminator_I': discriminator_I.state_dict(),
                'Discriminator_E': discriminator_E.state_dict(),
                'Generator': generator.state_dict(),
                'D_scheduler_I': d_scheduler_I.state_dict(),
                'D_scheduler_E': d_scheduler_E.state_dict(),
                'G_scheduler': g_scheduler.state_dict(),
                'D_opt_I': d_optimizer_I.state_dict(),
                'D_opt_E': d_optimizer_E.state_dict(),
                'G_opt': g_optimizer.state_dict(),
                'epoch': epoch + 1,
                'ratio_list1': ratio_list1,
                'ratio_list2': ratio_list2,
                "best_fid_original": best_fid,
                'count': count,
                'max_size_lr': max_size_lr,
                'max_size_hr': max_size_hr}, os.path.join(result_dir, 'Saved', 'KID', f"best.pth"))
    end_all_train = time.time()
    torch.save({
        'Discriminator_I': discriminator_I.state_dict(),
        'Discriminator_E': discriminator_E.state_dict(),
        'Generator': generator.state_dict(),
        'D_scheduler_I': d_scheduler_I.state_dict(),
        'D_scheduler_E': d_scheduler_E.state_dict(),
        'G_scheduler': g_scheduler.state_dict(),
        'D_opt_I': d_optimizer_I.state_dict(),
        'D_opt_E': d_optimizer_E.state_dict(),
        'G_opt': g_optimizer.state_dict(),
        'epoch': epoch + 1,
        'ratio_list1': ratio_list1,
        'ratio_list2': ratio_list2,
        "best_msid_original": best_msid_original,
        'best_msid_original_avg': best_msid_original_avg,
        'count': count,
        'max_size_lr': max_size_lr,
        'max_size_hr': max_size_hr}, os.path.join(result_dir, "last_save.pth"))
    print("END training")
    print('Total training time - ', time.strftime("%H:%M:%S", time.gmtime(end_all_train - start_all_train)))
    print("Start Testing")
    list_results = []
    temp_dict = {}
    FID_original_mean, FID_original_std, FID_original_lr_mean, FID_original_lr_std, \
    KID_original_mean, KID_original_std, KID_original_lr_mean, KID_original_lr_std = Test(generator, dl_test_lr,dl_test_lr,result_dir
                      ,config.patch_size,InceptionV3_model)


    temp_dict['TEST FID SR-HR weight mean'] = FID_original_mean
    temp_dict['TEST FID SR-HR weight std'] = FID_original_std
    temp_dict['TEST FID LR-HR weight mean'] = FID_original_lr_mean
    temp_dict['TEST FID LR-HR weight std'] = FID_original_lr_std
    temp_dict['TEST KID SR-HR weight mean'] = KID_original_mean
    temp_dict['TEST KID SR-HR weight std'] = KID_original_std
    temp_dict['TEST KID LR-HR weight mean'] = KID_original_lr_mean
    temp_dict['TEST KID LR-HR weight std'] = KID_original_lr_std
    list_results.append(temp_dict)

    keys = list_results[0].keys()
    with open(result_dir + "/" + "results_new.csv", "w") as file:
        csvwriter = csv.DictWriter(file, keys)
        csvwriter.writeheader()
        csvwriter.writerows(list_results)
    print(list_results)
