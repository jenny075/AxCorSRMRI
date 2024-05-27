
import os, time



batch_size = 100
num_workers = 12
# parameters for train dataset

time_str = time.strftime("_%d_%m_%Y_%H_%M")
print('Start Time-' + str(time_str))
title = ""
title_ = str(title) + str(time_str)
path =""
device = 0
use_multi_gpu = False

color_weight = 1

adversarial_weight_I = 1e-1
adversarial_weight_E = 2e-1


# Adam optimizer parameter for Discriminator
d_model_lr = 2
g_model_lr = 2
d_model_betas = (0.9, 0.999)
g_model_betas = (0.9, 0.999)

epochs = 1000

debug_mode = False
# MultiStepLR scheduler parameter for SRGAN

d_optimizer_step_size = 1e5
g_optimizer_step_size = 4e5


d_optimizer_gamma = 0.5
g_optimizer_gamma = 0.5
resume = False
path_to_trained_model = ""
strict = False
start_epoch = 0
print_frequency = 20
Multi_mode = True

train_separate = False
train_G = True
train_from_start = False
train_from_start_epoch = 0


load_data_list = ""
path_to_data = ""
LPF_kernel =  5
HF = True
patch_size=64
tau = 1
lambda_ = 1e-4
data_type = None
cross_validation_fold = None