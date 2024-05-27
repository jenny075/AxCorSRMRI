
import time

import torch.profiler
from Models import New_D_doubleconv ,FilterLow,InceptionV3

from DatasetCreation import *
import LossFunctions

import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

global_args = None
import csv
from scipy import signal
import General_config as config


def save_tensor_to_img(tensor,path_to_save,file_path ='_' ):
    """

    :param tensor:
    :param path_to_data:
    :param path_to_save:
    :param title:
    :return:
    """
    print(file_path)

    reader = sitk.ImageFileReader()
    # The option for nii images

    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file_path)
    original_file = reader.Execute()
    tensor_arr = tensor.cpu().detach().numpy()

    image = sitk.GetImageFromArray(tensor_arr)

    origin = original_file.GetOrigin()
    direction = original_file.GetDirection()
    image.SetOrigin(origin)
    image.SetDirection(direction)
    image.SetSpacing(original_file.GetSpacing())

    file_name = file_path.split('.nii.gz')[0].split('/')[-1]


    #dir_path = os.path.dirname(path)
    ending = '.nii.gz'
    writer = sitk.ImageFileWriter()
    if "COR" in file_path:
        output_name = path_to_save + '/'+ file_name + "COR_SR_"  +  ending
    elif "AX" in file_path:
        output_name = path_to_save + '/' + file_name + "AX_SR_" + ending
    else:
        output_name = path_to_save+'/'+file_name+"_SR_"+ending
    writer.SetFileName(output_name)
    writer.Execute(image)


def save_tensor(tensor,tensot_type,title,result_dir):
    """
    this function receives tensot type torch and saves it as torch tensor
    :param tensor:
    :param tensot_type:
    :param title:
    :param result_dir:
    :return:
    """

    print(tensor.shape)
    title_ = title.split('.nii')[0].split("/")[-1]
    path = result_dir + '/' + tensot_type + "_" + title_ +'.pt'
    torch.save(tensor, path)
    return



def visualize_Multi_slice(lr, sr, epoch, path,title_,rec_title,index, writer,step):

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(9, 12))
    lr = lr.squeeze(dim=1)
    sr = sr.squeeze(dim=1)
    title = rec_title + 'epoch-' + str(epoch) + '_batch_' + str(index)
    imgs = []
    for k in range((config.amount_of_slices + 2)*2):
        imgs.append('img'+str(k))

    lr_dim = lr.dim()
    lr = torch.rot90(lr, 2, [lr_dim - 2, lr_dim -1])
    sr_dim = sr.dim()
    sr = torch.rot90(sr, 2, [sr_dim - 2, sr_dim -1])



    for i in range(2):

        img1 = axs[i * 2, 0].imshow(np.squeeze(lr.cpu())[9+ i,0,:,:], vmin=0, vmax=1,cmap='gray')
        axs[i * 2, 0].set_title('Input - 1')
        fig.colorbar(img1, ax=axs[i * 2, 0],fraction=0.046, pad=0.04)

        img2 = axs[i * 2,1].imshow(np.squeeze(lr.cpu())[9 + i,1, :, :], vmin=0, vmax=1,cmap='gray')
        axs[i * 2, 1].set_title('Input ')
        fig.colorbar(img2, ax=axs[i * 2, 1],fraction=0.046, pad=0.04)


        img3 = axs[i * 2, 2].imshow(np.squeeze(lr.cpu())[9 + i,2, :, :], vmin=0, vmax=1,cmap='gray')
        axs[i * 2, 2].set_title('Input +1 ')
        fig.colorbar(img3, ax=axs[i * 2, 2],fraction=0.046, pad=0.04)



        img4 = axs[i * 2+1, 0].imshow(np.squeeze(lr.cpu())[9+ i,1,:,:], vmin=0, vmax=1,cmap='gray')
        axs[i * 2+1, 0].set_title('Input')
        fig.colorbar(img4, ax=axs[i * 2+1, 0],fraction=0.046, pad=0.04)
        img5 = axs[i * 2+1, 1].imshow(np.squeeze(sr.detach().cpu())[9 + i], vmin=0, vmax=1,cmap='gray')
        axs[i * 2+1, 1].set_title('Gen')
        fig.colorbar(img5, ax=axs[i * 2+1, 1],fraction=0.046, pad=0.04)
        img6 = axs[i * 2+1, 2].imshow(np.abs(np.squeeze(lr.cpu())[9 + i,1,:,:]- np.squeeze(sr.detach().cpu())[9 + i])
                                    , vmin=0, vmax=1,cmap='gray')
        axs[i * 2+1, 2].set_title('Diff')
        fig.colorbar(img6, ax=axs[i * 2+1, 2],fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    if not config.debug_mode:
        title_save = title + '_epoch_' +str(epoch)
        plt.savefig(path + '/' + '_' + title_save+ '.png')
        writer.add_figure(title_, fig, global_step=step)
        step += 1
        return step

    else:
        plt.show()


def visualize_Multi_slice_test(lr, sr, path, title_,test_=False):

    if test_:
        test_string = 'test'
        os.makedirs(path+'/test', exist_ok=True)

    else:
        test_string = 'val'
        os.makedirs(path + '/val', exist_ok=True)

    lr_dim = lr.dim()
    lr = torch.rot90(lr, 2, [lr_dim - 2, lr_dim -1])
    sr_dim = sr.dim()
    sr = torch.rot90(sr, 2, [sr_dim - 2, sr_dim -1])
    path = path + '/' + test_string
    title = title_.split("/")[-1] + "_" + title_.split("/")[0]
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(9, 12))
    lr = lr.squeeze(dim=1)
    sr = sr.squeeze(dim=1)

    imgs = []
    for k in range((3 + 2)*2):
        imgs.append('img'+str(k))


    for i in range(2):

        img1 = axs[i * 2, 0].imshow(np.squeeze(lr.cpu())[9+ i,0,:,:], vmin=0, vmax=1,cmap='gray')
        axs[i * 2, 0].set_title('Input - 1')
        axs[i * 2, 0].grid(False)
        axs[i * 2, 0].axis('off')

        img2 = axs[i * 2,1].imshow(np.squeeze(lr.cpu())[9 + i,1, :, :], vmin=0, vmax=1,cmap='gray')
        axs[i * 2, 1].set_title('Input ')
        axs[i * 2, 1].grid(False)
        axs[i * 2, 1].axis('off')


        img3 = axs[i * 2, 2].imshow(np.squeeze(lr.cpu())[9 + i,2, :, :], vmin=0, vmax=1,cmap='gray')
        axs[i * 2, 2].set_title('Input +1 ')
        axs[i * 2, 2].grid(False)
        axs[i * 2, 2].axis('off')



        img4 = axs[i * 2+1, 0].imshow(np.squeeze(lr.cpu())[9+ i,1,:,:], vmin=0, vmax=1,cmap='gray')
        axs[i * 2+1, 0].set_title('Input')
        axs[i * 2+1, 0].grid(False)
        axs[i * 2+1, 0].axis('off')

        img5 = axs[i * 2+1, 1].imshow(np.squeeze(sr.detach().cpu())[9 + i], vmin=0, vmax=1,cmap='gray')
        axs[i * 2+1, 1].set_title('Gen')

        axs[i * 2+1, 1].grid(False)
        axs[i * 2+1, 1].axis('off')

        img6 = axs[i * 2+1, 2].imshow(np.abs(np.squeeze(lr.cpu())[9 + i,1,:,:]- np.squeeze(sr.detach().cpu())[9 + i])
                                    , vmin=0, vmax=1,cmap='gray')
        axs[i * 2+1, 2].set_title('Diff')
        axs[i * 2+1, 2].grid(False)
        axs[i * 2+1, 2].axis('off')


    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    title_save = title
    plt.savefig(path + '/' +'_' + title_save+ '.png' )
    plt.close()




def visualize_reconstrated_no_hr(lr_0,lr_1,lr_2, sr, epoch, path, title_,rec_title,index, writer,step):

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 12))

    title = rec_title + 'reconstracted_epoch-' + str(epoch)+ '_batch_' + str(index)
    imgs = []
    for k in range((config.amount_of_slices + 2)*2):
        imgs.append('img'+str(k))

    lr_0_dim = lr_0.dim()
    lr_0 = torch.rot90(lr_0, 2, [lr_0_dim - 2, lr_0_dim -1])
    lr_1_dim = lr_1.dim()
    lr_1 = torch.rot90(lr_1, 2, [lr_1_dim - 2, lr_1_dim -1])
    lr_2_dim = lr_2.dim()
    lr_2 = torch.rot90(lr_2, 2, [lr_2_dim - 2, lr_2_dim -1])
    sr_dim = sr.dim()
    sr = torch.rot90(sr, 2, [sr_dim - 2, sr_dim -1])
    img1 = axs[0, 0].imshow(lr_0.cpu(), vmin=0, vmax=1,cmap='gray')
    axs[0, 0].set_title('Input - 1')
    fig.colorbar(img1, ax=axs[0, 0],fraction=0.046, pad=0.04)
    img2 = axs[0,1].imshow(lr_1.cpu(), vmin=0, vmax=1,cmap='gray')
    axs[0, 1].set_title('Input ')
    fig.colorbar(img2, ax=axs[0, 1],fraction=0.046, pad=0.04)
    img3 = axs[0, 2].imshow(lr_2.cpu(), vmin=0, vmax=1,cmap='gray')
    axs[0, 2].set_title('Input +1 ')
    fig.colorbar(img3, ax=axs[0, 2],fraction=0.046, pad=0.04)

    img4 = axs[1, 0].imshow(lr_1.cpu(), vmin=0, vmax=1,cmap='gray')
    axs[1, 0].set_title('Input')
    fig.colorbar(img4, ax=axs[1, 0],fraction=0.046, pad=0.04)

    img5 = axs[1, 1].imshow(sr.detach().cpu(), vmin=0, vmax=1,cmap='gray')
    axs[1, 1].set_title('Gen')
    fig.colorbar(img5, ax=axs[1, 1],fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    if not config.debug_mode:
        title_save = title + '_epoch_' +str(epoch)
        plt.savefig(path + '/' + '_' + title_save+ '.png')
        writer.add_figure(title_, fig, global_step=step)

    else:
        plt.show()


def visualize_reconstrated(lr_0,lr_1,lr_2, sr, path, title_,test_=False):

    if test_:
        test_string = 'test'
        os.makedirs(path+'/test', exist_ok=True)

    else:
        test_string = 'val'
        os.makedirs(path + '/val', exist_ok=True)
    path = path + '/' + test_string
    title = title_.split("/")[-1] + "_" + title_.split("/")[0]
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 12))
    lr_0_dim = lr_0.dim()
    lr_0 = torch.rot90(lr_0, 2, [lr_0_dim - 2, lr_0_dim -1])
    lr_1_dim = lr_1.dim()
    lr_1 = torch.rot90(lr_1, 2, [lr_1_dim - 2, lr_1_dim -1])
    lr_2_dim = lr_2.dim()
    lr_2 = torch.rot90(lr_2, 2, [lr_2_dim - 2, lr_2_dim -1])
    sr_dim = sr.dim()
    sr = torch.rot90(sr, 2, [sr_dim - 2, sr_dim -1])
    imgs = []
    for k in range((3 + 2)*2):
        imgs.append('img'+str(k))


    img1 = axs[0, 0].imshow(lr_0.cpu(), vmin=0, vmax=1,cmap='gray')
    axs[0, 0].set_title('Input - 1')
    #fig.colorbar(img1, ax=axs[0, 0],fraction=0.046, pad=0.04)
    axs[0,0].grid(False)
    axs[0,0].axis('off')

    img2 = axs[0,1].imshow(lr_1.cpu(), vmin=0, vmax=1,cmap='gray')
    axs[0, 1].set_title('Input ')

    axs[0,1].grid(False)
    axs[0,1].axis('off')

    img3 = axs[0, 2].imshow(lr_2.cpu(), vmin=0, vmax=1,cmap='gray')
    axs[0, 2].set_title('Input +1 ')
    axs[0,2].grid(False)
    axs[0,2].axis('off')

    img4 = axs[1, 0].imshow(lr_1.cpu(), vmin=0, vmax=1,cmap='gray')
    axs[1, 0].set_title('Input')
    axs[1,0].grid(False)
    axs[1,0].axis('off')

    #fig.colorbar(img4, ax=axs[1, 0],fraction=0.046, pad=0.04)
    img5 = axs[1, 1].imshow(sr.detach().cpu(), vmin=0, vmax=1,cmap='gray')

    axs[1, 1].set_title('Gen')
    axs[1,1].grid(False)
    axs[1,1].axis('off')

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    title_save = title
    plt.savefig(path + '/' + test_string + '_' + title_save+ '.png')
    plt.close()




def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d



def pad_to_max_size(image,max_size):
    """
    This function receives image and max_size and pads with zeros the image to max_size
    :param image:
    :param max_size:
    :return:
    """

    if int(image.size()[0]) == max_size[0] and int(image.size()[1]) == max_size[1]:
        return image
    else:
        diff = max_size - np.array(image.shape)
        p2d_first_dim = (0, 0, int(diff[0]/2),diff[0] - int(diff[0]/2))  # pad last dim by (0, 0) and 2nd to last by (diff/2, diff/2)
        image = F.pad(image, p2d_first_dim, "constant", 0)
        p2d_second_dim = (int(diff[1]/2),diff[1] - int(diff[1]/2),0,0)
        image = F.pad(image, p2d_second_dim, "constant", 0)
        return image


def recon_im_torch_rectangle_gaus(patches, im_h, im_w, stride_x, stride_y,device = None):
    """Reconstruct the image from all patches.
        Patches are assumed to be square and overlapping depending on the stride. The image is constructed
         by filling in the patches from left to right, top to bottom, averaging the overlapping parts.
    Parameters
    -----------
    patches: 3D ndarray with shape (patch_number,patch_height,patch_width)
        Array containing extracted patches.
    im_h: int
        original height of image to be reconstructed
    im_w: int
        original width of image to be reconstructed

    stride_x: int
           desired patch stride in x direction

    stride_y: int
           desired patch stride in y direction
    Returns
    -----------
    reconstructedim: ndarray with shape (height, width) Reconstructed image from the given patches
    """

    patch_size = patches.shape[2]  # patches assumed to be square
    patch_weight = gkern(patches.shape[2],patches.shape[2]/2+1)
    patch_weight = (torch.ones(patches.shape)*patch_weight).to(device)
    patches = patch_weight*patches
    patches = torch.permute(patches,(1,2,0))
    patches =patches.reshape(patches.size()[0]*patches.size()[1],patches.size()[-1]).unsqueeze(0)
    patch_weight = torch.permute(patch_weight, (1, 2, 0))
    patch_weight =patch_weight.reshape(patch_weight.size()[0]*patch_weight.size()[1],patch_weight.size()[-1]).unsqueeze(0)
    reconim = torch.nn.functional.fold(patches, [im_h, im_w], kernel_size=[patch_size,patch_size],stride=[stride_x, stride_y] )
    divim = torch.nn.functional.fold(patch_weight, [im_h, im_w], kernel_size=[patch_size,patch_size],stride=[stride_x, stride_y] )

    if torch.isnan(reconim).any():
        print("Error reconim")
        print("reconim -",reconim)
    reconstructedim = reconim / (divim)
    if torch.isnan(reconstructedim).any():
        print("error")
        print("divim - ",divim)
        print("reconstructedim - ",reconstructedim)
    return reconstructedim


def train(discriminator_I,
          discriminator_E,
          generator,
          train_dataloader,
          color_criterion,
          adversarial_criterion: object,
          d_optimizer_I,
          d_optimizer_E,
          g_optimizer,
          d_scheduler_I,
          d_scheduler_E,
          g_scheduler,
          scheduler_type,
          loss_type,
          epoch,
          scaler,
          writer,
          config):
    batches = len(train_dataloader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    g_losses = AverageMeter("G loss", ":6.6f")
    d_I_losses = AverageMeter("D_I loss", ":6.6f")
    d_E_losses = AverageMeter("D_E loss", ":6.6f")
    color_losses = AverageMeter("Color loss", ":6.6f")
    adversarial_losses_I = AverageMeter("Adversarial loss I", ":6.6f")
    adversarial_losses_E = AverageMeter("Adversarial loss E", ":6.6f")
    d_hr_probabilities_I = AverageMeter("D(HR_I)", ":6.3f")
    d_hr_probabilities_E = AverageMeter("D(HR_E)", ":6.3f")
    d_sr_probabilities_I = AverageMeter("D(SR_I)", ":6.3f")
    d_sr_probabilities_E = AverageMeter("D(SR_E)", ":6.3f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, g_losses , d_I_losses,d_E_losses,
                              color_losses, adversarial_losses_I,adversarial_losses_E,d_hr_probabilities_I,
                              d_hr_probabilities_E, d_sr_probabilities_I,d_sr_probabilities_E],
                             prefix=f"Epoch: [{epoch + 1}]")
    discriminator_I.train()
    discriminator_E.train()
    generator.train()
    d_loss_I = torch.tensor([0])
    d_loss_E = torch.tensor([0])
    g_loss = torch.tensor([0])
    color_loss = 0
    end = time.time()
    LPF: FilterLow = FilterLow(recursions=1, kernel_size=config.LPF_kernel, gaussian=True).to(config.device)
    for index, (lr, hr) in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr = lr.to(config.device, non_blocking=True)
        hr = hr.to(config.device, non_blocking=True)

        # Set the real sample label to 1, and the false sample label to 0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)

        # Use generators to create super-resolution images
        sr = generator(lr.squeeze())
        if torch.isnan(sr).any():
            print("NAN")
            break
        sr = sr.squeeze(dim=2).float()
        # Start training discriminator
        # At this stage, the discriminator needs to require a derivative gradient
        adversarial_loss_I = torch.zeros(1)
        adversarial_loss_E = torch.zeros(1)


        if index % 5 == 0:


            for p in discriminator_I.parameters():
                p.requires_grad = True

            for p in discriminator_E.parameters():
                p.requires_grad = True

            # Initialize the discriminator optimizer gradient
            d_optimizer_I.zero_grad()
            d_optimizer_E.zero_grad()

            hr_output_I = discriminator_I(hr)
            hr_E = LPF(hr)
            hr_output_E = discriminator_E(hr - hr_E)

            sr_output_I = discriminator_I(sr.detach())
            sr_E = LPF(sr.detach())
            sr_output_E = discriminator_E(sr.detach() - sr_E)
            if loss_type =='rel':
                d_loss_hr_I = adversarial_criterion(hr_output_I - torch.mean(sr_output_I), real_label)
                d_loss_sr_I = adversarial_criterion(sr_output_I - torch.mean(hr_output_I), fake_label)
                d_loss_hr_E = adversarial_criterion(hr_output_E - torch.mean(sr_output_E), real_label)
                d_loss_sr_E = adversarial_criterion(sr_output_E - torch.mean(hr_output_E), fake_label)
            else:
                d_loss_hr_I = adversarial_criterion(hr_output_I, real_label)
                d_loss_sr_I = adversarial_criterion(sr_output_I, fake_label)
                d_loss_hr_E = adversarial_criterion(hr_output_E, real_label)
                d_loss_sr_E = adversarial_criterion(sr_output_E, fake_label)
            # Gradient zoom
            scaler.scale(0.5*(d_loss_hr_I+d_loss_sr_I)).backward()
            scaler.step(d_optimizer_I)
            scale = scaler.get_scale()
            d_scheduler_I.step()
            scaler.scale(0.5 * (d_loss_hr_E + d_loss_sr_E)).backward()
            scaler.step(d_optimizer_E)
            scale = scaler.get_scale()
            if scale < scaler.get_scale():
                if scheduler_type == "Plateau":
                    d_scheduler_E.step(0)
            d_scheduler_E.step()
            # Count discriminator total loss
            d_loss_I = d_loss_hr_I + d_loss_sr_I
            d_loss_E = d_loss_hr_E + d_loss_sr_E
        else:
            for p in discriminator_I.parameters():
                p.requires_grad = False

            for p in discriminator_E.parameters():
                p.requires_grad = False
            # Initialize the generator optimizer gradient
            g_optimizer.zero_grad()
            sr_output_I = discriminator_I(sr)
            sr_E = LPF(sr)
            sr_output_E = discriminator_E(sr - sr_E)
            hr_output_I = discriminator_I(hr.detach())
            hr_E = LPF(hr).detach()
            hr_output_E = discriminator_E(hr.detach() - hr_E)
            mid_slice = int(lr.shape[2] / 2)
            color_loss = config.color_weight * LossFunctions.SelfSupervisedLoss_torch(lr[:, :, mid_slice, :, :], sr,
                                                                                       color_criterion)

            if loss_type == 'rel':
                g_loss_hr_I = adversarial_criterion(hr_output_I - torch.mean(sr_output_I), fake_label)
                g_loss_sr_I = adversarial_criterion(sr_output_I - torch.mean(hr_output_I), real_label)
                g_loss_hr_E = adversarial_criterion(hr_output_E - torch.mean(sr_output_E), fake_label)
                g_loss_sr_E = adversarial_criterion(sr_output_E - torch.mean(hr_output_E), real_label)
                # d_loss_gt = adversarial_criterion(gt_output - torch.mean(sr_output), fake_label) * 0.5
                # d_loss_sr = adversarial_criterion(sr_output - torch.mean(gt_output), real_label) * 0.0
                adversarial_loss_I = config.adversarial_weight_I * 0.5 * (g_loss_hr_I + g_loss_sr_I)
                adversarial_loss_E = config.adversarial_weight_E * 0.5 * (g_loss_hr_E + g_loss_sr_E)

            else:
                adversarial_loss_I = config.adversarial_weight_I * adversarial_criterion(sr_output_I, real_label)
                adversarial_loss_E = config.adversarial_weight_E * adversarial_criterion(sr_output_E, real_label)

            adversarial_loss_tot = 1 * (adversarial_loss_I + adversarial_loss_E)

            # Count discriminator total loss

            g_loss = color_loss + adversarial_loss_tot
            if torch.isnan(g_loss):
                print("loss is nan")

            # Gradient zoom
            scaler.scale(g_loss).backward()

            # Update generator parameters
            # scaler.step(g_optimizer)
            # scaler.update()
            # g_scheduler.step()

            scaler.step(g_optimizer)
            scale = scaler.get_scale()
            # scaler.update()

            if scale < scaler.get_scale():
                if scheduler_type == "Plateau":
                    g_scheduler.step(0)

            g_scheduler.step()
        scaler.update()
        d_hr_probability_I = torch.sigmoid(torch.mean(hr_output_I.detach()))
        d_hr_probability_E = torch.sigmoid(torch.mean(hr_output_E.detach()))
        d_sr_probability_I = torch.sigmoid(torch.mean(sr_output_I.detach()))
        d_sr_probability_E = torch.sigmoid(torch.mean(sr_output_E.detach()))
        color_losses.update(color_loss, lr.size(0))

        g_losses.update(g_loss.item(),lr.size(0))
        if index % 5 != 0:
            adversarial_losses_I.update(adversarial_loss_I.item(), lr.size(0))
            adversarial_losses_E.update(adversarial_loss_E.item(), lr.size(0))
        d_I_losses.update(d_loss_I.item(), lr.size(0))
        d_E_losses.update(d_loss_E.item(), lr.size(0))
        d_hr_probabilities_I.update(d_hr_probability_I.item(), lr.size(0))
        d_hr_probabilities_E.update(d_hr_probability_E.item(), lr.size(0))
        d_sr_probabilities_I.update(d_sr_probability_I.item(), lr.size(0))
        d_sr_probabilities_E.update(d_sr_probability_E.item(), lr.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if index % config.print_frequency == 0 and index != 0:
            progress.display(index)
    writer.add_scalar("Train/D_Loss_I", d_I_losses.avg, epoch)
    writer.add_scalar("Train/D_Loss_E", d_E_losses.avg, epoch)
    writer.add_scalar("Train/G_Loss", g_losses.avg, epoch)
    writer.add_scalar("Train/Pixel_Loss", color_losses.avg, epoch)
    writer.add_scalar("Train/Adversarial_Loss", adversarial_losses_I.avg, epoch)
    writer.add_scalar("Train/Adversarial_Loss", adversarial_losses_E.avg, epoch)
    writer.add_scalar("Train/D_I(HR)_Probability", d_hr_probabilities_I.avg, epoch)
    writer.add_scalar("Train/D_E(HR)_Probability", d_hr_probabilities_E.avg, epoch)
    writer.add_scalar("Train/D_E lr", d_optimizer_E.param_groups[0]["lr"], epoch)
    writer.add_scalar("Train/D_I lr", d_optimizer_I.param_groups[0]["lr"], epoch)
    writer.add_scalar("Train/G lr", g_optimizer.param_groups[0]["lr"], epoch)



def validate(model,InceptionV3_model, valid_dataloader_lr,valid_dataloader_hr, epoch, writer,step,result_dir
                      ,patch_size,max_size_lr,max_size_hr,config) -> float:

    batch_time = AverageMeter("Time", ":6.3f")


    # fid_score = AverageMeter("FID_Score", ":4.2f")
    progress = ProgressMeter(len(valid_dataloader_hr), [batch_time], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()
    rec_hr_tot = []
    rec_sr_tot = []
    rec_mid_tot = []
    rec_mid_tot_max_size = []
    rec_sr_tot_max_size = []
    max_patches_lr = 0
    max_patches_hr = 0
    print("Start Vlidation")

    with torch.no_grad():
        end = time.time()

        if epoch==0:
            for _, (lr, size_lr,title) in enumerate(valid_dataloader_lr):
                if lr.size()[0] == 1:
                    if int(size_lr[-1])>= max_size_lr[1] and int(size_lr[-2]) >= max_size_lr[0]:
                        max_size_lr = [int(size_lr[-2]),int(size_lr[-1])]
                    if size_lr[0] > max_patches_lr:
                        max_patches_lr =  size_lr[0]
                else:
                    max_1 = int(torch.max(size_lr[-1]))
                    max_0 = int(torch.max(size_lr[-2]))
                    max_p = int(torch.max(size_lr[0]))
                    if max_1>= max_size_lr[1] and max_0 >= max_size_lr[0]:
                        max_size_lr = [max_0,max_1]
                    if max_p >  max_patches_lr:
                        max_patches_lr = max_p

            for _, (hr, size_hr,_) in enumerate(valid_dataloader_hr):
                if hr.size()[0] == 1:
                    if int(size_hr[-1]) >= max_size_hr[1] and int(size_hr[-2]) >= max_size_hr[0]:
                        max_size_hr =[int(size_hr[-2]),int(size_hr[-1])]
                else:
                    max_1 = int(torch.max(size_hr[-1]))
                    max_0 = int(torch.max(size_hr[-2]))
                    max_p = int(torch.max(size_hr[0]))
                    if max_1 >= max_size_hr[1] and max_0 >= max_size_hr[0]:
                        max_size_hr = [max_0,max_1]
                    if max_p >  max_patches_hr:
                        max_patches_hr = max_p
            print("LR max size {} max patches {}".format(max_size_lr,max_patches_lr))
            print("HR max size {} max patches {}".format(max_size_hr,max_patches_hr))
        start_2 = time.time()
        print("Start HR reconstract")
        for index, (hr_tot, size_hr_tot,_) in enumerate(valid_dataloader_hr):

            #print(size_hr_tot.size())
            sample_num = hr_tot.size()[0]
            for i in range(sample_num):
                hr = hr_tot[i]
                size_hr = [size_hr_tot[-2][i],size_hr_tot[-1][i]]
                hr = torch.squeeze(hr)
                hr = hr.to(config.device, non_blocking=True)

                rec_hr = recon_im_torch_rectangle_gaus(hr, int(size_hr[0]), int(size_hr[1]),
                                                        calculate_overlap(int(size_hr[0]), patch_size),
                                                        calculate_overlap(int(size_hr[1]), patch_size),
                                                              device = config.device).detach().cpu()\
                    .squeeze(0).squeeze(0).float()

                rec_hr_resize =  pad_to_max_size(rec_hr,max_size_hr).detach().cpu()
                rec_hr_tot.append(rec_hr_resize)

        end_1 = time.time()
        print('Finish reconstrct HR', time.strftime("%H:%M:%S", time.gmtime(end_1 - start_2)))
        start_1 = time.time()
        print("Start LR reconstract")
        for index_, (lr_tot, size_lr_tot,title_tot) in enumerate(valid_dataloader_lr):

            sample_num = lr_tot.size()[0]
            for i in range(sample_num):
                index = index_*sample_num + i
                lr = lr_tot[i]
                size_lr = [size_lr_tot[-2][i], size_lr_tot[-1][i]]
                patch_amount = size_lr_tot[0][i]
                title = str(title_tot[i])
                lr = torch.unsqueeze(torch.squeeze(lr), dim=1)
                lr = lr.to(config.device, non_blocking=True)
                middle = lr[:patch_amount,:,1,:,:].detach().float()

                try:
                    sr = model(lr.squeeze()).float()
                except:
                    raise Exception("got error with the file {},index {},size {}".format(title,index,lr.size()))

                sr = sr.squeeze(dim=2)[:patch_amount]

                lr_1_rec = recon_im_torch_rectangle_gaus(middle.squeeze(), int(size_lr[0]), int(size_lr[1]),
                                                               calculate_overlap(int(size_lr[0]), patch_size),
                                                               calculate_overlap(int(size_lr[1]), patch_size),
                                                               device = config.device).squeeze(0).squeeze(0).to(config.device,
                                                                                                                non_blocking=True).float().detach().cpu()



                rec_mid = pad_to_max_size(lr_1_rec, max_size_lr).detach().cpu()

                rec_mid_tot.append(rec_mid)
                rec_mid_tot_max_size.append(pad_to_max_size(lr_1_rec, max_size_hr))
                rec_sr = recon_im_torch_rectangle_gaus(sr.squeeze(), int(size_lr[0]), int(size_lr[1]),
                                                             calculate_overlap(int(size_lr[0]), patch_size),
                                                             calculate_overlap(int(size_lr[1]), patch_size),
                                                             device = config.device).squeeze(0).squeeze(0).to(config.device,
                                                                                                              non_blocking=True).float().detach().cpu()

                rec_sr_resize = pad_to_max_size(rec_sr, max_size_lr).detach().cpu()
                rec_sr_tot.append(rec_sr_resize)
                rec_sr_tot_max_size.append(pad_to_max_size(rec_sr, max_size_hr))


                if (index %config.image_save_freq_batch == 0 ) and epoch %config.image_save_freq_epoch ==0 :
                    #Weighted rec
                    # rec_hr = recon_im_torch_rectangle_gaus(hr, int(size_hr[0]), int(size_hr[1]),
                    #                                        calculate_overlap(int(size_hr[0]),patch_size),
                    #                                        calculate_overlap(int(size_hr[1]),patch_size))
                    # if np.isnan(rec_hr).any():
                    #     print("NAN")
                    #rec_hr_tot.append(rec_hr)
                    temp_title = "Reconstracted_"
                    #rec_sr_tot.append(rec_sr)

                    step = visualize_Multi_slice(lr, sr, epoch, result_dir, config.title_,temp_title,index ,writer, step)
                    lr_0 =  lr[:patch_amount,:,0,:,:].detach()
                    lr_0_rec = recon_im_torch_rectangle_gaus(lr_0.squeeze(), int(size_lr[0]), int(size_lr[1]),
                                                                   calculate_overlap(int(size_lr[0]),patch_size), calculate_overlap(int(size_lr[1]),patch_size), device = config.device).detach().cpu().squeeze(0).squeeze(0)
                    lr_2 =  lr[:patch_amount,:,2,:,:].detach()
                    lr_2_rec = recon_im_torch_rectangle_gaus(lr_2.squeeze(), int(size_lr[0]), int(size_lr[1]),
                                                                   calculate_overlap(int(size_lr[0]),patch_size), calculate_overlap(int(size_lr[1]),patch_size), device = config.device).detach().cpu().squeeze(0).squeeze(0)

                    visualize_reconstrated_no_hr(lr_0_rec, lr_1_rec, lr_2_rec, rec_sr,epoch,result_dir, config.title_,temp_title ,index, writer,step)


        end_2 = time.time()
        print('Finish reconstrct LR', time.strftime("%H:%M:%S", time.gmtime(end_2 - start_1)))
        print('Reconstruct all images-', time.strftime("%H:%M:%S", time.gmtime(end_2- start_2)))

        rec_hr_tensor = torch.stack(rec_hr_tot).to(config.device)

        rec_mid_tensor_max_size = torch.stack(rec_mid_tot_max_size).to(config.device)
        rec_sr_tensor_max_size = torch.stack(rec_sr_tot_max_size).to(config.device)
        start_3 = time.time()
        FID_HR_SR = LossFunctions.calculate_fretchet(rec_hr_tensor, rec_sr_tensor_max_size,
                                                         InceptionV3_model)
        FID_HR_LR = LossFunctions.calculate_fretchet(rec_hr_tensor, rec_mid_tensor_max_size,
                                                            InceptionV3_model)
        end_3 = time.time()
        print('Finish FID calc', time.strftime("%H:%M:%S", time.gmtime(end_3- start_3)))
        KID_original = []
        KID_original_avg = []
        KID_original_lr = []
        KID_original_lr_avg = []
        start_4 = time.time()
        num = 25
        for i in range(num):
            np.random.seed(i)
            KID_original.append(LossFunctions.calculate_KID(rec_hr_tensor, rec_sr_tensor_max_size,
                                                             InceptionV3_model, i))
            KID_original_lr.append(LossFunctions.calculate_KID(rec_hr_tensor, rec_mid_tensor_max_size,
                                                                InceptionV3_model, i))


        end_4 = time.time()
        print('Finish KID calc', time.strftime("%H:%M:%S", time.gmtime(end_4- start_4)))

        start_3 = time.time()

        print(f"* FID SR - HR: {np.mean(FID_HR_SR):4.2f},.\n")
        print(f"* FID LR - HR: {np.mean(FID_HR_LR):4.2f} .\n")
        print(f"* KID SR - HR: {np.mean(KID_original):4.2f} +/- {np.std(KID_original):4.2f}.\n")
        print(f"* KID LR - HR: {np.mean(KID_original_lr):4.2f} +/- {np.std(KID_original_lr):4.2f}.\n")


        writer.add_scalar("Valid/FID_HR_SR", np.mean(FID_HR_SR), epoch)
        writer.add_scalar("Valid/FID_HR_LR", np.mean(FID_HR_LR), epoch)
        writer.add_scalar("Valid/KID_HR_SR", np.mean(KID_original), epoch)
        writer.add_scalar("Valid/KID_HR_LR", np.mean(KID_original_lr), epoch)


    return np.mean(FID_HR_SR),np.mean(KID_original), step, max_size_lr,max_size_hr


def Test(model, valid_dataloader_lr,valid_dataloader_hr,result_dir,patch_size,Incep_model = None):

    result_dir_for_tensors = result_dir +"/saved_tensors"
    os.makedirs(result_dir_for_tensors, exist_ok=True)

    batch_time = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(len(valid_dataloader_lr), [batch_time], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()
    rec_hr_tot = []
    max_patches_lr = 0
    max_patches_hr = 0
    rec_sr_tot = []
    rec_mid_tot = []
    rec_mid_tot_max_size = []
    rec_sr_tot_max_size = []
    print("Start Vlidation")
    max_size_lr = [0,0]
    max_size_hr = [0,0]
    with torch.no_grad():
        end = time.time()

        for _, (lr, size_lr, title) in enumerate(valid_dataloader_lr):
            if lr.size()[0] == 1:
                if int(size_lr[-1]) >= max_size_lr[1] and int(size_lr[-2]) >= max_size_lr[0]:
                    max_size_lr = [int(size_lr[-2]), int(size_lr[-1])]
                if size_lr[0] > max_patches_lr:
                    max_patches_lr = size_lr[0]
            else:
                max_1 = int(torch.max(size_lr[-1]))
                max_0 = int(torch.max(size_lr[-2]))
                max_p = int(torch.max(size_lr[0]))
                if max_1 >= max_size_lr[1] and max_0 >= max_size_lr[0]:
                    max_size_lr = [max_0, max_1]
                if max_p > max_patches_lr:
                    max_patches_lr = max_p

        for _, (hr, size_hr, _) in enumerate(valid_dataloader_hr):
            if hr.size()[0] == 1:
                if int(size_hr[-1]) >= max_size_hr[1] and int(size_hr[-2]) >= max_size_hr[0]:
                    max_size_hr = [int(size_hr[-2]), int(size_hr[-1])]
            else:
                max_1 = int(torch.max(size_hr[-1]))
                max_0 = int(torch.max(size_hr[-2]))
                max_p = int(torch.max(size_hr[0]))
                if max_1 >= max_size_hr[1] and max_0 >= max_size_hr[0]:
                    max_size_hr = [max_0, max_1]
                if max_p > max_patches_hr:
                    max_patches_hr = max_p

        temp_file_hr_weight=[]
        temp_title = []
        for index, (hr_tot, size_hr_tot,slice_title) in enumerate(valid_dataloader_hr):

            if index == 0:
                temp_title = slice_title[0].split('_slice')[0]

            if slice_title[0].split('_slice')[0] != temp_title:

                temp_file_hr_weight_ = torch.stack(temp_file_hr_weight)

                if config.save_tensor:

                    if slice_title[0].split('_slice')[0] != temp_title:
                        print(temp_file_hr_weight_.shape)
                        save_tensor(temp_file_hr_weight_, "HR", temp_title, result_dir_for_tensors)
                temp_file_hr_weight = []

            temp_title = slice_title[0].split('_slice')[0]
            temp_file_hr_avg = []

            sample_num = hr_tot.size()[0]

            for i in range(sample_num):

                hr = hr_tot[i]

                size_hr = [size_hr_tot[-2][i],size_hr_tot[-1][i]]

                # print(size_hr)

                hr = torch.squeeze(hr)

                hr = hr[:size_hr_tot[0][i],:,:]
                hr = hr.to(config.device, non_blocking=True)

                # print(hr.shape)
                rec_hr = recon_im_torch_rectangle_gaus(hr, int(size_hr[0]), int(size_hr[1]),
                                                        calculate_overlap(int(size_hr[0]), patch_size),
                                                        calculate_overlap(int(size_hr[1]), patch_size), device = config.device).detach().cpu().squeeze(0).squeeze(0).float()
                #rec_hr_resize = pad_and_resize(rec_hr, [240, 240])
                # print(size_hr_tot)
                # print(rec_hr.shape)
                # if rec_hr.shape[-1] == 320:
                #
                #     print(slice_title)
                temp_file_hr_weight.append(rec_hr)
                rec_hr_resize = pad_to_max_size(rec_hr,max_size_hr).detach().cpu()
                rec_hr_tot.append(rec_hr_resize)





        temp_file_hr_weight_ = torch.stack(temp_file_hr_weight)
        #hr_tot_volume_list_weight_full.append(temp_file_hr_weight.detach())
        #hr_tot_volume_list_weight.append(np.array(temp_file_hr_weight.detach().cpu()).reshape(temp_file_hr_weight.size()[0], -1))
        if config.save_tensor:
                    save_tensor(temp_file_hr_weight_, "HR",  temp_title, result_dir_for_tensors)


        print("rec_hr_tot len - {} ".format(len(rec_hr_tot)))
        temp_file_sr_avg =[]
        temp_file_sr_weight=[]
        temp_file_lr_avg =[]
        temp_file_lr_weight=[]
        temp_title = []
        print("Reconstract LR images")
        for index, (lr_tot, size_lr_tot,slice_title) in enumerate(valid_dataloader_lr):
            print(slice_title)
            # if size_lr_tot[-1] == 320:
                # print("320")
                # print(slice_title)

            if index == 0:
                temp_title = slice_title[0].split('_slice')[0]
            #print(temp_title)
            if slice_title[0].split('_slice')[0] != temp_title:
                # print ("Temp file - {}\n slice title - {}".format(temp_title,slice_title[0].split('_slice')[0]))
                # print("Enter SR vol save")
                temp_file_sr_weight_ = torch.stack(temp_file_sr_weight)

                temp_file_lr_weight_ = torch.stack(temp_file_lr_weight)
                if config.save_tensor:

                    if slice_title[0].split('_slice')[0] != temp_title :
                       # print(temp_file_sr_weight_.shape)
                        save_tensor(temp_file_sr_weight_, "SR", temp_title, result_dir_for_tensors)
                        save_tensor(temp_file_lr_weight_, "LR", temp_title, result_dir_for_tensors)

                if config.save_nifti:
                    # print(temp_file_sr_weight_.shape)
                    # temp_title_ = temp_title + "_SR"
                    save_tensor_to_img(temp_file_sr_weight_, result_dir_for_tensors, temp_title)
                temp_title = slice_title[0].split('_slice')[0]
                temp_file_sr_avg = []
                temp_file_sr_weight = []
                temp_file_lr_avg = []
                temp_file_lr_weight = []
            sample_num = lr_tot.size()[0]

            for i in range(sample_num):
                index = index*sample_num + i
                lr = lr_tot[i]

                size_lr = [size_lr_tot[-2][i], size_lr_tot[-1][i]]

                patch_amount = size_lr_tot[0][i]
                title = str(slice_title[i])

                lr = torch.unsqueeze(torch.squeeze(lr), dim=1)
                lr = lr.to(config.device, non_blocking=True)
                middle = lr[:patch_amount,:,1,:,:].detach().float()
                # Mixed precision
                # with amp.autocast():
                #sr = model(lr).float()

                sr = model(lr.squeeze()).float()
                sr = torch.clamp(sr,min=0,max=1)
                sr = sr.squeeze(dim=2)[:patch_amount]
                # print(sr.shape)

                lr_1_rec = recon_im_torch_rectangle_gaus(middle.squeeze(), int(size_lr[0]), int(size_lr[1]),
                                                               calculate_overlap(int(size_lr[0]), patch_size),
                                                               calculate_overlap(int(size_lr[1]), patch_size),
                                                               device = config.device).squeeze(0).squeeze(0).float().detach().cpu()
                # if lr_1_rec.shape[-1] == 320:
                #     print(slice_title)
                temp_file_lr_weight.append(lr_1_rec)
                rec_mid = pad_to_max_size(lr_1_rec, max_size_lr).detach().cpu()
                # print(size_lr_tot)
                # print(lr_1_rec.shape)
                # rec_mid = pad_and_resize(lr_1_rec, [240, 240])
                rec_mid_tot.append(rec_mid)

                rec_mid_tot_max_size.append(pad_to_max_size(lr_1_rec, max_size_hr))


                rec_sr = recon_im_torch_rectangle_gaus(sr.squeeze(), int(size_lr[0]), int(size_lr[1]),
                                                             calculate_overlap(int(size_lr[0]), patch_size),
                                                             calculate_overlap(int(size_lr[1]), patch_size),
                                                               device = config.device).squeeze(0).squeeze(0).float().detach().cpu()
                # rec_sr_resize = pad_and_resize(rec_sr,[240,240])
                temp_file_sr_weight.append(rec_sr)
                rec_sr_resize = pad_to_max_size(rec_sr, max_size_lr).detach().cpu()
                rec_sr_tot.append(rec_sr_resize)

                rec_sr_tot_max_size.append(pad_to_max_size(rec_sr, max_size_hr))
                #temp_file_sr_weight.append(rec_sr_resize)

                # rec_mid_avg = pad_and_resize(lr_1_rec_avg, [240, 240])



                if index %config.image_save_freq_batch == 0  :

                    rec_hr = recon_im_torch_rectangle_gaus(hr, int(size_hr[0]), int(size_hr[1]), calculate_overlap(int(size_hr[0]),patch_size)
                                                           , calculate_overlap(int(size_hr[1]),patch_size), device = config.device).detach().cpu().squeeze(0).squeeze(0)
                    if np.isnan(rec_hr).any():
                        print("NAN")
                    #rec_hr_tot.append(rec_hr)

                    #rec_sr_tot.append(rec_sr)

                    visualize_Multi_slice_test(lr, sr, result_dir,'compare'+list(slice_title)[0], index)

                    lr_0 =  lr[:patch_amount,:,0,:,:].detach()
                    lr_0_rec = recon_im_torch_rectangle_gaus(lr_0.squeeze(), int(size_lr[0]), int(size_lr[1]), calculate_overlap(int(size_lr[0]),patch_size),
                                                             calculate_overlap(int(size_lr[1]),patch_size), device = config.device).detach().cpu().squeeze(0).squeeze(0)
                    lr_2 =  lr[:patch_amount,:,2,:,:].detach()
                    lr_2_rec = recon_im_torch_rectangle_gaus(lr_2.squeeze(), int(size_lr[0]), int(size_lr[1]), calculate_overlap(int(size_lr[0]),patch_size),
                                                             calculate_overlap(int(size_lr[1]),patch_size), device = config.device).detach().cpu().squeeze(0).squeeze(0)

                    visualize_reconstrated(lr_0_rec, lr_1_rec, lr_2_rec, rec_sr,result_dir,  'compare'+list(slice_title)[0], True)

                    rec_hr = recon_im_torch_rectangle_gaus(hr, int(size_hr[0]), int(size_hr[1]),
                                                                 calculate_overlap(int(size_hr[0]), patch_size),
                                                                 calculate_overlap(int(size_hr[1]), patch_size), device = config.device).detach().cpu().squeeze(0).squeeze(0)
                    if np.isnan(rec_hr).any():
                        print("NAN")
                    # rec_hr_tot.append(rec_hr)

                    # rec_sr_tot.append(rec_sr)




        temp_file_sr_weight_ = torch.stack(temp_file_sr_weight)

        temp_file_lr_weight_ = torch.stack(temp_file_lr_weight)

        if config.save_tensor:
            #print(temp_file_sr_weight_.shape)
            save_tensor(temp_file_sr_weight_, "SR",  temp_title, result_dir_for_tensors)
            save_tensor(temp_file_lr_weight_, "LR", temp_title, result_dir_for_tensors)


        if config.save_nifti:
            #print(temp_file_sr_weight_.shape)
            # temp_title_ = temp_title + "_SR"
            save_tensor_to_img(temp_file_sr_weight_,result_dir_for_tensors, temp_title )


        rec_hr_tensor = torch.stack(rec_hr_tot).to(config.device)


        print("Calculating FID")


        rec_sr_tensor_max_size = torch.stack(rec_sr_tot_max_size).to(config.device)

        FID_original = LossFunctions.calculate_fretchet(rec_hr_tensor, rec_sr_tensor_max_size,
                                          Incep_model)
        rec_sr_tensor_max_size = rec_sr_tensor_max_size.detach().cpu()
        rec_mid_tensor_max_size = torch.stack(rec_mid_tot_max_size).to(config.device)
        FID_original_lr = LossFunctions.calculate_fretchet(rec_hr_tensor, rec_mid_tensor_max_size,
                                          Incep_model)
        rec_mid_tensor_max_size = rec_mid_tensor_max_size.detach().cpu()

        print("Calculating KID")

        KID_original = []
        KID_original_avg = []
        KID_original_lr = []
        KID_original_lr_avg = []
        for i in range(25):
            np.random.seed(i)
            rec_sr_tensor_max_size = rec_sr_tensor_max_size.to(config.device)
            KID_original.append(LossFunctions.calculate_KID(rec_hr_tensor, rec_sr_tensor_max_size,
                                              Incep_model,i).numpy())
            rec_sr_tensor_max_size = rec_sr_tensor_max_size.detach().cpu()
            rec_mid_tensor_max_size = rec_mid_tensor_max_size.to(config.device)
            KID_original_lr.append(LossFunctions.calculate_KID(rec_hr_tensor, rec_mid_tensor_max_size,
                                              Incep_model,i).numpy())
            rec_mid_tensor_max_size = rec_mid_tensor_max_size.detach().cpu()
    return np.mean(FID_original), np.std(FID_original),  \
            np.mean(FID_original_lr), np.std(FID_original_lr), \
            np.mean(KID_original), np.std(KID_original), \
            np.mean(KID_original_lr), np.std(KID_original_lr)






class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

