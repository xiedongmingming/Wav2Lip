from os.path import dirname, join, basename, isfile

from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip

import audio

import torch

from torch import nn
from torch import optim

import torch.backends.cudnn as cudnn

from torch.utils import data as data_utils

import numpy as np

from glob import glob

import os, random, cv2, argparse

from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument(
    "--data_root",
    help="Root folder of the preprocessed LRS2 dataset",
    required=True,
    type=str
)
parser.add_argument(
    '--checkpoint_dir',
    help='Save checkpoints to this directory',
    required=True,
    type=str
)
parser.add_argument(
    '--syncnet_checkpoint_path',
    help='Load the pre-trained Expert discriminator',
    required=True,
    type=str
)
parser.add_argument(
    '--checkpoint_path',
    help='Resume from this checkpoint',
    default=None,
    type=str
)

args = parser.parse_args()

global_step = 0
global_epoch = 0

use_cuda = torch.cuda.is_available()

print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


class Dataset(object):

    def __init__(self, split):
        #
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        #
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):

        start_id = self.get_frame_id(start_frame)

        vidname = dirname(start_frame)

        window_fnames = []

        for frame_id in range(start_id, start_id + syncnet_T):

            frame = join(vidname, '{}.jpg'.format(frame_id))

            if not isfile(frame):
                #
                return None

            window_fnames.append(frame)

        return window_fnames

    def read_window(self, window_fnames):  # 改动

        if window_fnames is None:
            #
            return None

        window = []

        for fname in window_fnames:

            img = cv2.imread(fname)

            if img is None:
                #
                return None

            try:

                img = cv2.resize(img, (hparams.img_size, hparams.img_size))

            except Exception as e:

                print('resize: {}'.format(e))

                return None

            window.append(img)

        return window  # list{ 5 * ndarray: (96, 96, 3) }

    def crop_audio_window(self, spec, start_frame):

        if type(start_frame) == int:  # 改动：支持直接指定ID
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)  # 0-indexing ---> 1-indexing

        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):  # 改动：

        mels = []

        assert syncnet_T == 5

        start_frame_num = self.get_frame_id(start_frame) + 1  # +1？？？0-indexing ---> 1-indexing

        if start_frame_num - 2 < 0:
            #
            return None

        for i in range(start_frame_num, start_frame_num + syncnet_T):

            m = self.crop_audio_window(spec, i - 2)

            if m.shape[0] != syncnet_mel_step_size:
                #
                return None

            mels.append(m.T)

        mels = np.asarray(mels)

        return mels  # {ndarray: (5, 80, 16)}

    def prepare_window(self, window):  # 改动：
        #
        # 3 x T x H x W
        #
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x  # {ndarray: (3, 5, 96, 96)}

    def __len__(self):
        #
        return len(self.all_videos)

    def __getitem__(self, idx):

        while 1:

            idx = random.randint(0, len(self.all_videos) - 1)  # 随机获取一个视频索引

            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))

            if len(img_names) <= 3 * syncnet_T:
                #
                continue

            right_img_name = random.choice(img_names)  # 随机选一个正样本（图片名称）
            wrong_img_name = random.choice(img_names)  # 随机选一个负样本（图片名称）

            while wrong_img_name == right_img_name:
                #
                wrong_img_name = random.choice(img_names)

            ########################################################################
            right_window_fnames = self.get_window(right_img_name)  # 正样例（窗口）
            wrong_window_fnames = self.get_window(wrong_img_name)  # 负样例（窗口）

            if right_window_fnames is None or wrong_window_fnames is None:
                #
                continue

            right_window = self.read_window(right_window_fnames)

            if right_window is None:
                #
                continue

            wrong_window = self.read_window(wrong_window_fnames)

            if wrong_window is None:
                #
                continue

            ########################################################################
            try:

                wavpath = join(vidname, "audio.wav")

                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T  # {ndarray: (108, 80)}

            except Exception as e:
                #
                print('melspectrogram: {}'.format(e))

                continue

            mel = self.crop_audio_window(orig_mel.copy(), right_img_name)  # {ndarray: (16, 80)}：针对正样本图片200MS区间内的MEL频谱数据

            if mel.shape[0] != syncnet_mel_step_size:
                #
                continue

            ########################################################################
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), right_img_name)  # {ndarray: (5, 80, 16)}？？？当前图片前一帧图片对应的音频

            if indiv_mels is None:
                #
                continue

            ########################################################################
            right_window = self.prepare_window(right_window)  # {ndarray: (3, 5, 96, 96)}

            y = right_window.copy()

            right_window[:, :, right_window.shape[2] // 2:] = 0.  # 下半部分清空

            wrong_window = self.prepare_window(wrong_window)  # {ndarray: (3, 5, 96, 96)}

            ##########################################################
            x = np.concatenate([right_window, wrong_window], axis=0)  # {Tensor: (6, 5, 96, 96)}

            x = torch.FloatTensor(x)

            mel = torch.FloatTensor(mel.T).unsqueeze(0)  # {Tensor: (1, 80, 16)}

            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)  # 附近[-1:4]帧对应的MEL值

            y = torch.FloatTensor(y)  # y: {Tensor: (3, 5, 96, 96)}
            #
            # x: {Tensor: (6, 5, 96, 96)} -- 正样本[下半部分空]+负样本
            # y: {Tensor: (3, 5, 96, 96)} -- 正样本
            #
            # mel: {Tensor: (1, 80, 16)} -- 正样本音频
            #
            # indiv_mels: {Tensor: (5, 1, 80, 16)} -- 正样本[-1:4]音频(共5个)
            #
            return x, indiv_mels, mel, y


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    #
    # x: 正样本[下半部分空]+负样本
    # g: {Tensor: (16, 3, 5, 96, 96)}
    #
    # gt: 正样本
    #
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)  # 预测

    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]  # 负样本 正样本[下半部分空]

    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))

    if not os.path.exists(folder):
        #
        os.mkdir(folder)

    collage = np.concatenate((refs, inps, g, gt), axis=-2)  # 列合并

    for batch_idx, c in enumerate(collage):

        for t in range(len(c)):
            #
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    #
    d = nn.functional.cosine_similarity(a, v)

    loss = logloss(d.unsqueeze(1), y)

    return loss


device = torch.device("cuda" if use_cuda else "cpu")

syncnet = SyncNet().to(device)

for p in syncnet.parameters():
    #
    p.requires_grad = False

recon_loss = nn.L1Loss()


def get_sync_loss(mel, g):
    #
    g = g[:, :, :, g.size(3) // 2:]

    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    #
    # B, 3 * T, H//2, W
    #
    a, v = syncnet(mel, g)

    y = torch.ones(g.size(0), 1).float().to(device)

    return cosine_loss(a, v, y)


def train(
        device,
        model,
        train_data_loader,
        tests_data_loader,
        optimizer,
        checkpoint_dir=None,
        checkpoint_interval=None,
        nepochs=None
):
    #
    global global_step, global_epoch

    resumed_step = global_step

    while global_epoch < nepochs:
        #
        print('starting epoch: {}'.format(global_epoch))

        running_sync_loss, running_l1_loss = 0., 0.

        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (x, indiv_mels, mel, gt) in prog_bar:
            #
            # x: {Tensor: (N, 6, 5, 96, 96)} -- 正样本[下半部分空]+负样本
            # y: {Tensor: (N, 3, 5, 96, 96)} -- 正样本
            #
            # mel: {Tensor: (N, 1, 80, 16)} -- 正样本音频
            #
            # indiv_mels: {Tensor: (N, 5, 1, 80, 16)} -- 正样本[-1:4]音频(共5个)
            #
            model.train()

            optimizer.zero_grad()

            x = x.to(device)  # move data to cuda device

            mel = mel.to(device)

            indiv_mels = indiv_mels.to(device)

            gt = gt.to(device)

            g = model(indiv_mels, x)  # 预测的样本输出 {Tensor: (16, 3, 5, 96, 96)}

            # torch.cat([g[:, :, i] for i in range(g.size(2))], dim=3)

            # cv2.imwrite('temp/ddd.jpg', np.transpose(g.cpu().detach().numpy(), (0, 2, 3, 4, 1))[0][0]*255)
            # cv2.imwrite('temp/ddd.jpg', np.transpose(torch.cat([g[:, :, i] for i in range(g.size(2))], dim=3).cpu().detach().numpy(), (0, 2, 3, 1))[0]*255)

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss

            loss.backward()

            optimizer.step()

            if global_step % checkpoint_interval == 0:
                #
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1

            cur_session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()

            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                #
                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    checkpoint_dir,
                    global_epoch
                )

            if global_step == 1 or global_step % hparams.eval_interval == 0:

                with torch.no_grad():

                    average_sync_loss = eval_model(
                        tests_data_loader,
                        global_step,
                        device,
                        model,
                        checkpoint_dir
                    )

                    if average_sync_loss < .75:
                        #
                        hparams.set_hparam('syncnet_wt', 0.01)  # without image GAN a lesser weight is sufficient

            prog_bar.set_description('L1: {}, Sync Loss: {}'.format(
                running_l1_loss / (step + 1),
                running_sync_loss / (step + 1))
            )

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    #
    eval_steps = 700

    print('evaluating for {} steps'.format(eval_steps))

    sync_losses, recon_losses = [], []

    step = 0

    while 1:

        for x, indiv_mels, mel, gt in test_data_loader:

            step += 1

            model.eval()

            x = x.to(device)  # move data to cuda device

            gt = gt.to(device)

            indiv_mels = indiv_mels.to(device)

            mel = mel.to(device)

            g = model(indiv_mels, x)

            sync_loss = get_sync_loss(mel, g)

            l1loss = recon_loss(g, gt)

            sync_losses.append(sync_loss.item())

            recon_losses.append(l1loss.item())

            if step > eval_steps:
                #
                averaged_sync_loss = sum(sync_losses) / len(sync_losses)

                averaged_recon_loss = sum(recon_losses) / len(recon_losses)

                print('L1: {}, Sync loss: {}'.format(averaged_recon_loss, averaged_sync_loss))

                return averaged_sync_loss


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    #
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step)
    )

    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)

    print("saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    #
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    #
    global global_step

    global global_epoch

    print("Load checkpoint from: {}".format(path))

    checkpoint = _load(path)

    s = checkpoint["state_dict"]

    new_s = {}

    for k, v in s.items():
        #
        new_s[k.replace('module.', '')] = v

    model.load_state_dict(new_s)

    if not reset_optimizer:
        #
        optimizer_state = checkpoint["optimizer"]

        if optimizer_state is not None:
            #
            print("load optimizer state from {}".format(path))

            optimizer.load_state_dict(checkpoint["optimizer"])

    if overwrite_global_states:
        #
        global_step = checkpoint["global_step"]

        global_epoch = checkpoint["global_epoch"]

    return model


# !python wav2lip_train.py --data_root F:\datasets\lrs2_preprocessed --checkpoint_dir ./checkpoints/ --syncnet_checkpoint_path F:\datasets\syncnet_checkpoints\checkpoint_step000211000.pth
if __name__ == "__main__":
    #
    checkpoint_dir = args.checkpoint_dir

    #
    # dataset and dataloader setup
    #
    train_dataset = Dataset('train')
    tests_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
    )

    tests_data_loader = data_utils.DataLoader(
        tests_dataset,
        batch_size=hparams.batch_size,
        # shuffle=False,
        num_workers=4,
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = Wav2Lip().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        #
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        #
        os.mkdir(checkpoint_dir)

    # Train!
    train(
        device,
        model,
        train_data_loader,
        tests_data_loader,
        optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.checkpoint_interval,
        nepochs=hparams.nepochs
    )
