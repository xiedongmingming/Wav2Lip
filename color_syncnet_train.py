from os.path import dirname, join, basename, isfile

from tqdm import tqdm

from models import SyncNet_color as SyncNet

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

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument(
    "--data_root",
    help="Root folder of the preprocessed LRS2 dataset",
    required=True
)
parser.add_argument(
    '--checkpoint_dir',
    help='Save checkpoints to this directory',
    required=True,
    type=str
)
parser.add_argument(
    '--checkpoint_path',
    help='Resumed from this checkpoint',
    default=None,
    type=str
)

args = parser.parse_args()

global_step = 0  # 历史总STEP
global_epoch = 0  # 历史总EPOCH

use_cuda = torch.cuda.is_available()  # 训练的设备CPU或GPU

print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5  # 每次选取200MS的视频片段进行训练，视频的FPS为25，所以200MS对应的帧数为：25*0.2=5帧
syncnet_mel_step_size = 16  # 200MS对应的声音的MEL-SPECTROGRAM特征的长度为16.


class Dataset(object):
    #
    def __init__(self, split):

        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):  # f:/workspace/archive/lrs2_preprocessed/6234169082415778641/00029/22.jpg
        #
        return int(basename(frame).split('.')[0])  # 视频帧索引编号

    def get_window(self, start_frame):

        start_id = self.get_frame_id(start_frame)

        vidname = dirname(start_frame)

        window_fnames = []

        for frame_id in range(start_id, start_id + syncnet_T):  # 随后的200MS内的所有帧

            frame = join(vidname, '{}.jpg'.format(frame_id))

            if not isfile(frame):  # 必须是连贯的帧
                #
                return None

            window_fnames.append(frame)

        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        #
        # num_frames = (T x hop_size * fps) / sample_rate
        #
        start_frame_num = self.get_frame_id(start_frame)

        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def __len__(self):
        #
        return len(self.all_videos)

    def __getitem__(self, idx):
        """
        return: x,mel,y
        x: 五张嘴唇图片
        mel：对应的语音的MEL-SPECTROGRAM
        t：同步OR不同步
        """
        while 1:

            idx = random.randint(0, len(self.all_videos) - 1)  # 随机选一个视频

            vidname = self.all_videos[idx]  # 对应视频文件夹：lrs2_preprocessed/6234169082415778641/00029

            img_names = list(glob(join(vidname, '*.jpg')))  # 该视频对应的所有帧

            if len(img_names) <= 3 * syncnet_T:  # 3*200MS
                #
                continue

            img_name = random.choice(img_names)  # 随机选一个正样本（图片名称）

            wrong_img_name = random.choice(img_names)  # 随机选一个负样本（图片名称）

            while wrong_img_name == img_name:
                #
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):  # 随机决定是产生负样本还是正样本

                y = torch.ones(1).float()  # 标签

                chosen = img_name  # 输入

            else:

                y = torch.zeros(1).float()

                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)  # 目标图片对应位置200MS内的所有帧名称

            if window_fnames is None:
                #
                continue

            window = []  # 存放获取到的帧（调整过尺寸的）

            all_read = True  # 是否都成功

            for fname in window_fnames:

                img = cv2.imread(fname)

                if img is None:
                    #
                    all_read = False

                    break

                try:

                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))  # 调整尺寸

                except Exception as e:

                    print('resize: {}'.format(e))

                    all_read = False

                    break

                window.append(img)

            if not all_read: continue

            try:

                wavpath = join(vidname, "audio.wav")

                wav = audio.load_wav(wavpath, hparams.sample_rate)  # {ndarray: (29696,)}

                orig_mel = audio.melspectrogram(wav).T  # {ndarray: (149, 80)}

            except Exception as e:

                print('melspectrogram: {}'.format(e))

                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)  # {ndarray: (16, 80)}：针对正样本图片200MS区间内的MEL频谱数据

            if mel.shape[0] != syncnet_mel_step_size:
                #
                continue
            #
            # H x W x 3 * T
            #
            x = np.concatenate(window, axis=2) / 255.  # {ndarray: (96, 96, 15)}：200MS内的图片数据合并

            x = x.transpose(2, 0, 1)  # 转置：(15像素值，长，宽)

            x = x[:, x.shape[1] // 2:]  # 下半部分

            x = torch.FloatTensor(x)  # {Tensor: {15, 48, 96}}

            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            # x: {Tensor: (15, 48, 96)}
            # mel: {Tensor: (1, 80, 16)}
            # y: {Tensor: (1,)}
            return x, mel, y

        ###########################################################################


# 损失函数的定义
logloss = nn.BCELoss()  # 交叉熵损失


def cosine_loss(a, v, y):  # 余弦相似度损失
    """
    a: AUDIO_ENCODER的输出
    v: VIDEO FACE_ENCODER的输出
    y: 是否同步的真实值
    """
    d = nn.functional.cosine_similarity(a, v)

    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(
        device,
        model,
        train_data_loader,
        tests_data_loader,
        optimizer,
        checkpoint_dir=None,
        checkpoint_interval=None,
        nepochs=None  # 指定的EPOCH数
):
    #
    global global_step, global_epoch

    resumed_step = global_step

    while global_epoch < nepochs:

        running_loss = 0.

        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (x, mel, y) in prog_bar:

            model.train()

            optimizer.zero_grad()

            x = x.to(device)  # 补全模型的训练：transform data to cuda device

            mel = mel.to(device)

            a, v = model(mel, x)  # 数据输入：

            y = y.to(device)

            loss = cosine_loss(a, v, y)

            loss.backward()

            optimizer.step()

            global_step += 1

            cur_session_steps = global_step - resumed_step

            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                #
                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    checkpoint_dir,
                    global_epoch
                )

            if global_step % hparams.syncnet_eval_interval == 0:
                #
                with torch.no_grad():
                    #
                    eval_model(
                        tests_data_loader,
                        global_step,
                        device,
                        model,
                        checkpoint_dir
                    )

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    #
    # 在测试集上进行评估
    #
    eval_steps = 1400

    print('evaluating for {} steps'.format(eval_steps))

    losses = []

    while 1:

        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            x = x.to(device)  # transform data to cuda device

            mel = mel.to(device)

            a, v = model(mel, x)

            y = y.to(device)

            loss = cosine_loss(a, v, y)

            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)

        print(averaged_loss)

        return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    #
    # 保存训练的结果CHECKPOINT
    #
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step)
    )

    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None

    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
        },
        checkpoint_path
    )

    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    #
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    #
    # 读取指定CHECKPOINT的保存信息
    #
    global global_step
    global global_epoch

    print("load checkpoint from: {}".format(path))

    checkpoint = _load(path)

    model.load_state_dict(checkpoint["state_dict"])

    if not reset_optimizer:  # 不重置优化器

        optimizer_state = checkpoint["optimizer"]

        if optimizer_state is not None:
            #
            print("load optimizer state from {}".format(path))

            optimizer.load_state_dict(checkpoint["optimizer"])

    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


# ds = Dataset("train")
#
# x, mel, t = ds[0]
#
# print(x.shape)  # 图像数据：torch.Size([15, 48, 96])
# print(mel.shape)  # 音频数据：torch.Size([1, 80, 16])
# print(t.shape)  # 标签数据：torch.Size([1])
#
# import matplotlib.pyplot as plt
#
# plt.imshow(mel[0].numpy())
# plt.imshow(x[:3, :, :].transpose(0, 2).numpy())

if __name__ == "__main__":

    checkpoint_dir = args.checkpoint_dir  # 保存CHECKPOINT的位置

    checkpoint_path = args.checkpoint_path  # 指定加载CHECKPOINT的路径，第一次训练时不需要，后续如果想从某个CHECKPOINT恢复训练，可指定。

    if not os.path.exists(checkpoint_dir):
        #
        os.mkdir(checkpoint_dir)

    #
    # 创建数据集和数据加载器
    #
    train_dataset = Dataset('train')
    tests_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=hparams.syncnet_batch_size,
        shuffle=True,
        num_workers=1  # hparams.num_workers
    )

    tests_data_loader = data_utils.DataLoader(
        tests_dataset,
        batch_size=hparams.syncnet_batch_size,
        # shuffle=False,
        num_workers=1  # 8
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    model = SyncNet().to(device)  # 定义SYNNET模型并加载到指定的DEVICE上

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam(  # 定义优化器，使用ADAM（LR参考HPARAMS.PY文件)
        [p for p in model.parameters() if p.requires_grad],  # 指定需要优化的参数
        lr=hparams.syncnet_lr
    )

    if checkpoint_path is not None:  # 加载预训练的模型
        #
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(
        device,
        model,
        train_data_loader,
        tests_data_loader,
        optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.syncnet_checkpoint_interval,
        nepochs=hparams.nepochs
    )
