from os.path import dirname, join, basename, isfile

from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual

import audio

import torch

from torch import nn
from torch.nn import functional as F
from torch import optim

import torch.backends.cudnn as cudnn

from torch.utils import data as data_utils

import numpy as np

from glob import glob

import os, random, cv2, argparse

from hparams import hparams, get_image_list

##############################################################################
parser = argparse.ArgumentParser(description='code to train the wav2Lip model with the visual quality discriminator')

parser.add_argument(
    "--data_root",
    help="root folder of the preprocessed lrs2 dataset",
    # required=True, # TODO
    type=str,
    default="F:\datasets\lrs2\lrs2_preprocessed"  # TODO
)
parser.add_argument(
    '--checkpoint_dir',
    help='save checkpoints to this directory',
    # required=True, # TODO
    type=str,
    default="F:\datasets\wav2lip96\lrs2\checkpoint_hqwav2lip"  # TODO
)
parser.add_argument(
    '--syncnet_checkpoint_path',
    help='load the pre-trained expert discriminator',
    # required=True, # TODO
    type=str,
    default="F:\datasets\wav2lip96\lrs2\checkpoint_syncnet\checkpoint_step000380000.pth"  # TODO
)
parser.add_argument(
    '--checkpoint_path',
    help='resume generator from this checkpoint',
    default="F:\datasets\wav2lip96\lrs2\checkpoint_hqwav2lip\checkpoint_step000165000.pth",  # TODO
    type=str
)
parser.add_argument(
    '--disc_checkpoint_path',
    help='resume quality disc from this checkpoint',
    default="F:\datasets\wav2lip96\lrs2\checkpoint_hqwav2lip\disc_checkpoint_step000165000.pth",  # TODO
    type=str
)

args = parser.parse_args()

##############################################################################
global_steps = 0
global_epoch = 0

use_cuda = torch.cuda.is_available()

print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


##############################################################################
class Dataset(object):

    def __init__(self, split):

        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):

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

    def read_window(self, window_fnames):

        if window_fnames is None: return None

        window = []

        for fname in window_fnames:

            img = cv2.imread(fname)

            if img is None:
                #
                return None

            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        #
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)

        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):

        mels = []

        assert syncnet_T == 5

        start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing

        if start_frame_num - 2 < 0: return None

        for i in range(start_frame_num, start_frame_num + syncnet_T):

            m = self.crop_audio_window(spec, i - 2)

            if m.shape[0] != syncnet_mel_step_size:
                #
                return None

            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        #
        # 3 x T x H x W
        #
        x = np.asarray(window) / 255.

        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        #
        return len(self.all_videos)

    def __getitem__(self, idx):

        while 1:

            idx = random.randint(0, len(self.all_videos) - 1)

            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))

            if len(img_names) <= 3 * syncnet_T:
                #
                continue

            right_img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)

            while wrong_img_name == right_img_name:
                #
                wrong_img_name = random.choice(img_names)

            right_window_fnames = self.get_window(right_img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)

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

            try:

                wavpath = join(vidname, "audio.wav")

                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T

            except Exception as e:

                continue

            mel = self.crop_audio_window(orig_mel.copy(), right_img_name)

            if mel.shape[0] != syncnet_mel_step_size:
                #
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), right_img_name)

            if indiv_mels is None:
                #
                continue

            right_window = self.prepare_window(right_window)

            y = right_window.copy()

            right_window[:, :, right_window.shape[2] // 2:] = 0.

            wrong_window = self.prepare_window(wrong_window)

            x = np.concatenate([right_window, wrong_window], axis=0)

            x = torch.FloatTensor(x)

            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)

            y = torch.FloatTensor(y)
            #
            # x: {Tensor: (6, 5, 96, 96)} -- 正样本[下半部分空]+负样本
            # y: {Tensor: (3, 5, 96, 96)} -- 正样本
            #
            # mel: {Tensor: (1, 80, 16)} -- 正样本音频
            #
            # indiv_mels: {Tensor: (5, 1, 80, 16)} -- 正样本[-1:4]音频(共5个)
            #
            return x, indiv_mels, mel, y


##############################################################################
def save_sample_images(x, g, gt, global_steps, checkpoint_dir):
    #
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]

    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_steps))

    if not os.path.exists(folder): os.mkdir(folder)

    collage = np.concatenate((refs, inps, g, gt), axis=-2)

    for batch_idx, c in enumerate(collage):

        for t in range(len(c)):
            #
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])


##############################################################################
logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    #
    d = nn.functional.cosine_similarity(a, v)

    loss = logloss(d.unsqueeze(1), y)

    return loss


##############################################################################
device = torch.device("cuda" if use_cuda else "cpu")

syncnet_model = SyncNet().to(device)

for p in syncnet_model.parameters():
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
    a, v = syncnet_model(mel, g)

    y = torch.ones(g.size(0), 1).float().to(device)

    return cosine_loss(a, v, y)


def train(
        device,
        wav2_model,
        disc_model,
        train_data_loader,
        tests_data_loader,
        wav2_optimizer,
        disc_optimizer,
        checkpoint_dir=None,
        checkpoint_interval=None,
        nepochs=None
):
    #
    global global_steps, global_epoch

    resumed_step = global_steps

    while global_epoch < nepochs:
        #
        print('Starting Epoch: {}'.format(global_epoch))

        running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.

        running_disc_real_loss, running_disc_fake_loss = 0., 0.

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
            disc_model.train()
            wav2_model.train()

            x = x.to(device)

            mel = mel.to(device)

            indiv_mels = indiv_mels.to(device)

            gt = gt.to(device)
            #
            # train generator now. remove all grads.
            #
            wav2_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            g = wav2_model(indiv_mels, x)

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            if hparams.disc_wt > 0.:
                perceptual_loss = disc_model.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + (
                    1. - hparams.syncnet_wt - hparams.disc_wt
            ) * l1loss

            loss.backward()

            wav2_optimizer.step()

            # remove all gradients before training disc
            disc_optimizer.zero_grad()

            pred = disc_model(gt)  # 真实值

            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
            disc_real_loss.backward()

            pred = disc_model(g.detach())  # 预测值

            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
            disc_fake_loss.backward()

            disc_optimizer.step()

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()

            if global_steps % checkpoint_interval == 0:
                #
                save_sample_images(x, g, gt, global_steps, checkpoint_dir)

            # Logs
            global_steps += 1

            cur_session_steps = global_steps - resumed_step

            running_l1_loss += l1loss.item()

            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if hparams.disc_wt > 0.:
                running_perceptual_loss += perceptual_loss.item()
            else:
                running_perceptual_loss += 0.

            if global_steps == 1 or global_steps % checkpoint_interval == 0:
                #
                save_checkpoint(
                    wav2_model,
                    wav2_optimizer,
                    global_steps,
                    checkpoint_dir,
                    global_epoch
                )

                save_checkpoint(
                    disc_model,
                    disc_optimizer,
                    global_steps,
                    checkpoint_dir,
                    global_epoch,
                    prefix='disc_'
                )

            if global_steps % hparams.eval_interval == 0:
                #
                with torch.no_grad():

                    average_sync_loss = eval_model(
                        tests_data_loader,
                        global_steps,
                        device,
                        wav2_model,
                        disc_model
                    )

                    if average_sync_loss < .75:
                        #
                        hparams.set_hparam('syncnet_wt', 0.03)

            prog_bar.set_description('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(
                running_l1_loss / (step + 1),
                running_sync_loss / (step + 1),
                running_perceptual_loss / (step + 1),
                running_disc_fake_loss / (step + 1),
                running_disc_real_loss / (step + 1)
            ))

        global_epoch += 1


def eval_model(test_data_loader, global_steps, device, model, disc):
    #
    eval_steps = 300

    print('Evaluating for {} steps'.format(eval_steps))

    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []

    while 1:

        for step, (x, indiv_mels, mel, gt) in enumerate((test_data_loader)):

            model.eval()

            disc.eval()

            x = x.to(device)

            mel = mel.to(device)

            indiv_mels = indiv_mels.to(device)

            gt = gt.to(device)

            pred = disc(gt)

            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

            g = model(indiv_mels, x)

            pred = disc(g)

            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            sync_loss = get_sync_loss(mel, g)

            if hparams.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + (
                    1. - hparams.syncnet_wt - hparams.disc_wt
            ) * l1loss

            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())

            if hparams.disc_wt > 0.:
                running_perceptual_loss.append(perceptual_loss.item())
            else:
                running_perceptual_loss.append(0.)

            if step > eval_steps: break

        print('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(
            sum(running_l1_loss) / len(running_l1_loss),
            sum(running_sync_loss) / len(running_sync_loss),
            sum(running_perceptual_loss) / len(running_perceptual_loss),
            sum(running_disc_fake_loss) / len(running_disc_fake_loss),
            sum(running_disc_real_loss) / len(running_disc_real_loss)
        ))

        return sum(running_sync_loss) / len(running_sync_loss)


def save_checkpoint(model, optimizer, steps, checkpoint_dir, epoch, prefix=''):
    #
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_steps)
    )

    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_steps": steps,
        "global_epoch": epoch,
    }, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    #
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    #
    global global_steps
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

        optimizer_state = checkpoint["optimizer"]

        if optimizer_state is not None:
            #
            print("Load optimizer state from {}".format(path))

            optimizer.load_state_dict(checkpoint["optimizer"])

    if overwrite_global_states:
        #
        global_steps = checkpoint["global_steps"]
        global_epoch = checkpoint["global_epoch"]

    return model


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
        num_workers=1,  # hparams.num_workers
    )
    tests_data_loader = data_utils.DataLoader(
        tests_dataset,
        batch_size=hparams.batch_size,
        # shuffle=True,
        num_workers=1,  # 4
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    # model
    wav2_model = Wav2Lip().to(device)
    disc_model = Wav2Lip_disc_qual().to(device)

    print('total trainable params: wav2lip {}'.format(
        sum(p.numel() for p in wav2_model.parameters() if p.requires_grad)
    ))
    print('total trainable params: wavdisc {}'.format(
        sum(p.numel() for p in disc_model.parameters() if p.requires_grad)
    ))

    wav2_optimizer = optim.Adam(
        [p for p in wav2_model.parameters() if p.requires_grad],
        lr=hparams.initial_learning_rate,
        betas=(0.5, 0.999)
    )
    disc_optimizer = optim.Adam(
        [p for p in disc_model.parameters() if p.requires_grad],
        lr=hparams.disc_initial_learning_rate,
        betas=(0.5, 0.999)
    )

    if args.checkpoint_path is not None:
        #
        load_checkpoint(
            args.checkpoint_path,
            wav2_model,
            wav2_optimizer,
            reset_optimizer=False
        )

    if args.disc_checkpoint_path is not None:
        #
        load_checkpoint(
            args.disc_checkpoint_path,
            disc_model,
            disc_optimizer,
            reset_optimizer=False,
            overwrite_global_states=False
        )

    load_checkpoint(
        args.syncnet_checkpoint_path,
        syncnet_model,
        None,
        reset_optimizer=True,
        overwrite_global_states=False
    )

    if not os.path.exists(checkpoint_dir):
        #
        os.mkdir(checkpoint_dir)

    # train!
    train(
        device,
        wav2_model,
        disc_model,
        train_data_loader,
        tests_data_loader,
        wav2_optimizer,
        disc_optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.checkpoint_interval,
        nepochs=hparams.nepochs
    )
