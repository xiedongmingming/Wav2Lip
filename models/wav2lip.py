import torch

from torch import nn
from torch.nn import functional as F

import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class Wav2Lip(nn.Module):

    def __init__(self):

        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([

            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),  # 96,96

            nn.Sequential(
                Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),

            nn.Sequential(
                Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),

        ])
        self.face_decoder_blocks = nn.ModuleList([

            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),

            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),  # 6, 6

            nn.Sequential(
                Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            ),  # 12, 12

            nn.Sequential(
                Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            ),  # 24, 24

            nn.Sequential(
                Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ),  # 48, 48

            nn.Sequential(
                Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ),

        ])  # 96,96

        self.audio_encoder = nn.Sequential(

            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),

            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),

            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),

            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),

        )

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, faces_sequences):
        #
        # audio_sequences = (B, T, 1, 80, 16)  B：BATCH_SIZE T：
        # faces_sequences = (B, F, 5, 96, 96)  B：BATCH_SIZE F：
        #
        # audio_sequences：{Tensor: (N, 5, 1, 80, 16)} [-1:4]音频数据
        # faces_sequences：{Tensor: (N, 6, 5, 96, 96)} 正样本[下半部分空]+负样本
        #
        B = audio_sequences.size(0)  # BATCHSIZE

        input_dim_size = len(faces_sequences.size())  # torch.Size([N, 6, 5, 96, 96])：5

        if input_dim_size > 4:
            #
            audio_sequences = torch.cat(
                [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0
            )  # 按第二个维度合并：{Tensor: (N, 5, 1, 80, 16)}->{Tensor: (5*N, 1, 80, 16)}
            faces_sequences = torch.cat(
                [faces_sequences[:, :, i] for i in range(faces_sequences.size(2))], dim=0
            )  # 按第三个维度合并：{Tensor: (N, 6, 5, 96, 96)}->{Tensor: (5*N, 6, 96, 96)}

        audio_embedding = self.audio_encoder(audio_sequences)  # {Tensor: (5*N, 1, 80, 16)}->{Tensor: (5*N, 512, 1, 1)}

        feats = []  # 对应7个中间状态

        x = faces_sequences  # {Tensor: (5*N, 6, 96, 96)} 对FACES进行ENCODER

        for f in self.face_encoder_blocks:
            #
            x = f(x)

            feats.append(x)

        x = audio_embedding  # {Tensor: (5*N, 512, 1, 1)} 对AUDIO进行DECODER

        for f in self.face_decoder_blocks:

            x = f(x)

            try:

                x = torch.cat((x, feats[-1]), dim=1)

            except Exception as e:

                print(x.size())
                print(feats[-1].size())

                raise e

            feats.pop()

        x = self.output_block(x)  # {Tensor: (5*N, 80, 96, 96)}->{Tensor: (5*N, 3, 96, 96)} 半张脸

        if input_dim_size > 4:

            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]->5 * {Tensor: (N, 3, 96, 96)}

            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)：5 * {Tensor: (N, 3, 96, 96)}->{Tensor: (N, 3, 5, 96, 96)}

        else:

            outputs = x

        return outputs  # {Tensor: (N, 3, 5, 96, 96)}


class Wav2Lip_disc_qual(nn.Module):

    def __init__(self):
        #
        super(Wav2Lip_disc_qual, self).__init__()

        self.faces_encoder_blocks = nn.ModuleList([

            nn.Sequential(
                nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
            ),  # 48,96

            nn.Sequential(
                nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 48,48
                nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 24,24
                nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 12,12
                nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6,6
                nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            ),

            nn.Sequential(
                nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ),

            nn.Sequential(
                nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),

        ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

        self.label_noise = .0

    def get_lower_half(self, faces_sequences):
        #
        return faces_sequences[:, :, faces_sequences.size(2) // 2:]

    def to_2d(self, faces_sequences):

        B = faces_sequences.size(0)

        faces_sequences = torch.cat([faces_sequences[:, :, i] for i in range(faces_sequences.size(2))], dim=0)

        return faces_sequences

    def perceptual_forward(self, false_faces_sequences):

        false_faces_sequences = self.to_2d(false_faces_sequences)
        false_faces_sequences = self.get_lower_half(false_faces_sequences)

        false_feats = false_faces_sequences

        for f in self.faces_encoder_blocks:
            #
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(
            self.binary_pred(false_feats).view(len(false_feats), -1),
            torch.ones((len(false_feats), 1)).cuda()
        )

        return false_pred_loss

    def forward(self, faces_sequences):

        faces_sequences = self.to_2d(faces_sequences)

        faces_sequences = self.get_lower_half(faces_sequences)

        x = faces_sequences

        for f in self.faces_encoder_blocks:
            #
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
