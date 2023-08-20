import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d


class SyncNet_color(nn.Module):

    def __init__(self):
        #
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(  # {Tensor: (N, 15, 48, 96)}

            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),  # 表示输入15个通道输出32个通道

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),

            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),

            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),

            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # {Tensor: (N, 512, 1, 1)}

        )

        self.audio_encoder = nn.Sequential(  # audio_sequences: {Tensor: (N, 1, 80, 16)}

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
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # {Tensor: (N, 512, 1, 1)}

        )

    def forward(self, audio_sequences, faces_sequences):  # audio_sequences := (B, dim, T)
        #
        # audio_sequences: {Tensor: (N, 1, 80, 16)}
        # faces_sequences: {Tensor: (N, 15, 48, 96)}
        #
        audio_embedding = self.audio_encoder(audio_sequences)  # {Tensor: (N, 512, 1, 1)}
        faces_embedding = self.face_encoder(faces_sequences)  # {Tensor: (N, 512, 1, 1)}

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)  # {Tensor: (N, 512)}
        faces_embedding = faces_embedding.view(faces_embedding.size(0), -1)  # {Tensor: (N, 512)}

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)  # {Tensor: (N, 512)}
        faces_embedding = F.normalize(faces_embedding, p=2, dim=1)  # {Tensor: (N, 512)}

        return audio_embedding, faces_embedding
