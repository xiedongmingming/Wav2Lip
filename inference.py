from os import listdir, path

import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string

from tqdm import tqdm
from glob import glob

import torch, face_detection

from models import Wav2Lip

import platform

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument(
    '--checkpoint_path',
    type=str,
    help='Name of saved checkpoint to load weights from',
    required=True
)
parser.add_argument(
    '--face',
    type=str,
    help='Filepath of video/image that contains faces to use',
    required=True
)
parser.add_argument(
    '--audio',
    type=str,
    help='Filepath of video/audio file to use as raw audio source',
    required=True
)
parser.add_argument(
    '--outfile',
    type=str,
    help='Video path to save result. See default for an e.g.',
    default='results/result_voice.mp4'
)
parser.add_argument(  # 如果为真则只使用第一视频帧进行推理
    '--static',
    type=bool,
    help='If True, then use only first video frame for inference',
    default=False
)
parser.add_argument(  # 只能在输入为静态图像时指定（默认：25）
    '--fps',
    type=float,
    help='Can be specified only if input is a static image (default: 25)',
    default=25.,
    required=False
)
parser.add_argument(
    '--pads',
    nargs='+',
    type=int,
    default=[0, 10, 0, 0],
    help='Padding (top, bottom, left, right). Please adjust to include chin at least'
)
parser.add_argument(
    '--face_det_batch_size',
    type=int,
    help='Batch size for face detection',
    default=4
)
parser.add_argument(
    '--wav2lip_batch_size',
    type=int,
    help='Batch size for Wav2Lip model(s)',
    default=128
)
parser.add_argument(
    '--resize_factor',
    default=1,
    type=int,
    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p'
)
parser.add_argument(  # 将视频裁剪到较小的区域(上、下、左、右)。应用于RESIZE_FACTOR和ROTATE参数之后。
    '--crop',
    nargs='+',
    type=int,
    default=[0, -1, 0, -1],
    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
         'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width'
)
parser.add_argument(  # 为人脸指定一个恒定的边界框。只有在脸部没有被发现的情况下才使用。而且，只有脸部不怎么动的时候才有效。语法：(上、下、左、右)。
    '--box',
    nargs='+',
    type=int,
    default=[-1, -1, -1, -1],
    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
         'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).'
)
parser.add_argument(
    '--rotate',
    default=False,
    action='store_true',
    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
         'Use if you get a flipped result, despite feeding a normal looking video'
)
parser.add_argument(  # 防止在短时间窗口内平滑人脸检测
    '--nosmooth',
    # default=False,
    default=True,
    action='store_true',
    help='Prevent smoothing face detections over a short temporal window'
)

args = parser.parse_args()

args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    #
    args.static = True


def get_smoothened_boxes(boxes, T):
    #
    for i in range(len(boxes)):  # 没一帧的人脸

        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]

        boxes[i] = np.mean(window, axis=0)

    return boxes


def face_detect(images):
    #
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

    batch_size = args.face_det_batch_size  # 人脸检测的批大小

    while 1:

        predictions = []

        try:

            for i in tqdm(range(0, len(images), batch_size)):  # tqdm：进度条
                #
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))

        except RuntimeError:

            if batch_size == 1:
                #
                raise RuntimeError(
                    'image too big to run face detection on gpu. please use the --resize_factor argument'
                )

            batch_size //= 2

            print('recovering from oom error; new batch size: {}'.format(batch_size))

            continue

        break

    results = []

    pady1, pady2, padx1, padx2 = args.pads  # (top, bottom, left, right)

    for rect, image in zip(predictions, images):  # rect: (x1, y1, x2, y2)
        #
        # cv2.imwrite('temp/demo-001.jpg', images[0][y1:y2,x1:x2,:])
        #
        if rect is None:
            #
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.

            raise ValueError('face not detected! ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)  # y1

        y2 = min(image.shape[0], rect[3] + pady2)  # y2+pad

        x1 = max(0, rect[0] - padx1)  # x1

        x2 = min(image.shape[1], rect[2] + padx2)  # x2+pad

        results.append([x1, y1, x2, y2])

        # cv2.imwrite('temp/demo-001.jpg', images[0][y1:y2, x1:x2, :])

    boxes = np.array(results)

    if not args.nosmooth:
        #
        boxes = get_smoothened_boxes(boxes, T=5)

    results = [
        [
            image[y1: y2, x1:x2], (y1, y2, x1, x2)
        ] for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    del detector

    return results  # 得到人脸图像


def datagen(frames, mels):  # 视频帧+音频MEL
    #
    # img_batch: 人脸数据（RESIZE之后）
    # mel_batch: 对应的音频数据
    # frame_batch：视频帧
    # coords_batch：人脸坐标
    #
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    face_batch = []

    if args.box[0] == -1:  # 没有指定人脸区域

        if not args.static:  # 视频
            face_det_results = face_detect(frames)  # bgr2rgb for cnn face detection
        else:  # 图片：只使用第一视频帧进行推理
            face_det_results = face_detect([frames[0]])

    else:  # 直接使用指定的人脸区域

        print('using the specified bounding box instead of face detection...')

        y1, y2, x1, x2 = args.box

        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]  # 前面是脸部数据而后面是坐标

    for i, m in enumerate(mels):

        idx = 0 if args.static else i % len(frames)  # 索引指定帧

        frame_to_save = frames[idx].copy()

        face, coords = face_det_results[idx].copy()  # 对应的脸部区域数据和坐标

        orgin = face.copy()

        face_batch.append(orgin)

        face = cv2.resize(face, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)  # 96 ？？？

        img_batch.append(face)

        mel_batch.append(m)

        frame_batch.append(frame_to_save)

        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            #
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()  # {ndarray: (128, 96, 96, 3)}->

            img_masked[:, args.img_size // 2:] = 0  # MASK处理：下半张图片数据清空

            # 把两张图片合成一张（由3个通道变成6个通道）--前三个是MASKED数据后三个是正常数据（都是RESIZE过的）
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

            mel_batch = np.reshape(  # {ndarray: (128, 80, 16, 1)}
                mel_batch,
                [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield face_batch, img_batch, mel_batch, frame_batch, coords_batch

            face_batch, img_batch, mel_batch, frame_batch, coords_batch = [], [], [], [], []

    if len(img_batch) > 0:  # 不够一个批次？
        #
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()

        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

        mel_batch = np.reshape(
            mel_batch,
            [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        yield face_batch, img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16  #

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('using {} for inference.'.format(device))


def _load(checkpoint_path):
    #
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint


def load_model(path):
    #
    model = Wav2Lip()

    print("load checkpoint from: {}".format(path))

    checkpoint = _load(path)

    s = checkpoint["state_dict"]

    new_s = {}

    for k, v in s.items():  # ？？？把K中的MODULE去除
        #
        new_s[k.replace('module.', '')] = v

    model.load_state_dict(new_s)

    model = model.to(device)

    return model.eval()


def main():
    #
    if not os.path.isfile(args.face):  # 人脸：./input/1.mp4

        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:  # 人脸是图片

        full_frames = [cv2.imread(args.face)]

        fps = args.fps

    else:  # TODO：预处理

        video_stream = cv2.VideoCapture(args.face)

        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('reading video frames...')

        full_frames = []

        while 1:

            still_reading, frame = video_stream.read()  # frame: {ndarray: (1920, 1080, 3)}

            if not still_reading:  # 读取结束
                #
                video_stream.release()

                break

            if args.resize_factor > 1:  # 缩放因子
                #
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

            if args.rotate:  # 90度旋转
                #
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop  # 裁剪

            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("number of frames available for inference: " + str(len(full_frames)))

    ##############################################################################
    # TODO：音频处理--预处理
    if not args.audio.endswith('.wav'):
        #
        print('extracting raw audio...')

        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)

        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)  # {ndarray: (118515, )}

    mel = audio.melspectrogram(wav)  # TODO ？？？{ndarray: (80, 593)}

    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        #
        raise ValueError('mel contains nan! using a tts voice? add a small epsilon noise to the wav file and try again')

    mel_chunks = []  # 一个CHUNK对应一帧(一帧图片对应的音频MEL数据)

    mel_idx_multiplier = 80. / fps  # 3.2

    i = 0

    while 1:

        start_idx = int(i * mel_idx_multiplier)

        if start_idx + mel_step_size > len(mel[0]):
            #
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])

            break

        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])

        i += 1

    print("length of mel chunks: {}".format(len(mel_chunks)))  # list:182->{ndarray: (80, 16)}

    full_frames = full_frames[:len(mel_chunks)]  # 从视频中剪切目标音频对应时长的帧

    batch_size = args.wav2lip_batch_size  # 默认128帧作为一个批次

    gen = datagen(full_frames.copy(), mel_chunks)

    for i, (faces, img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))
    ):

        if i == 0:
            #
            model = load_model(args.checkpoint_path)

            print("model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]  # ???长宽

            out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        # {ndarray: (54, 96, 96, 6)}-->torch.Size([54, 6, 96, 96])

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            #
            pred = model(mel_batch, img_batch)  # {tensor: (128, 3, 96, 96)}

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.  # {ndarray: (128, 96, 96, 3)}

        for o, p, f, c in zip(faces, pred, frames, coords):
            #
            y1, y2, x1, x2 = c

            # cv2.imwrite('temp/1.jpg', p)

            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1), cv2.INTER_CUBIC)

            # TODO 将预测得到的P与原始人脸进行比对以便解决边框问题
            #
            for i in range(0, o.shape[0]):

                for j in range(0, o.shape[1]):
                    #
                    if abs(o[i][j][0] - 68) <= 7 and abs(o[i][j][1] - 169) <= 7 and (o[i][j][2] - 57) <= 7:
                        #
                        p[i][j][0] = 68
                        p[i][j][1] = 169
                        p[i][j][2] = 57

            # cv2.imwrite('temp/2.jpg', p)

            f[y1:y2, x1:x2] = p

            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
        args.audio,
        'temp/result.avi',
        args.outfile
    )  # 视频与音频合成

    subprocess.call(command, shell=platform.system() != 'Windows')


def save_video(frame_data, video_path, output_path='output/'):
    #
    print('save video ..')

    # video_path = cfg.DEMO.DATA_SOURCE

    cap = cv2.VideoCapture(video_path)

    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    video_name = video_path.split('/')[-1].split('.')[0] + '.mp4'

    print(output_path + video_name)

    if not os.path.exists(output_path):
        #
        os.makedirs(output_path)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')    # avi格式用

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4格式用

    video_writer = cv2.VideoWriter(
        output_path + video_name,
        fourcc,
        fps,
        size
    )  # (cfg.DEMO.DISPLAY_WIDTH, cfg.DEMO.DISPLAY_HEIGHT))

    for i, frame in enumerate(frame_data):
        # if i > 2:
        video_writer.write(frame)

    video_writer.release()


if __name__ == '__main__':
    #
    main()
