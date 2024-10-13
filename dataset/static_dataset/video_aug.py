import torch
import imageio
import numpy as np
import torch.nn.functional as F
from decord import VideoReader
import os
from PIL import Image


def aug_with_cam_motion(src_media, cx: float = 0.0, cy: float = 0.0, cz: float = 1.0,
          f: int = 8, h: int = 256, w: int = 256):
        """
        该函数使用一个源img/video（src_media）来模拟相机的平移和缩放动作, and generates a video
        Args:
            src_media (torch.Tensor or None): source image/video used to create video, the size can be [h w 3],[f h w 3],[3 h w],[1 3 h w]
            h (int): height of generated video
            w (int): width of generated video
            f (int): num frames of generated video if src_media is an image. If src_media is a video, f will be inherited from it.
            cx (float): x-translation ratio (-1~1), defined as total x-shift of the center between first-last frame / first-frame width
            cy (float): y-translation ratio (-1~1), defined as total y-shift of the center between first-last frame / first-frame height
            cz (float): zoom ratio (0.5~2), defined as scale ratio of last frame / that of the first frame. cz>1 for zoom-in, cz<1 for zoom-out
        Returns:
            generated video in f*c*3*h*w tensor format, and cam_boxes in f*4 tensor format
        """

        cam_boxes = torch.zeros(f, 4)  # 1st box is always [0,0,1,1]
        cam_boxes[:, 0] = torch.linspace(0, cx + (1 - 1 / cz) / 2, f)  # x1
        cam_boxes[:, 1] = torch.linspace(0, cy + (1 - 1 / cz) / 2, f)  # y1
        cam_boxes[:, 2] = torch.linspace(1, cx + (1 + 1 / cz) / 2, f)  # x2
        cam_boxes[:, 3] = torch.linspace(1, cy + (1 + 1 / cz) / 2, f)  # y2

        if src_media is None:
            return None, cam_boxes
        if isinstance(src_media, str) and src_media.endswith(('.png', '.jpg')):  # image
            import torchvision.transforms.functional as TF
            src_frames = torch.stack([TF.to_tensor(Image.open(src_media).convert("RGB"))] * f)
        elif isinstance(src_media, torch.Tensor):
            # [h w 3],[f h w 3],[3 h w],[1 3 h w] -> [f 3 h w]
            #print(src_media.shape)#(f, 1440, 2560, c)
            src_frames = src_media.unsqueeze(0) if src_media.dim() == 3 else src_media
            src_frames = src_frames.repeat(f, 1, 1, 1) if src_frames.shape[0] == 1 else src_frames
            src_frames = src_frames.permute(0, 3, 1, 2) if src_frames.shape[-1] == 3 else src_frames
            #print(src_frames.shape)
            assert src_frames.dim() == 4, 'src_media should be in shape of [f, c, h, w]'
            assert f == src_frames.shape[0], f'f={f} should be the same as src_media.shape[0]={src_frames.shape[0]}'
        else:
            raise TypeError("src_media should be torch.Tensor")

        min_x = torch.min(cam_boxes[:, 0::2])
        max_x = torch.max(cam_boxes[:, 0::2])
        min_y = torch.min(cam_boxes[:, 1::2])
        max_y = torch.max(cam_boxes[:, 1::2])

        normalized_boxes = torch.zeros_like(cam_boxes)
        normalized_boxes[:, 0::2] = (cam_boxes[:, 0::2] - min_x) / (max_x - min_x)
        normalized_boxes[:, 1::2] = (cam_boxes[:, 1::2] - min_y) / (max_y - min_y)

        _, _, src_h, src_w = src_frames.shape

        new_frames = torch.zeros(f, 3, h, w)
        for i, frame in enumerate(src_frames):
            # 定位截取框
            x1, y1, x2, y2 = normalized_boxes[i] * torch.tensor([src_w, src_h, src_w, src_h])
            crop = frame[:, int(y1):int(y2), int(x1):int(x2)].float()
            #print(crop.shape)
            new_frames[i] = F.interpolate(crop.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        #print(new_frames.shape)
        return new_frames, cam_boxes

def get_valid_frame_indices(video_length, n_sample_frames, sample_start_idx=0, sample_frame_stride=1):
        while True:
            sample_frame_indices = [sample_start_idx + i * sample_frame_stride for i in range(n_sample_frames)]
            if video_length > sample_frame_indices[-1]:
                break
            sample_frame_stride -= 1
        return sample_frame_indices


static_path = '/mnt/workspace/SVD_Xtend-main/static_dataset/static/'
save_dir  = '/mnt/workspace/SVD_Xtend-main/static_dataset_longer/right_frames/'

video_list = os.listdir(static_path)

for video_name in video_list:
    video_dir = static_path+video_name
    video_reader = VideoReader(video_dir)#[:1000]
    index_list=[]
    for i in range(len(video_reader)):
        #if i%5==0 and i<=1000:
        if i%2==0 and i<=400:
            index_list.append(i)
    index_list = index_list[:200]#为防止增强不明显，只取前100帧
    video = (video_reader.get_batch(index_list))#.asnumpy().astype(np.uint8)
    video = torch.tensor(video.asnumpy().astype(np.uint8))
    # augment video
    cam_x = 0.3#-0.3  #if np.random.uniform(0, 1) < 0.33 else np.random.uniform(-1, 1) (left,right) [-1，1]
    cam_y = 0#-0.25#-0.25  #if np.random.uniform(0, 1) < 0.33 else np.random.uniform(-1, 1) (up,down)  [-1,1]
    cam_z = 1#2**(-1)#2**(1/2) #if np.random.uniform(0, 1) < 0.33 else 2 ** np.random.uniform(-1, 1) (out,in) [0.5,2]
    h,w = 576,1024
    n_sample_frames = len(video)
    print(video_name, n_sample_frames)
    video, _ = aug_with_cam_motion(video[:], f=n_sample_frames, h=h, w=w, cx=cam_x, cy=cam_y, cz=cam_z)
    frame_list = []
    for i in range(video.shape[0]):
        frame_list.append(video[i].permute(1,2,0))
    for i in range(len(frame_list)):
        save_path = save_dir+video_name[:-4]+'/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        #print(frame_list[i].shape)
        frame = np.uint8(frame_list[i].numpy())
        imageio.imsave(save_path+str(i)+'.png',frame)