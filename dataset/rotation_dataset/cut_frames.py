import os
#import torch
import imageio
import numpy as np
#import torch.nn.functional as F
from decord import VideoReader

video_path = "/mnt/workspace/SVD_Xtend-main/static_dataset/static//"
save_dir = '/mnt/workspace/SVD_Xtend-main/static_dataset_longer/static_frames/'
video_list = []
for video_name in os.listdir(video_path):
    video_list.append(video_name)

for video_dir in video_list:
    print(video_dir)
    if video_dir[:-4] in os.listdir(save_dir):
        continue
    video_reader = VideoReader(video_path+video_dir)
    index_list=[]
    for i in range(len(video_reader)):
        if i%2==0:
            index_list.append(i)
    index_list.reverse()
    video = (video_reader.get_batch(index_list[:200])).asnumpy().astype(np.uint8)
    for i in range(video.shape[0]):
        save_path = save_dir+video_dir[:-4]+'/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        imageio.imsave(save_path+str(i).zfill(4)+'.png',video[i])