import torch
from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
from diffusers.loaders.lora import LoraLoaderMixin
from diffusers.loaders.peft import PeftAdapterMixin  

pipe = StableVideoDiffusionPipeline.from_pretrained(
    './pretrained_svd/svd_ckpt/',
    #unet=unet,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float16, variant="fp16", local_files_only=False,
)

# 将类cls 的所有方法绑定到实例 instance
def bind_all_methods(instance, cls):
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        # 确保属性/方法是 不是特殊的（如__init__）
        if not attr_name.startswith('__'):
            if callable(attr):   # 方法
                setattr(instance, attr_name, attr.__get__(instance, instance.__class__))
            else:   # 普通属性
                setattr(instance, attr_name, attr)

bind_all_methods(pipe, LoraLoaderMixin) # 将PeftAdapterMixin的方法绑定到UNetSpatioTemporalConditionModel 对象
bind_all_methods(pipe.unet,PeftAdapterMixin)

lora_path = "ckpts/left/pytorch_lora_weights.safetensors"
#pipe.unload_lora_weights()#删除前面的lora
#pipe._lora_scale = 0.3#设置lora_scale权重
pipe.text_encoder = pipe.image_encoder  # 为了让load_lora_into_text_encoder成功运行
pipe.load_lora_weights(lora_path)
#pipe.fuse_lora(lora_scale=1)
pipe.text_encoder = None
pipe.to("cuda:0")

from PIL import Image
image = Image.open("test_images/base/image (2).png")
(h,w) = image.size
image = image.resize((1024, 576))
generator = torch.manual_seed(-1)
with torch.inference_mode():
    frames = pipe(image,
                num_frames=14,
                width=1024, #width=int(w//64*64), #1024,
                height=576, #height=int(h//64*64), #576,
                decode_chunk_size=8, generator=generator, motion_bucket_id=127, fps=8, num_inference_steps=30).frames[0]
for i in range(len(frames)):
    frames[i]=frames[i].resize((h,w))
export_to_video(frames, "generated.mp4", fps=4)