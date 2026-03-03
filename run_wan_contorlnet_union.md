```bash
python examples/wan/run_wan_controlnet_union.py \
    --depth_path "/vePFS-buaa/wangyuzhen/Dataset/train/0001/depth" \
    --normal_path "/vePFS-buaa/wangyuzhen/Dataset/train/0001/normal" \
    --mask_path "/vePFS-buaa/yinli/datasets/test_dataset/0001/mask" \
    --masked_frames_path "/vePFS-buaa/yinli/datasets/test_dataset/0001/rgb" \
    --image "/vePFS-buaa/wangyuzhen/Dataset/train/0001/firstframe.png" \
    --prompt "An ancient, extensive stone fortress with multiple towers and archways, nestled amongst towering, rugged cliffs under a hazy sky.In detail, the picture presents a sprawling, seemingly abandoned stone complex. On the right, a massive, multi-tiered structure dominates the landscape, featuring several smaller turrets and appearing to be constructed from large, grey blocks. This main building is partially obscured by a hazy atmosphere, giving it a weathered and ancient look, with faint patches of green moss suggesting age. To the left, a portion of another large stone structure is visible, integrated into or built against a colossal, craggy cliff face that occupies the entire left side of the frame. This cliff is extremely rough and textured, with visible fissures and small patches of vegetation clinging to its surfaces.Connecting these distant structures is a long, elevated stone bridge or aqueduct, supported by a series of arches. This bridge spans a wide gap, leading towards the center-left where another pointed, pyramid-like structure can be seen emerging from the haze in the far distance. Below the bridge and between the main buildings, there are lower grounds with scattered green bushes and areas of dry, yellowish grass.In the foreground, large, dark rocks with rough textures are prominent, framing the lower part of the scene and contributing to a sense of depth. The overall lighting is diffuse, characteristic of an overcast or foggy day, which contributes to the misty, ethereal atmosphere. There are no people or animals visible. The mood is one of quiet grandeur, mystery, and perhaps a touch of melancholy due to the apparent abandonment and weathered state of the structures." \
    --base_model "Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
    --controlnet_model "YuryyyLee/world-renderer-controlnet-union"  \
    --output_path "/vePFS-buaa/yinli/workspace/Diff-Factory/outputs/test_dataset/0001/output_union_depth_normal.mp4" \
    --output_dir "/vePFS-buaa/yinli/workspace/Diff-Factory/outputs/test_dataset/0001" \
    --video_width 512 \
    --video_height 384 \
    --num_frames 81
```


````python
# run_wan_controlnet_union.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append('..')
import argparse
import imageio.v2
import cv2
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import crop, resize

from transformers import UMT5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import export_to_video, load_video
from diffactory.models.transformers.transformer_controlnet_wan import WanTransformerControlnet3DModel
from diffactory.models.controlnets.controlnet_wan_union import WanControlnetUnion, WanControlNetUnionInput
from diffactory.pipelines.video_synthesis.pipeline_wan_controlnet_union import WanControlnetPipeline



def apply_gaussian_blur(image, ksize=5, sigmaX=1.0):
    image_np = np.array(image)
    if ksize % 2 == 0:
        ksize += 1
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return Image.fromarray(blurred_image)

class TilePreprocessor:
    def __call__(self, image, target_h, target_w, ksize=5, downscale_coef=4):
        img = image.resize((target_w // downscale_coef, target_h // downscale_coef))
        img = apply_gaussian_blur(img, ksize=ksize, sigmaX=ksize // 2)
        return img.resize((target_w, target_h))

def resize_for_crop(image, crop_h, crop_w):
    img_h, img_w = image.shape[-2:]
    if img_h >= crop_h and img_w >= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    elif img_h <= crop_h and img_w <= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    else:
        coef = crop_h / img_h if crop_h > img_h else crop_w / img_w 
    out_h, out_w = int(img_h * coef), int(img_w * coef)
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image

def prepare_image(input_image, size, do_resize=True, do_crop=True):
    image_tensor = torch.from_numpy(np.array(input_image)).permute(2, 0, 1) / 127.5 - 1  # CHW
    if do_resize:
        image_tensor = resize_for_crop(image_tensor, crop_h=size[0], crop_w=size[1])
    if do_crop:
        image_tensor = transforms.functional.center_crop(image_tensor, size)
    return image_tensor.unsqueeze(0) # 1CHW

def prepare_frames(input_images, video_size, do_resize=True, do_crop=True):
    input_images = np.stack([np.array(x) for x in input_images]) #FHWC
    images_tensor = torch.from_numpy(input_images).permute(0, 3, 1, 2) / 127.5 - 1  # FCHW
    if do_resize:
        images_tensor = [resize_for_crop(x, crop_h=video_size[0], crop_w=video_size[1]) for x in images_tensor]
    if do_crop:
        images_tensor = [transforms.functional.center_crop(x, video_size) for x in images_tensor]
    if isinstance(images_tensor, list):
        images_tensor = torch.stack(images_tensor)
    return images_tensor.unsqueeze(0) # BFCHW


def prepare_controlnet_frames(controlnet_type, controlnet_frames, height, width):
    prepared_frames = None
    if controlnet_type == "gt":
        prepared_frames = prepare_frames(controlnet_frames, (height, width))
    elif controlnet_type == "depth":
        # 堆叠成 [F, C, H, W]
        prepared_frames = torch.stack(controlnet_frames, dim=0)
        # 增加 batch 维，得到 [B, F, C, H, W]
        prepared_frames = prepared_frames.unsqueeze(0)  # B=1
        print("depth: ", prepared_frames.shape)
    elif controlnet_type == "normal":
        # 堆叠成 [F, C, H, W]
        prepared_frames = torch.stack(controlnet_frames, dim=0)
        # 增加 batch 维，得到 [B, F, C, H, W]
        prepared_frames = prepared_frames.unsqueeze(0)  # B=1
        print("normal: ", prepared_frames.shape)
    
    return prepared_frames  # BFCHW


import imageio
import numpy as np
import torch
import cv2

def process_depth(img_paths, target_w, target_h):
    """
    多帧 depth 处理：
        1. 读取为 float
        2. 全局 5%~95% 分位数归一化
        3. 裁剪保持比例
        4. resize
        5. 扩展到 3 通道 CHW
    """

    # ---------- 1. 读取所有深度 ----------
    depths = []
    for p in img_paths:
        depthmap = imageio.v2.imread(p).astype(np.float32) / 65535.0
        if depthmap.ndim == 3:           # RGB 深度图 → 取第一通道
            depthmap = depthmap[..., 0]
        depthmap = depthmap.astype(np.float32)
        # ---------- 裁剪保持比例 ----------
        H, W = depthmap.shape
        target_ratio = target_w / target_h
        current_ratio = W / H

        if current_ratio > target_ratio:       # 宽 → 裁左右
            new_W = int(H * target_ratio)
            start = (W - new_W) // 2
            depthmap = depthmap[:, start:start + new_W]
        else:                                  # 高 → 裁上下
            new_H = int(W / target_ratio)
            start = (H - new_H) // 2
            depthmap = depthmap[start:start + new_H, :]

        # ---------- resize ----------
        depthmap = cv2.resize(depthmap, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Mask
        near_mask = depthmap < 0.0015   # 1. too close
        far_mask = depthmap > (65500.0 / 65535.0) # 2. filter sky
        # omniworld处理
        # near, far = 1., 1000.
        # depthmap = depthmap / (far - depthmap * (far - near)) / 0.004

        valid = ~(near_mask | far_mask)
        depthmap[~valid] = np.nan  # 非法标1

        depths.append(depthmap)

    stacked = np.stack(depths, 0)   # [N,H,W]

    global_min = np.nanmin(stacked)
    global_max = np.nanmax(stacked)

    print(f"[Depth Normalize] min={global_min}, max={global_max}")

    outputs = []

    # ---------- 3. 处理每一帧 ----------
    for depthmap in depths:

        # ---------- 归一化到 0~1 ----------
        depthmap = (depthmap - global_min) / (global_max - global_min + 1e-8)
        # depthmap = 1 - depthmap
        # 去除无效像素
        depthmap = np.nan_to_num(depthmap, nan=1)
        depthmap = depthmap * 2 - 1

        # ---------- 转 3 通道 (C,H,W) ----------
        t = torch.from_numpy(depthmap).float()
        t = t.unsqueeze(0).repeat(3, 1, 1)     # 3×H×W
        outputs.append(t)
    
    return outputs   # list[tensor(3,H,W)]

def process_normal(img_paths, target_w, target_h, normal_format="opencv"):
    """
    多帧 normal 处理：
        1. 读取 normal（RGB / uint8 / float）
        2. 裁剪保持比例
        3. resize
        4. [0,255] → [-1,1]
        5. 坐标系修正（OpenCV → OpenGL 可选）
        6. 输出 list[tensor(3,H,W)]
    
    Args:
        img_paths: list[str]
        normal_format: "opencv" or "opengl"
            - opencv: (x, y, z) = (right, down, forward)
            - opengl: (x, y, z) = (right, up, backward)
    """

    outputs = []

    for p in img_paths:
        normal = imageio.v2.imread(p)

        # ---------- 1. 读入 ----------
        if normal.ndim == 2:
            raise ValueError(f"Normal map must be RGB: {p}")
        normal = normal[..., :3].astype(np.float32)

        # ---------- 2. 裁剪保持比例 ----------
        H, W, _ = normal.shape
        target_ratio = target_w / target_h
        current_ratio = W / H

        if current_ratio > target_ratio:       # 裁左右
            new_W = int(H * target_ratio)
            start = (W - new_W) // 2
            normal = normal[:, start:start + new_W]
        else:                                  # 裁上下
            new_H = int(W / target_ratio)
            start = (H - new_H) // 2
            normal = normal[start:start + new_H, :]

        # ---------- 3. resize ----------
        normal = cv2.resize(normal, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # ---------- 4. [0,255] → [-1,1] ----------
        if normal.max() > 1.5:   # uint8 / 16bit
            normal = normal / 127.5 - 1.0

        # ---------- 5. 坐标系修正 ----------
        if normal_format == "opencv":
            # OpenCV → OpenGL
            # x: right (same)
            # y: down  -> up
            # z: forward -> backward
            normal[..., 1] *= -1
            normal[..., 2] *= -1

        # ---------- 6. 转 torch (C,H,W) ----------
        t = torch.from_numpy(normal).permute(2, 0, 1).float()
        outputs.append(t)

    return outputs   # list[tensor(3,H,W)]

def process_image(img_path, target_w, target_h):
    img = read_image(img_path)
    target_aspect = target_w / target_h
    _, h, w = img.shape
    src_aspect = w / h

    # 裁剪保持比例
    if src_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = crop(img, top=0, left=left, height=h, width=new_w)
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        img = crop(img, top=top, left=0, height=new_h, width=w)

    # 下采样
    img = resize(img, [target_h, target_w])

    img = img / 255.0

    return img.to(dtype=torch.float32)

def process_mask(img_path, target_w, target_h):
    mask = imageio.v2.imread(img_path).astype(np.float32)

    # 1. Center crop
    Ht, Wt = target_h, target_w
    H, W = mask.shape
    target_ratio = Wt / Ht
    current_ratio = W / H

    if abs(target_ratio - current_ratio) > 1e-3:
        if current_ratio > target_ratio:  # 图太宽 → 裁左右
            new_W = int(H * target_ratio)
            start = (W - new_W) // 2
            mask = mask[:, start:start + new_W]
        else:  # 图太高 → 裁上下
            new_H = int(W / target_ratio)
            start = (H - new_H) // 2
            mask = mask[start:start + new_H, :]

    # 2. Resize
    mask = cv2.resize(mask, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

    # 3. Convert to binary 0/1
    mask = (mask > 0).astype(np.uint8)
    mask = torch.from_numpy(mask.copy()).float()
    mask = mask.unsqueeze(0)   
    return mask.to(dtype=torch.float32)

def prepare_pipeline(
    base_model,
    controlnet_model,
    device,
    dtype,
):
    controlnet: WanControlnetUnion = WanControlnetUnion.from_pretrained(
        controlnet_model,
    )
    transformer = WanTransformerControlnet3DModel.from_pretrained(
        base_model,
        subfolder="transformer",
    )
    pipe = WanControlnetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        transformer=transformer,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    # pipe.enable_model_cpu_offload()
    pipe.to(device=device, dtype=dtype)

    return pipe


@torch.no_grad()
def run_pipeline(
    pipe,
    prompt: str,
    depth_path: str,
    normal_path: str,
    mask_path: str,
    masked_frames_path: str,
    image_path: str,
    controlnet_type: str,
    controlnet_weight: float = 0.8,
    controlnet_guidance_start: float = 0.0,
    controlnet_guidance_end: float = 0.8,
    output_path: str = "./output.mp4",
    output_dir: str = "./",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    video_height: int = 480,
    video_width: int = 832,
    num_frames: int = 81,
    negative_prompt: str = "bad quality, worst quality",
    seed: int = 42,
    out_fps: int = 16,
    lora_path: str = None,
    lora_rank: int = 128,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - depth_path (str): The video for controlnet processing.
    - base_model (str): The path of the pre-trained model to be used.
    - controlnet_model (str): The path of the pre-trained conrolnet model to be used.
    - controlnet_type (str): Type of controlnet model (e.g. canny, hed).
    - controlnet_weight (float): Strenght of controlnet
    - controlnet_guidance_start (float): The stage when the controlnet starts to be applied
    - controlnet_guidance_end (float): The stage when the controlnet end to be applied
    - controlnet_stride (int): Stride for controlnet blocks
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - teacache_treshold (float): TeaCache value. Best from [0.3, 0.5, 0.7, 0.9].
    - video_height (int): Output video height.
    - video_width (int): Output video width.
    - num_frames (int): Output frames count.
    - seed (int): The seed for reproducibility.
    - out_fps (int): FPS of output video.
    """
    # image
    image = None
    image_tensor = None
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        image_tensor = prepare_image(image, (video_height, video_width)) # 1,3,H,W [-1,1]
    
    union_input = WanControlNetUnionInput()
    # depth
    video = []
    controlnet_depth_frames = None
    if depth_path is not None and os.path.isdir(depth_path):
        video = sorted([
            os.path.join(depth_path, f)
            for f in os.listdir(depth_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if num_frames:
            video = video[:num_frames]
        video = process_depth(video, video_width, video_height)
        controlnet_depth_frames = prepare_controlnet_frames("depth", video, video_height, video_width) #BFCHW
        union_input.depth = controlnet_depth_frames
    else:
        print("[INFO] No depth path provided.")
    
    # normal
    normal_video = []
    controlnet_normal_frames = None 
    if normal_path is not None and os.path.isdir(normal_path):
        normal_video = sorted([
            os.path.join(normal_path, f)
            for f in os.listdir(normal_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if num_frames:
            normal_video = normal_video[:num_frames]
        
        normal_video = process_normal(normal_video, video_width, video_height)
        controlnet_normal_frames = prepare_controlnet_frames("normal", normal_video, video_height, video_width) #BFCHW
        union_input.depth = controlnet_normal_frames
    else:
        print("[INFO] No normal path provided.")

    # mask
    masks = []
    if mask_path is not None and os.path.isdir(mask_path):
        mask_paths = sorted([
            os.path.join(mask_path, f)
            for f in os.listdir(mask_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if num_frames:
            mask_paths = mask_paths[:num_frames]
        for fp in mask_paths:
            mask = process_mask(fp, video_width, video_height)
            masks.append(mask)
    else:
        print("[INFO] No mask path provided. Using black masks.")
        zero_mask = torch.zeros(1, video_height, video_width, dtype=torch.float32)
        masks = [zero_mask for _ in range(num_frames)]

    # 如果给了 image_path：第一帧 mask = 全 1
    if image_path is not None:
        print("[INFO] image_path provided: set first mask frame to all-ones.")
        masks[0] = torch.ones(1, video_height, video_width, dtype=torch.float32)


    # Convert list -> tensor (B=1,F,1,H,W)
    masks = torch.stack(masks, dim=0).unsqueeze(0)  # 1,F,1,H,W

    # masked frames
    masked_frames = []
    if masked_frames_path is not None and os.path.isdir(masked_frames_path):
        masked_files = sorted([
            os.path.join(masked_frames_path, f)
            for f in os.listdir(masked_frames_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if num_frames:
            masked_files = masked_files[:num_frames]

        # Load masked RGB frames
        for fp in masked_files:
            masked_img = process_image(fp, video_width, video_height)
            masked_frames.append(masked_img)
    else:
        # If no masked_frames_path, simply use black frames
        print("[INFO] No masked frames path provided. Using black masked frames.")
        zero_rgb = torch.zeros(3, video_height, video_width)
        masked_frames = [zero_rgb.clone() for _ in range(num_frames)]

    # 如果给了 image_path：第一帧 masked_rgb = image
    if image_path is not None:
        print("[INFO] image_path provided: set first masked frame to image.")
        # image_tensor: [1,3,H,W] in [-1,1]
        masked_frames[0] = (image_tensor.squeeze(0) + 1.0) / 2.0  # 转到 [0,1]

    masked_frames = torch.stack(masked_frames, dim=0).unsqueeze(0)  # 1,F,3,H,W
    
    vid_list = []
    # depth
    if controlnet_depth_frames is not None:
        depth_vid = rearrange(
            controlnet_depth_frames, "b f c h w -> f h (b w) c"
        ).float()  # [T,H,BW,3], [-1,1]
        depth_vid = depth_vid * 0.5 + 0.5
        vid_list.append(depth_vid)

    # normal: [-1,1] -> [0,1]
    if controlnet_normal_frames is not None:
        normal_vid = rearrange(
            controlnet_normal_frames, "b f c h w -> f h (b w) c"
        ).float()
        normal_vid = normal_vid * 0.5 + 0.5
        vid_list.append(normal_vid)

    # mask
    mask_vid = rearrange(
        masks.repeat(1, 1, 3, 1, 1), "b f c h w -> f h (b w) c"
    ).float()  # [T,H,BW,1], [0,1]
    vid_list.append(mask_vid)

    # masked
    masked_vid = rearrange(
        masked_frames, "b f c h w -> f h (b w) c"
    ).float()  # [T,H,BW,1], [0,1]
    vid_list.append(masked_vid)

    concat_vid = torch.cat(
        vid_list,
        dim=2  # W 方向拼接
    )  # [T, H, 3*BW, 3]

    concat_vid = (
        concat_vid
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )

    concat_pils = [
        Image.fromarray(concat_vid[t])
        for t in range(concat_vid.shape[0])
    ]

    os.makedirs(output_dir, exist_ok=True)

    save_path = f"{output_dir}/concat.mp4"
    
    export_to_video(concat_pils, save_path, fps=16)
    
    # Generate the video frames based on the prompt.
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=video_height,
        width=video_width,
        num_frames=num_frames,
        controlnet_cond=union_input,
        masked_video_frames=masked_frames * 2 - 1, # masked_video_key
        mask_frames=masks.repeat(1, 1, 3, 1, 1), # mask
        image=image_tensor,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type="pil",
    
        # controlnet_frames=controlnet_frames,
        controlnet_guidance_start=controlnet_guidance_start,
        controlnet_guidance_end=controlnet_guidance_end,
        controlnet_weight=controlnet_weight,

    ).frames[0]
    export_to_video(output, output_path, fps=out_fps)

    # 1. 将生成的视频帧转为 Tensor [T, H, W, 3]，范围 [0, 1]
    gen_vid = np.stack([np.array(f) for f in output])
    gen_vid_tensor = torch.from_numpy(gen_vid).float() / 255.0

    # 2. 将之前用于可视化的 concat_vid (已经拼接了 depth/normal/mask 等) 
    # 与生成的 gen_vid_tensor 在宽度方向 (dim=2) 拼接
    # 注意：之前可视化用的 concat_vid 是 numpy uint8 格式，
    # 建议直接使用之前构造 vid_list 的逻辑重新拼接，包含生成的视频
    
    vid_list.append(gen_vid_tensor) # 将生成的视频加入列表
    
    final_concat_vid = torch.cat(
        vid_list,
        dim=2  # W 方向拼接
    )  # [T, H, (N+1)*BW, 3]

    # 3. 转换回 numpy 以便保存
    final_concat_vid = (
        final_concat_vid
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )

    final_pils = [
        Image.fromarray(final_concat_vid[t])
        for t in range(final_concat_vid.shape[0])
    ]

    # 4. 保存最终的对比视频
    final_save_path = f"{output_dir}/concat_with_output.mp4"
    export_to_video(final_pils, final_save_path, fps=out_fps)
    # print(f"对比视频已保存至: {final_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan2.1")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--depth_path",
        type=str,
        default=None,
        help="The depth path of the video for controlnet processing.",
    )
    parser.add_argument(
        "--normal_path",
        type=str,
        default=None,
        help="The normal path of the video for controlnet processing.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="The path of the mask for controlnet processing.",
    )
    parser.add_argument(
        "--masked_frames_path",
        type=str,
        default=None,
        help="The path of the masked video for controlnet processing.",
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="The path of the image as the first frame.",
    )
    parser.add_argument(
        "--base_model", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model", type=str, default="TheDenk/wan2.1-t2v-1.3b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--controlnet_type", type=str, default='depth', help="Type of controlnet model (e.g. canny, hed)")
    parser.add_argument("--controlnet_weight", type=float, default=0.8, help="Strenght of controlnet")
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.8, help="The stage when the controlnet end to be applied")
    parser.add_argument("--controlnet_stride", type=int, default=3, help="Strenght of controlnet")
    
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./", help="The path where the visulized video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--video_height", type=int, default=704, help="Output video height")
    parser.add_argument("--video_width", type=int, default=1280, help="Output video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Output frames count")
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worst quality", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--out_fps", type=int, default=16, help="FPS of output video")
    parser.add_argument("--teacache_treshold", type=float, default=0.0, help="TeaCache value. Best from [0.3, 0.5, 0.7, 0.9]")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()


    pipe = prepare_pipeline(
        base_model=args.base_model,
        controlnet_model=args.controlnet_model,
        device=args.device,
        dtype=torch.float32,
    )

    run_pipeline(
        pipe,
        prompt=args.prompt,
        depth_path=args.depth_path,
        normal_path=args.normal_path,
        mask_path=args.mask_path,
        masked_frames_path=args.masked_frames_path,
        image_path=args.image_path,
        controlnet_type=args.controlnet_type,
        controlnet_weight=args.controlnet_weight,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        output_path=args.output_path,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        video_height=args.video_height,
        video_width=args.video_width,
        num_frames=args.num_frames,
        seed=args.seed,
        out_fps=args.out_fps,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
    )

````

