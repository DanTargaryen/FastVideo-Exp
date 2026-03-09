## inference_long

```bash
# script
python examples/wan/run_wan_controlnet_union_long.py \
	--cam_k "/vePFS-buaa/wangyuzhen/Dataset/train/0039/camera/camera_K.txt"\
	--cam_rt_dir "/vePFS-buaa/wangyuzhen/Dataset/train/0039/camera" \
	--total_frames 401 \
	--num_keyframes 4 \
	--prompt "A slow, sweeping pan around a cozy log cabin nestled in a vast, snowy landscape under a vibrant purple and pink twilight sky. A steady stream of soft, translucent smoke curls upward from the chimney. The thick, white snow on the roof has a soft, pillowy texture. Small evergreen trees are visible in the far distance. Lo-fi aesthetic, nostalgic and peaceful." \
    --depth_path "/vePFS-buaa/wangyuzhen/Dataset/train/0039/depth" \
    --normal_path "/vePFS-buaa/wangyuzhen/Dataset/train/0039/normal" \
    --image_path "/vePFS-buaa/wangyuzhen/Dataset/train/0039/firstframe.png" \
    --base_model "Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
    --controlnet_model "YuryyyLee/world-renderer-controlnet-union"  \
    --output_dir "/vePFS-buaa/yinli/workspace/Diff-Factory/outputs/test_dataset_long/0039_49" \
    --video_width 512 \
    --video_height 384 \
    --num_frames 49
```



```python
# run_wan_controlnet_long.py
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
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import export_to_video
from diffactory.models.transformers.transformer_controlnet_wan import WanTransformerControlnet3DModel
from diffactory.models.controlnets.controlnet_wan_union import WanControlnetUnion, WanControlNetUnionInput
from diffactory.pipelines.video_synthesis.pipeline_wan_controlnet_union import WanControlnetPipeline

# ==========================================
# --- 1. 3D POINT CLOUD WARPING PIPELINE ---
# ==========================================
class PointCloudPipeline:
    @staticmethod
    def load_camera(path):
        return np.loadtxt(path) if path.endswith('.txt') else np.load(path)

    def back_project(self, rgb, depth, K, Rt):
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        bg_val = np.max(depth)
        mask = (depth > 0) & (depth < bg_val * 0.99)
        
        z_val = depth[mask]
        u_val = u[mask]
        v_val = v[mask]
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        x_cam = (u_val - cx) * z_val / fx
        y_cam = -((v_val - cy) * z_val / fy)
        z_cam = -z_val
        p_cam = np.stack((x_cam, y_cam, z_cam), axis=-1)

        R_raw = Rt[:3, :3]
        T_raw = Rt[:3, 3]
        
        R_inv = np.linalg.inv(R_raw)
        T_neg = -T_raw
        
        points_world = (p_cam + T_neg) @ R_inv.T
        colors = rgb[mask][:, :3]
        
        return points_world, colors

    def project_to_view(self, points, colors, K, Rt, out_h, out_w, target_depth=None):
        R = Rt[:3, :3]
        T = Rt[:3, 3]
        cam_points = (points @ R.T) + T
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        zs = -cam_points[:, 2]
        valid = zs > 0.05
        zs_safe = np.where(valid, zs, 1e-6)
        
        u_proj = (fx * (cam_points[:, 0] / zs_safe) + cx).astype(np.int32)
        v_proj = (fy * (-cam_points[:, 1] / zs_safe) + cy).astype(np.int32)
        
        in_bounds = valid & (u_proj >= 0) & (u_proj < out_w) & (v_proj >= 0) & (v_proj < out_h)
        
        u_v = u_proj[in_bounds]
        v_v = v_proj[in_bounds]
        zs_v = zs[in_bounds]
        colors_v = colors[in_bounds]

        if len(u_v) == 0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8), np.zeros((out_h, out_w), dtype=np.uint8)

        if target_depth is not None:
            ref_depths = target_depth[v_v, u_v]
            margin = ref_depths * 0.05 
            not_occluded = (ref_depths == 0) | (zs_v <= ref_depths + margin)
            
            u_v = u_v[not_occluded]
            v_v = v_v[not_occluded]
            zs_v = zs_v[not_occluded]
            colors_v = colors_v[not_occluded]
            
            if len(u_v) == 0:
                return np.zeros((out_h, out_w, 3), dtype=np.uint8), np.zeros((out_h, out_w), dtype=np.uint8)

        sort_idx = np.argsort(zs_v)[::-1] 
        
        img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        mask = np.zeros((out_h, out_w), dtype=np.uint8)
        
        img[v_v[sort_idx], u_v[sort_idx]] = colors_v[sort_idx]
        mask[v_v[sort_idx], u_v[sort_idx]] = 255
        
        return img, mask


# ==========================================
# --- 2. ALIGNMENT & PREPROCESSING ---
# ==========================================
def get_crop_params(src_w, src_h, target_w, target_h):
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:  
        crop_w = int(src_h * target_ratio)
        crop_h = src_h
        left = (src_w - crop_w) // 2
        top = 0
    else: 
        crop_h = int(src_w / target_ratio)
        crop_w = src_w
        top = (src_h - crop_h) // 2
        left = 0
    return top, left, crop_h, crop_w

def prepare_controlnet_frames(controlnet_frames, height, width):
    return torch.stack(controlnet_frames, dim=0).unsqueeze(0)

def process_depth(depth_list, target_w, target_h, crop_params):
    top, left, ch, cw = crop_params
    depths = []
    for item in depth_list:
        depthmap = imageio.v2.imread(item).astype(np.float32) / 65535.0
        if depthmap.ndim == 3: depthmap = depthmap[..., 0]
        depths.append(depthmap)

    stacked = np.stack(depths, 0)
    global_min, global_max = np.nanmin(stacked), np.nanmax(stacked)

    outputs = []
    for depthmap in depths:
        depthmap = (depthmap - global_min) / (global_max - global_min + 1e-8)
        depthmap = np.nan_to_num(depthmap, nan=1.0)
        depthmap = depthmap * 2 - 1
        t = torch.from_numpy(depthmap).float().unsqueeze(0)
        t = F.resized_crop(t, top, left, ch, cw, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)
        outputs.append(t.repeat(3, 1, 1))
    return outputs

def process_normal(normal_list, target_w, target_h, crop_params, normal_format="opencv"):
    top, left, ch, cw = crop_params
    outputs = []
    for item in normal_list:
        normal = imageio.v2.imread(item)
        if normal.ndim == 2: raise ValueError("Normal map must be RGB")
        normal = normal[..., :3].astype(np.float32)
        if normal.max() > 1.5: normal = normal / 127.5 - 1.0
        if normal_format == "opencv":
            normal[..., 1] *= -1
            normal[..., 2] *= -1

        t = torch.from_numpy(normal).permute(2, 0, 1).float()
        t = F.resized_crop(t, top, left, ch, cw, (target_h, target_w), interpolation=InterpolationMode.BILINEAR, antialias=True)
        outputs.append(t)
    return outputs

def prepare_pipeline(base_model, controlnet_model, device, dtype):
    controlnet: WanControlnetUnion = WanControlnetUnion.from_pretrained(controlnet_model)
    transformer = WanTransformerControlnet3DModel.from_pretrained(base_model, subfolder="transformer")
    pipe = WanControlnetPipeline.from_pretrained(base_model, controlnet=controlnet, transformer=transformer)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.to(device=device, dtype=dtype)
    return pipe

# 物理深度图读取工具 (用于 3D Warp)
def read_physical_depth(path):
    depth_raw = imageio.v2.imread(path)
    if depth_raw.ndim == 3: depth_raw = depth_raw[..., 0]
    return depth_raw.astype(np.float32) / 1000.0 if depth_raw.dtype == np.uint16 else depth_raw.astype(np.float32)

def adjust_intrinsics(K, crop_params, target_w, target_h):
    """根据 Crop 和 Resize 操作，等价修正相机的内参矩阵 K"""
    top, left, crop_h, crop_w = crop_params
    scale_x = target_w / crop_w
    scale_y = target_h / crop_h
    
    K_new = K.copy()
    # 1. 处理裁剪引起的光心偏移
    K_new[0, 2] -= left
    K_new[1, 2] -= top
    
    # 2. 处理缩放引起的焦距和光心等比缩放
    K_new[0, 0] *= scale_x
    K_new[1, 1] *= scale_y
    K_new[0, 2] *= scale_x
    K_new[1, 2] *= scale_y
    
    return K_new

def get_aligned_physical_depth(depth_path, crop_params, target_w, target_h):
    """读取物理深度，并将其裁剪/缩放到与视频完全一致的尺寸 (使用最近邻插值防边缘模糊)"""
    top, left, ch, cw = crop_params
    depth_raw = read_physical_depth(depth_path) # 返回 float32 的米
    
    depth_t = torch.from_numpy(depth_raw).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    depth_aligned = F.resized_crop(depth_t, top, left, ch, cw, (target_h, target_w), interpolation=InterpolationMode.NEAREST)
    
    return depth_aligned.squeeze().numpy() # 返回 [target_h, target_w] 的 numpy 数组

# ==========================================
# --- 3. DYNAMIC WARPING & MERGING ---
# ==========================================
def warp_and_merge_keyframes(
    pipeline_3d: PointCloudPipeline,
    keyframes_rgb, 
    keyframes_indices, 
    target_global_indices,
    all_depth_files,
    camera_K, 
    camera_RT_dir,
    crop_params,  
    target_w,     
    target_h,     
    chunk_idx, 
    output_dir
):
    merged_rgbs = []
    merged_masks = []
    
    h, w, _ = keyframes_rgb[0].shape
    
    chunk_save_dir = os.path.join(output_dir, f"chunk_{chunk_idx}_warped_inputs")
    
    dir_rgb = os.path.join(chunk_save_dir, "rgb")
    dir_mask = os.path.join(chunk_save_dir, "mask")
    dir_merged_rgb = os.path.join(chunk_save_dir, "merged_rgb")
    dir_merged_mask = os.path.join(chunk_save_dir, "merged_mask")
    
    os.makedirs(dir_rgb, exist_ok=True)
    os.makedirs(dir_mask, exist_ok=True)
    os.makedirs(dir_merged_rgb, exist_ok=True)
    os.makedirs(dir_merged_mask, exist_ok=True)

    for local_t, target_global_idx in enumerate(target_global_indices):
        
        # ===================================================================
        # 【核心修正】：如果是当前 Chunk 的第一帧，直接使用无损的关键帧，跳过 Warp
        # ===================================================================
        if local_t == 0:
            # keyframes_rgb 的最后一帧，完美对应了当前 Chunk 的起始帧
            final_rgb = keyframes_rgb[-1].copy()
            final_mask = np.ones((h, w), dtype=np.float32) * 255.0
            
            # 直接保存无损的合并结果
            Image.fromarray(final_rgb.astype(np.uint8)).save(os.path.join(dir_merged_rgb, f"frame_{local_t:04d}.png"))
            Image.fromarray(final_mask.astype(np.uint8)).save(os.path.join(dir_merged_mask, f"frame_{local_t:04d}.png"))
            
            merged_rgbs.append(final_rgb)
            merged_masks.append(final_mask)
            
            print(f"  -> Frame {local_t:04d} (Global {target_global_idx:04d}): 注入完整参考帧")
            continue # 直接进入下一帧循环，跳过繁重的 3D 计算
        # ===================================================================

        warped_rgbs_t = []
        warped_masks_t = []
        
        target_rt_path = os.path.join(camera_RT_dir, f"camera_RT_{target_global_idx:04d}.txt")
        if not os.path.exists(target_rt_path): 
            target_rt_path = os.path.join(camera_RT_dir, f"{target_global_idx:04d}.txt")
        Rt_tgt = pipeline_3d.load_camera(target_rt_path)
        
        tgt_depth = get_aligned_physical_depth(all_depth_files[target_global_idx], crop_params, target_w, target_h)

        for i, (k_rgb, k_idx) in enumerate(zip(keyframes_rgb, keyframes_indices)):
            src_rt_path = os.path.join(camera_RT_dir, f"camera_RT_{k_idx:04d}.txt")
            if not os.path.exists(src_rt_path): 
                src_rt_path = os.path.join(camera_RT_dir, f"{k_idx:04d}.txt")
            Rt_src = pipeline_3d.load_camera(src_rt_path)
            
            src_depth = get_aligned_physical_depth(all_depth_files[k_idx], crop_params, target_w, target_h)

            points, colors = pipeline_3d.back_project(k_rgb, src_depth, camera_K, Rt_src)
            w_rgb, w_mask = pipeline_3d.project_to_view(points, colors, camera_K, Rt_tgt, h, w, target_depth=tgt_depth)

            warped_rgbs_t.append(w_rgb)
            warped_masks_t.append(w_mask)
            
            Image.fromarray(w_rgb.astype(np.uint8)).save(os.path.join(dir_rgb, f"frame_{local_t:04d}_src_kf_{i}.png"))
            Image.fromarray(w_mask.astype(np.uint8)).save(os.path.join(dir_mask, f"frame_{local_t:04d}_src_kf_{i}.png"))
            
        final_rgb = np.zeros_like(warped_rgbs_t[0])
        final_mask = np.zeros_like(warped_masks_t[0])
        
        for w_rgb, w_mask in zip(warped_rgbs_t, warped_masks_t):
            valid_area = w_mask > 127.5
            final_rgb[valid_area] = w_rgb[valid_area]
            final_mask[valid_area] = 255.0

        Image.fromarray(final_rgb.astype(np.uint8)).save(os.path.join(dir_merged_rgb, f"frame_{local_t:04d}.png"))
        Image.fromarray(final_mask.astype(np.uint8)).save(os.path.join(dir_merged_mask, f"frame_{local_t:04d}.png"))

        merged_rgbs.append(final_rgb)
        merged_masks.append(final_mask)
        
    return merged_rgbs, merged_masks


# ==========================================
# --- 4. SINGLE CHUNK INFERENCE ---
# ==========================================
@torch.no_grad()
def run_pipeline_chunk(pipe, prompt, chunk_depth_files, chunk_normal_files, chunk_mask_arrays, chunk_masked_rgb_arrays, crop_params, video_height, video_width, num_frames, guidance_scale, num_inference_steps, negative_prompt, seed, controlnet_weight, controlnet_guidance_start, controlnet_guidance_end, device):
    top, left, ch, cw = crop_params
    union_input = WanControlNetUnionInput()

    if chunk_depth_files:
        v_processed = process_depth(chunk_depth_files, video_width, video_height, crop_params)
        union_input.depth = prepare_controlnet_frames(v_processed, video_height, video_width).to(device)
    
    if chunk_normal_files:
        n_processed = process_normal(chunk_normal_files, video_width, video_height, crop_params)
        union_input.normal = prepare_controlnet_frames(n_processed, video_height, video_width).to(device)

    masks_list = []
    for mask_arr in chunk_mask_arrays:
        if mask_arr.ndim == 3: mask_arr = mask_arr[..., 0]
        mask_t = torch.from_numpy(mask_arr).float().unsqueeze(0) / 255.0
        # mask_t = F.resized_crop(mask_t, top, left, ch, cw, (video_height, video_width), interpolation=InterpolationMode.NEAREST)
        masks_list.append((mask_t > 0.5).float())
    masks_tensor = torch.stack(masks_list, dim=0).unsqueeze(0).to(device) 

    masked_frames_list = []
    for rgb_arr in chunk_masked_rgb_arrays:
        rgb_t = torch.from_numpy(rgb_arr).permute(2, 0, 1).float() / 255.0
        # rgb_t = F.resized_crop(rgb_t, top, left, ch, cw, (video_height, video_width), interpolation=InterpolationMode.BILINEAR, antialias=True)
        masked_frames_list.append(rgb_t)
    masked_frames_tensor = torch.stack(masked_frames_list, dim=0).unsqueeze(0).to(device)

    # 第一帧特殊处理
    image_tensor = masked_frames_tensor[:, 0, :, :, :].squeeze(0).cpu() 
    image_tensor_normalized = (image_tensor * 2.0 - 1.0).unsqueeze(0) 

    masks_tensor[:, 0] = 1.0 
    masked_frames_tensor[:, 0] = image_tensor.to(device) 

    output_frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=video_height,
        width=video_width,
        num_frames=num_frames,
        controlnet_cond=union_input,
        masked_video_frames=masked_frames_tensor * 2.0 - 1.0, 
        mask_frames=masks_tensor.repeat(1, 1, 3, 1, 1), 
        image=image_tensor_normalized.to(device),
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        output_type="pil",
        controlnet_guidance_start=controlnet_guidance_start,
        controlnet_guidance_end=controlnet_guidance_end,
        controlnet_weight=controlnet_weight,
    ).frames[0]
    
    return output_frames


# ==========================================
# --- 5. MAIN AUTOREGRESSIVE LOOP ---
# ==========================================
def generate_long_video(args, pipe):
    pipeline_3d = PointCloudPipeline()
    camera_K = pipeline_3d.load_camera(args.cam_k)
    
    all_depth_files = sorted([os.path.join(args.depth_path, f) for f in os.listdir(args.depth_path) if f.lower().endswith(('.png', '.jpg'))])
    all_normal_files = sorted([os.path.join(args.normal_path, f) for f in os.listdir(args.normal_path) if f.lower().endswith(('.png', '.jpg'))])
    
    ref_img = imageio.v2.imread(args.image_path)
    src_h, src_w = ref_img.shape[:2]
    crop_params = get_crop_params(src_w, src_h, args.video_width, args.video_height)
    
    # 【新增】计算等价于 512x384 空间的修正内参 K
    adjusted_K = adjust_intrinsics(camera_K, crop_params, args.video_width, args.video_height)
    
    total_frames = args.total_frames
    chunk_size = args.num_frames
    stride = chunk_size - 1 
    num_chunks = int(np.ceil((total_frames - 1) / stride))
    
    print(f"[INFO] 计划生成 {total_frames} 帧视频，Chunk 大小为 {chunk_size}，总计 {num_chunks} 个Chunks。")

    final_video_frames = []
    
    # 提前对第一帧原图进行同样的裁剪与缩放
    top, left, ch, cw = crop_params
    init_img = Image.open(args.image_path).convert("RGB")
    # 1. 裁剪
    init_img_cropped = init_img.crop((left, top, left + cw, top + ch))
    # 2. 缩放
    init_img_resized = init_img_cropped.resize((args.video_width, args.video_height), Image.Resampling.BILINEAR)
    
    current_keyframes_rgb = [np.array(init_img_resized)]
    current_keyframes_indices = [0]
    
    for chunk_idx in range(num_chunks):
        print(f"\n========== 开始处理 Chunk {chunk_idx + 1}/{num_chunks} ==========")
        start_idx = chunk_idx * stride
        end_idx = min(start_idx + chunk_size, total_frames)
        current_chunk_length = end_idx - start_idx
        
        target_global_indices = list(range(start_idx, end_idx))
        
        chunk_depths = all_depth_files[start_idx : end_idx]
        chunk_normals = all_normal_files[start_idx : end_idx]
        
        pad_len = chunk_size - current_chunk_length
        if pad_len > 0:
            chunk_depths += [chunk_depths[-1]] * pad_len
            chunk_normals += [chunk_normals[-1]] * pad_len
            target_global_indices += [target_global_indices[-1]] * pad_len
            
        # 核心变动：传入 3D Pipeline 及相关相机矩阵、全局深度路径
        chunk_masked_rgbs, chunk_masks = warp_and_merge_keyframes(
            pipeline_3d=pipeline_3d,
            keyframes_rgb=current_keyframes_rgb,
            keyframes_indices=current_keyframes_indices,
            target_global_indices=target_global_indices,
            all_depth_files=all_depth_files,
            camera_K=adjusted_K,  # 使用修正后的 K
            camera_RT_dir=args.cam_rt_dir,
            crop_params=crop_params,     # 传入裁剪参数
            target_w=args.video_width,   # 传入目标宽
            target_h=args.video_height,  # 传入目标高
            chunk_idx=chunk_idx,
            output_dir=args.output_dir
        )
        
        chunk_output_pil = run_pipeline_chunk(
            pipe=pipe, prompt=args.prompt, chunk_depth_files=chunk_depths, chunk_normal_files=chunk_normals,
            chunk_mask_arrays=chunk_masks, chunk_masked_rgb_arrays=chunk_masked_rgbs, crop_params=crop_params,
            video_height=args.video_height, video_width=args.video_width, num_frames=chunk_size, 
            guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt, seed=args.seed + chunk_idx, controlnet_weight=args.controlnet_weight,
            controlnet_guidance_start=args.controlnet_guidance_start, controlnet_guidance_end=args.controlnet_guidance_end,
            device=args.device
        )
        
        valid_chunk_output = chunk_output_pil[:current_chunk_length]
        
        chunk_video_path = os.path.join(args.output_dir, f"chunk_{chunk_idx}_output.mp4")
        export_to_video(chunk_output_pil, chunk_video_path, fps=args.out_fps)

        if chunk_idx == 0:
            final_video_frames.extend(valid_chunk_output)
        else:
            final_video_frames.extend(valid_chunk_output[1:])
            
        if chunk_idx < num_chunks - 1:
            num_kfs = min(args.num_keyframes, current_chunk_length)
            current_keyframes_rgb = [np.array(img) for img in valid_chunk_output[-num_kfs:]]
            kf_start = start_idx + current_chunk_length - num_kfs
            current_keyframes_indices = list(range(kf_start, kf_start + num_kfs))
            print(f"[INFO] 提取关键帧，全局索引为: {current_keyframes_indices}")

    export_to_video(final_video_frames, args.output_path, fps=args.out_fps)
    print(f"\n[INFO] 全部生成完毕！长视频已保存至: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # === 新增相机参数 ===
    parser.add_argument("--cam_k", type=str, required=True, help="Path to camera intrinsics matrix (.txt or .npy)")
    parser.add_argument("--cam_rt_dir", type=str, required=True, help="Directory containing camera_RT_xxxx.txt files")
    
    parser.add_argument("--total_frames", type=int, default=161)
    parser.add_argument("--num_keyframes", type=int, default=4)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--normal_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    
    parser.add_argument("--base_model", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--controlnet_model", type=str, default="YuryyyLee/world-renderer-controlnet-union")
    parser.add_argument("--controlnet_weight", type=float, default=0.8)
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.8)
    parser.add_argument("--output_path", type=str, default="./output_long.mp4")
    parser.add_argument("--output_dir", type=str, default="./output")
    
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--video_height", type=int, default=384)
    parser.add_argument("--video_width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worst quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_fps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    # ./output_long
    args.output_path = os.path.join(args.output_dir, f"output_long.mp4")
    args.output_dir = os.path.join(args.output_dir, "output_chunks")

    pipe = prepare_pipeline(base_model=args.base_model, controlnet_model=args.controlnet_model, device=args.device, dtype=torch.float32)
    generate_long_video(args, pipe)
```

