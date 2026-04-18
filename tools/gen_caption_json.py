import os, json, re, argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_json(text: str):
    """
    尽量从模型输出里抠出一个合法 JSON（允许模型前后夹杂说明文本）。
    """
    text = text.strip()
    # 先直接尝试整段解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再找第一个大括号块
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("模型输出中未找到 JSON 对象。原输出：\n" + text[:800])

    block = m.group(0)
    # 再试解析
    try:
        return json.loads(block)
    except Exception as e:
        raise ValueError(f"找到 JSON 块但解析失败：{e}\nJSON块前800字符：\n{block[:800]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="clip mp4 path")
    ap.add_argument("--out_dir", required=True, help="output folder, e.g. .../0023/text")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=160)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = f"{args.start:06d}_{args.end:07d}.json"  # 000000_0000160.json
    out_path = os.path.join(args.out_dir, out_name)

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    instruction = (
        "You are a professional video captioner for text-to-video datasets.\n"
        "Given the video clip, produce ONE JSON object exactly in the following schema:\n"
        "{\n"
        '  "captions": {\n'
        '    "Short_Caption": "...",\n'
        '    "PC_Caption": "...",\n'
        '    "Background_Caption": "...",\n'
        '    "Camera_Caption": "...",\n'
        '    "Video_Caption": "...",\n'
        '    "Key_Tags": "tag1, tag2, tag3, ..."\n'
        "  }\n"
        "}\n"
        "Rules:\n"
        "- Output JSON only. No markdown, no extra keys.\n"
        "- Short_Caption: 1 sentence (<=25 words).\n"
        "- PC_Caption: describe the primary subject(s) and their actions over time.\n"
        "- Background_Caption: environment, objects, atmosphere.\n"
        "- Camera_Caption: camera movement, framing, stability.\n"
        "- Video_Caption: combine everything into a detailed multi-sentence description.\n"
        "- Key_Tags: 8~20 concise comma-separated tags.\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    gen = out_ids[0, inputs["input_ids"].shape[1]:]
    out_text = processor.decode(gen, skip_special_tokens=True).strip()

    data = extract_json(out_text)

    # 最基本的结构检查
    if not isinstance(data, dict) or "captions" not in data:
        raise ValueError("输出 JSON 结构不符合要求：缺少顶层 captions。原输出：\n" + out_text[:800])

    # 写文件
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
