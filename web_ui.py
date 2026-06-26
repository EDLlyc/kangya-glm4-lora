#!/usr/bin/env python3
"""
抗吧老哥 AI 网页对话界面
基于 Gradio，支持 4-bit 量化

用法:
    python web_ui.py              # 默认 fp16
    python web_ui.py --load_in_4bit   # 8GB 显存模式
"""

import argparse
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "THUDM/glm-4-9b-chat"
ADAPTER_PATH = "."
SYSTEM_PROMPT = '你是一位百度贴吧"抗压背锅吧"的吧友，说话直接、玩梗多、情绪强烈。'

model = None
tokenizer = None


def load_model(load_in_4bit=False):
    global model, tokenizer

    print(f"正在加载基础模型: {BASE_MODEL}")
    if load_in_4bit:
        print("使用 4-bit 量化，约需 6-8GB 显存")
    else:
        print("使用 fp16，约需 16-20GB 显存")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="left",
    )

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    print(f"正在加载 LoRA 适配器: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.eval()
    print("模型加载完成！")


def chat_fn(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response


def main():
    parser = argparse.ArgumentParser(description="抗吧老哥 AI 网页版")
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="使用 4-bit 量化，8GB 显存用户必选",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="生成 Gradio 公开分享链接",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务端口，默认 7860",
    )
    args = parser.parse_args()

    load_model(load_in_4bit=args.load_in_4bit)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="🏆 抗吧老哥 AI",
        description="基于 GLM-4-9B + LoRA 微调的贴吧风格对话模型",
        examples=[
            "评价一下 jackeylove",
            "BLG 今年能夺冠吗？",
            "TheShy 这场表现怎么样？",
            "左手和右手谁更强？",
        ],
        theme="soft",
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
