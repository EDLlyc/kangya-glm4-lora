#!/usr/bin/env python3
"""
GLM-4-9B 抗压背锅吧风格 LoRA 推理脚本
用法:
    python inference.py                    # 交互式聊天
    python inference.py --prompt "评价一下 jackeylove"   # 单条推理
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "THUDM/glm-4-9b-chat"
ADAPTER_PATH = "."  # 当前目录，包含 adapter_model.safetensors
SYSTEM_PROMPT = '你是一位百度贴吧"抗压背锅吧"的吧友，说话直接、玩梗多、情绪强烈。'


def load_model():
    print(f"正在加载基础模型: {BASE_MODEL}")
    print("首次运行会从 Hugging Face 下载，请耐心等待...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="left",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"正在加载 LoRA 适配器: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.eval()

    return model, tokenizer


def chat(model, tokenizer, user_input, history=None):
    if history is None:
        history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_input})

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

    history.append((user_input, response))
    return response, history


def interactive_chat(model, tokenizer):
    print("\n===== 抗吧老哥 AI 已就绪 =====")
    print("输入问题开始聊天，输入 'quit' 或 'exit' 退出\n")

    history = []
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        response, history = chat(model, tokenizer, user_input, history)
        print(f"模型：{response}\n")


def single_inference(model, tokenizer, prompt):
    response, _ = chat(model, tokenizer, prompt)
    print(f"用户：{prompt}")
    print(f"模型：{response}")


def main():
    parser = argparse.ArgumentParser(description="抗吧风格 GLM-4 推理")
    parser.add_argument("--prompt", type=str, default=None, help="单条推理提示词")
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.prompt:
        single_inference(model, tokenizer, args.prompt)
    else:
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
