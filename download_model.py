#!/usr/bin/env python3
"""
预下载基础模型 THUDM/glm-4-9b-chat
用法:
    python download_model.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "THUDM/glm-4-9b-chat"

print(f"正在下载基础模型: {BASE_MODEL}")
print("国内用户建议先设置: export HF_ENDPOINT=https://hf-mirror.com")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
_ = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

print("基础模型下载完成！")
