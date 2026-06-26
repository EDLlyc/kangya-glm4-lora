# 抗压背锅吧风格 GLM-4-9B LoRA 微调

基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架，对 [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) 进行 LoRA 微调，让模型学会百度贴吧「抗压背锅吧」老哥风格的回复。

## 硬件要求

| 模式 | 显存需求 | 说明 |
|------|---------|------|
| 默认 fp16 | ≥ 16GB，推荐 24GB | 速度快，显存占用大 |
| 4-bit 量化 | ≥ 6GB，推荐 8GB | 速度慢一些，8GB 显卡可跑 |

- **内存**: ≥ 16GB（推荐 32GB）
- **磁盘**: ≥ 20GB 空闲空间

## 仓库文件

| 文件 | 说明 |
|------|------|
| `adapter_model.safetensors` | LoRA 权重（~81MB） |
| `adapter_config.json` | LoRA 配置 |
| `kangya_beitie.jsonl` | 清洗后的训练数据（10,212 条） |
| `inference.py` | 本地一键推理脚本 |
| `requirements.txt` | 依赖列表 |
| `download.py` | 下载基础模型脚本（可选） |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/EDLlyc/kangya-glm4-lora.git
cd kangya-glm4-lora
```

国内用户可使用代理加速：

```bash
git clone https://ghproxy.com/https://github.com/EDLlyc/kangya-glm4-lora.git
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载基础模型（自动）

运行推理脚本时会自动从 Hugging Face 下载 `THUDM/glm-4-9b-chat`。

国内用户建议先设置镜像：

```bash
# Linux / Mac
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### 4. 运行推理

**8GB 显存用户（4-bit 量化）：**

```bash
python inference.py --load_in_4bit
```

**16GB 及以上显存用户（fp16）：**

```bash
python inference.py
```

**单条测试：**

```bash
python inference.py --load_in_4bit --prompt "评价一下 jackeylove"
```

## 训练参数

| 参数 | 数值 |
|------|------|
| 基础模型 | THUDM/glm-4-9b-chat |
| 微调方式 | LoRA |
| LoRA r | 8 |
| LoRA alpha | 16 |
| cutoff_len | 1024 |
| batch size | 1 |
| gradient accumulation | 16 |
| epochs | 3 |
| 训练样本 | 10,212 |

## 效果示例

```text
用户：评价一下 jackeylove
模型：🐦🐦: 小🐦辛苦了

用户：BLG 今年能夺冠吗？
模型：首先得赢tes

用户：TheShy 这场的表现？
模型：不如48
```

## 继续训练

如果你想用本项目的数据集继续训练：

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 复制数据集
mkdir -p data
cp /path/to/kangya-glm4-lora/kangya_beitie.jsonl data/

# 创建 dataset_info.json
cat > data/dataset_info.json <<'EOF'
{
  "kangya_beitie": {
    "file_name": "kangya_beitie.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
EOF

# 训练
llamafactory-cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path THUDM/glm-4-9b-chat \
  --finetuning_type lora \
  --lora_target all \
  --template glm4 \
  --dataset kangya_beitie \
  --cutoff_len 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --fp16 True \
  --output_dir ./output
```

## 免责声明

训练数据来源于网络公开贴吧评论，仅供学习研究使用。模型输出不代表作者观点，请勿用于违法违规用途。
