# 抗压背锅吧风格 GLM-4-9B LoRA 权重与数据集

仅包含 LoRA 权重、配置文件和数据集，方便在 AutoDL 等云服务器上快速拉取和运行。

## 文件说明

| 文件 | 说明 | 大小 |
|------|------|------|
| `adapter_model.safetensors` | LoRA 权重 | ~81MB |
| `adapter_config.json` | LoRA 配置 | 711B |
| `kangya_beitie.jsonl` | 清洗后的训练数据 | ~5.5MB |
| `download.sh` | 一键下载脚本（带 AutoDL 学术加速） | - |

## 基础模型

- **GLM-4-9B-Chat**: https://huggingface.co/THUDM/glm-4-9b-chat

在 AutoDL 上可通过学术加速快速下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download THUDM/glm-4-9b-chat
```

## 快速开始

### 1. 克隆本仓库（AutoDL 学术加速）

```bash
git clone https://ghproxy.com/https://github.com/EDLlyc/kangya-glm4-lora.git
```

或直接：

```bash
git clone https://github.com/EDLlyc/kangya-glm4-lora.git
```

### 2. 安装 LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 3. 复制权重和数据集

```bash
cd /path/to/LLaMA-Factory
mkdir -p data saves/glm-4-9b/lora/sft

# 数据集
cp /path/to/kangya-glm4-lora/kangya_beitie.jsonl data/

# 注册数据集
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

# 权重
cp /path/to/kangya-glm4-lora/adapter_config.json saves/glm-4-9b/lora/sft/
cp /path/to/kangya-glm4-lora/adapter_model.safetensors saves/glm-4-9b/lora/sft/
```

### 4. 推理测试

```bash
source /root/miniconda3/bin/activate llama  # 或你自己的环境
llamafactory-cli chat \
  --model_name_or_path /path/to/glm-4-9b-chat \
  --adapter_name_or_path saves/glm-4-9b/lora/sft \
  --template glm4
```

### 5. 继续训练

```bash
llamafactory-cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path /path/to/glm-4-9b-chat \
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
  --output_dir /path/to/output
```

## 训练参数

| 参数 | 数值 |
|------|------|
| 基础模型 | GLM-4-9B-Chat |
| 微调方式 | LoRA |
| LoRA r | 8 |
| LoRA alpha | 16 |
| cutoff_len | 1024 |
| batch size | 1 |
| gradient accumulation | 16 |
| epochs | 3 |
| 训练样本 | 10,212 |

## 免责声明

训练数据来源于网络公开贴吧评论，仅供学习研究使用。模型输出不代表作者观点。
