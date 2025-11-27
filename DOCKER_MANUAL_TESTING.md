# Manual Docker Testing Guide for NeurIPS

This guide is for running **manual** tests inside Docker on the H200 to find optimal TP/MoE configurations.

## Setup

### 1. SSH into H200
```bash
ssh <user>@<h200-machine>
```

### 2. Start Docker Container
```bash
docker run --gpus all \
  --shm-size 128g \
  -p 30000:30000 \
  -v /mnt/data/cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=<your_hf_token>" \
  --ipc=host \
  --name doug_test_capacity \
  -it \
  lmsysorg/sglang:v0.5.5.post3 \
  bash
```

## Quick Test Template

For each model, test different TP sizes and MoE backends:

### DeepSeek V3 (MoE Model)

**TP4 + flashinfer_trtllm:**
```bash
# Start server
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 4 \
  --quantization fp8 \
  --moe-runner-backend flashinfer_trtllm \
  --trust-remote-code \
  --port 30000 &

# Wait for server to start, then benchmark
sleep 30
python3 -m sglang.bench_one_batch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --base-url http://127.0.0.1:30000 \
  --batch-size 1 \
  --input-len 4096 \
  --output-len 512 \
  --show-report

# Kill server
pkill -f sglang.launch_server
sleep 5
```

**TP4 + flashinfer_cutlass:**
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 4 \
  --quantization fp8 \
  --moe-runner-backend flashinfer_cutlass \
  --trust-remote-code \
  --port 30000 &

sleep 30
python3 -m sglang.bench_one_batch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --base-url http://127.0.0.1:30000 \
  --batch-size 1 \
  --input-len 4096 \
  --output-len 512 \
  --show-report

pkill -f sglang.launch_server
sleep 5
```

**TP8 + flashinfer_trtllm:**
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --quantization fp8 \
  --moe-runner-backend flashinfer_trtllm \
  --trust-remote-code \
  --port 30000 &

sleep 30
python3 -m sglang.bench_one_batch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --base-url http://127.0.0.1:30000 \
  --batch-size 1 \
  --input-len 4096 \
  --output-len 512 \
  --show-report

pkill -f sglang.launch_server
sleep 5
```

**TP8 + flashinfer_cutlass:**
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --quantization fp8 \
  --moe-runner-backend flashinfer_cutlass \
  --trust-remote-code \
  --port 30000 &

sleep 30
python3 -m sglang.bench_one_batch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --base-url http://127.0.0.1:30000 \
  --batch-size 1 \
  --input-len 4096 \
  --output-len 512 \
  --show-report

pkill -f sglang.launch_server
sleep 5
```

### Qwen 235B (MoE Model)

Repeat the same pattern with different model:

```bash
# TP4 + flashinfer_trtllm
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tp 4 \
  --quantization fp8 \
  --moe-runner-backend flashinfer_trtllm \
  --trust-remote-code \
  --port 30000 &

sleep 30
python3 -m sglang.bench_one_batch_server \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --base-url http://127.0.0.1:30000 \
  --batch-size 1 \
  --input-len 4096 \
  --output-len 512 \
  --show-report

pkill -f sglang.launch_server
sleep 5

# ... repeat for other configs
```

### GLM 4.6 (Non-MoE Model)

For non-MoE models, skip the `--moe-runner-backend` flag:

```bash
# TP4 (no MoE backend)
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6 \
  --tp 4 \
  --quantization fp8 \
  --trust-remote-code \
  --port 30000 &

sleep 30
python3 -m sglang.bench_one_batch_server \
  --model zai-org/GLM-4.6 \
  --base-url http://127.0.0.1:30000 \
  --batch-size 1 \
  --input-len 4096 \
  --output-len 512 \
  --show-report

pkill -f sglang.launch_server
sleep 5
```

## All Models to Test

### MoE Models (test both backends):
1. **DeepSeek V3**: `deepseek-ai/DeepSeek-V3`
2. **Qwen 235B**: `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
3. **Qwen 480B Coder**: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`
4. **Minimax M2**: `MiniMaxAI/Minimax-M2`
5. **Kimi K2**: `moonshotai/Kimi-K2-Thinking` (add `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2`)

### Non-MoE Models (no MoE backend):
6. **GLM 4.6**: `zai-org/GLM-4.6`
7. **Llama 3.2**: `meta-llama/Llama-3.2-90B-Vision-Instruct`

## TP Sizes to Test

Based on mentor's feedback:
- **TP1** and **TP2** for smaller models (GLM 4.6, Llama 3.2)
- **TP4** and **TP8** for larger models (DeepSeek, Qwen, etc.)

## What to Record

For each config, note:
- **Output throughput** (tokens/sec) - from benchmark report
- **Latency** (seconds) - from benchmark report
- **ITL** (inter-token latency in ms) - from benchmark report
- **Success/Failure** - did it run or OOM?

## Quick Results Table

Create a spreadsheet like this:

| Model | TP Size | MoE Backend | Throughput (tok/s) | Latency (s) | ITL (ms) | Status |
|-------|---------|-------------|-------------------|-------------|----------|--------|
| DeepSeek V3 | 4 | flashinfer_trtllm | 105.2 | 5.1 | 9.5 | ✓ |
| DeepSeek V3 | 4 | flashinfer_cutlass | 98.3 | 5.3 | 10.2 | ✓ |
| DeepSeek V3 | 8 | flashinfer_trtllm | 110.5 | 4.9 | 9.0 | ✓ |
| DeepSeek V3 | 8 | flashinfer_cutlass | 102.1 | 5.0 | 9.8 | ✓ |
| ... | ... | ... | ... | ... | ... | ... |

Then find the best config for each model!

## Tips

- **If OOM**: Try smaller TP size or skip that model
- **Between tests**: Always wait 5-10 seconds for GPU memory to clear
- **Check GPU**: Run `nvidia-smi` to verify memory is free before next test
- **Save output**: Redirect to file: `... --show-report > results_deepseek_tp4_trtllm.txt`

Good luck! 🚀


Commands:

Deepseek tp4
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 4 \
  --quantization fp8 \
  --moe-runner-backend flashinfer_trtllm \
  --trust-remote-code \
  --port 30000
```
OOM

Deepseek tp8
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --quantization fp8 \
  --moe-runner-backend flashinfer_trtllm \
  --trust-remote-code \
  --port 30000
```
