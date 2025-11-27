# NeurIPS TP and MoE Configuration Testing

This document explains the batch-1 performance testing setup for finding optimal TP (Tensor Parallelism) and MoE (Mixture of Experts) configurations for the NeurIPS demo.

## What This Does

The test script ([test/nightly/test_neurips_tp_moe_configs.py](test/nightly/test_neurips_tp_moe_configs.py)) runs **batch size 1** benchmarks with different configurations to generate performance traces similar to the nightly tests.

### Models Tested

1. **DeepSeek V3** (MoE) - `deepseek-ai/DeepSeek-V3`
2. **Qwen 235B** (MoE) - `Qwen/Qwen3-235B-A22B-Instruct-2507`
3. **Qwen 480B Coder** (MoE) - `Qwen/Qwen3-Coder-480B-A35B-Instruct`
4. **Minimax M2** (MoE) - `MiniMaxAI/Minimax-M2`
5. **Kimi K2 Thinking** (MoE) - `moonshotai/Kimi-K2-Thinking`
6. **GLM 4.6** - `zai-org/GLM-4.6`
7. **Llama 3.2** - `meta-llama/Llama-3.2-90B-Vision-Instruct`

### Configurations Tested

For each model:
- **TP4** (4 GPUs with Tensor Parallelism)
- **TP8** (8 GPUs with Tensor Parallelism)

For MoE models, additionally test each TP size with:
- **flashinfer_trtllm** (optimized for low latency, batch size < 48)
- **flashinfer_cutlass** (optimized for high throughput, batch size > 48)

### Total Configurations

- **MoE models** (5 models × 2 TP sizes × 2 MoE backends) = **20 configs**
- **Non-MoE models** (2 models × 2 TP sizes) = **4 configs**
- **Total**: **24 configurations**

## How to Run

### Via GitHub Actions (Recommended)

1. Go to: **Actions** → **Stress Test** workflow
2. Click **"Run workflow"**
3. Click **"Run workflow"** button (ignore the input fields, they're not used)
4. Wait for completion (~4-8 hours)
5. View results in the workflow summary

### Output

The workflow will generate a markdown table in the GitHub Actions summary like this:

```markdown
### deepseek-ai/DeepSeek-V3 (TP4_flashinfer_trtllm) [8-gpu-h200]
| batch size | input len | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) | profile (extend) | profile (decode)|
| ---------- | --------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ | ---------------- | --------------- |
| 1 | 4096 | 5.17 | 21040.18 | 102.90 | n/a | 9.72 | 0.04 | 5.40 | [trace](https://...) | [trace](https://...) |
```

You'll get one table for each of the 24 configurations.

### Profile Traces

Each configuration generates two traces:
- **profile (extend)** - Prefill/extend phase trace
- **profile (decode)** - Decode phase trace

Click the `[trace]` links to view performance in Perfetto (Chrome required).

## What to Look For

For each model, compare:

1. **Throughput** (tokens/sec) - Higher is better
2. **Latency** (seconds) - Lower is better
3. **ITL** (Inter-Token Latency in ms) - Lower is better

### Example Analysis

If you see for DeepSeek V3:
- TP4 + flashinfer_trtllm: **105 tok/s output**
- TP4 + flashinfer_cutlass: **98 tok/s output**
- TP8 + flashinfer_trtllm: **110 tok/s output**
- TP8 + flashinfer_cutlass: **102 tok/s output**

**Conclusion**: Use **TP8 + flashinfer_trtllm** for DeepSeek V3

## Modifying the Test

### Change Models

Edit [test/nightly/test_neurips_tp_moe_configs.py](test/nightly/test_neurips_tp_moe_configs.py):

```python
MODELS = {
    "your-model": {
        "path": "org/model-name",
        "is_moe": True,  # or False
        "extra_args": ["--trust-remote-code"],
    },
}
```

### Change TP Sizes

```python
TP_SIZES = [1, 2, 4, 8]  # Test TP1, TP2, TP4, TP8
```

### Change MoE Backends

```python
MOE_BACKENDS = [
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "triton_fuse_moe"  # Add fallback option
]
```

### Change Input/Output Lengths

```python
INPUT_LENS = (4096, 8192)  # Test multiple input lengths
OUTPUT_LENS = (512, 1024)  # Test multiple output lengths
```

## Files Changed

1. **[.github/workflows/stress-test.yml](.github/workflows/stress-test.yml)** - Workflow now runs TP/MoE config tests
2. **[test/nightly/test_neurips_tp_moe_configs.py](test/nightly/test_neurips_tp_moe_configs.py)** - New test script

## How It Works Internally

1. **For each model and config:**
   - Launch SGLang server with specific TP size and MoE backend
   - Run `sglang.bench_one_batch_server` with batch size 1
   - Capture performance profiles (extend and decode traces)
   - Parse results into BenchmarkResult objects

2. **Generate markdown report:**
   - Convert results to markdown tables
   - Upload traces to sglang-ci-data repo
   - Write summary to GitHub Actions step summary

3. **Cleanup:**
   - Kill server process
   - Move to next configuration

## Troubleshooting

### Workflow Fails on a Model

Check the logs - the test continues even if one model fails. The summary will show which configs passed/failed.

### Traces Don't Appear

Make sure:
- `GITHUB_TOKEN` secret is set (for uploading traces)
- `PERFETTO_RELAY_URL` variable is configured
- The workflow has write access to sglang-ci-data repo

### Out of Memory

Some models may not fit with TP4. The test will fail but continue to other configs.

## Next Steps

After the workflow completes:

1. **Download the results** from the GitHub Actions summary
2. **Analyze the traces** using Perfetto
3. **Compare metrics** across configurations
4. **Report optimal configs** to your mentor:
   - "DeepSeek V3: TP8 + flashinfer_trtllm"
   - "Qwen 235B: TP4 + flashinfer_cutlass"
   - etc.

Good luck with the NeurIPS demo! 🚀
