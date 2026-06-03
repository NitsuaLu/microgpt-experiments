# microgpt-experiments

Some attempts, based on [microgpt](https://github.com/karpathy/makemore).  
一些尝试，基于 MicroGPT。

---

## KV Cache Benchmark

KV Cache 性能对比实验，比较 attention 计算在有无缓存下的复杂度差异（O(n) vs O(n²)）。

### 文件说明

| 文件 | 作用 |
|---|---|
| `microgpt_kv_cache.py` | 带 KV Cache 的完整训练 + 推理，每步打印耗时 |
| `microgpt_no_kv_cache.py` | 无缓存版本，推理时每生成一个新 token 都从头重跑全序列 |
| `microgpt_benchmark.py` | 训练一次，用同一份权重对比两种推理方式，画图验证 |
| `attention_complexity.py` | 纯 micro-benchmark，不训模型，直接测 attention 的 O(n) vs O(n²) 复杂度 |

### 快速开始

```bash
pip install torch matplotlib seaborn

# 1. 复杂度对比（最快，秒出图）
python attention_complexity.py

# 2. 实际模型对比（训练 + benchmark + 画图）
python microgpt_benchmark.py

# 3. 单独跑两个版本看每步耗时
python microgpt_kv_cache.py
python microgpt_no_kv_cache.py
```

数据集 `input.txt` 首次运行时会自动下载。

### 预期结果

- `attention_complexity.py` 的 log-log 图上，无缓存斜率为 2（O(n²)），有缓存斜率为 1（O(n)）
- `microgpt_benchmark.py` 验证两个版本生成的名字完全一致，KV Cache 版本快 8-10 倍（取决于 block_size）
