# Parallel Training - Quick Reference Card

## 🚀 For Local Training (40GB GPU)

### Basic Command
```bash
./train_probe_pipeline_parallel.sh
```

### With Custom Settings
```bash
./train_probe_pipeline_parallel.sh \
  --num-workers 8 \
  --batch-size 512 \
  --layer 27
```

### Maximum Speed (12 workers)
```bash
./train_probe_pipeline_parallel.sh \
  --num-workers 12 \
  --batch-size 1024
```

### Conservative (4 workers)
```bash
./train_probe_pipeline_parallel.sh \
  --num-workers 4 \
  --batch-size 256 \
  --no-pin-activations
```

---

## 📊 For Google Colab

### Key Changes in Notebook

**1. Configuration:**
```python
CONFIG = {
    'num_workers': 8,
    'batch_size': 512,
    'pin_activations_to_gpu': True,
}
```

**2. Training Command:**
```python
# Change this line:
'python', 'src/probes/train_binary_probes.py',

# To this:
'python', 'src/probes/train_binary_probes_parallel.py',

# And add:
'--num-workers', str(CONFIG['num_workers']),
'--pin-activations-to-gpu' if CONFIG['pin_activations_to_gpu'] else '--no-pin-activations',
```

---

## ⚙️ Configuration Presets

### 🏎️ Maximum Performance (40GB GPU)
| Setting | Value |
|---------|-------|
| num_workers | 12 |
| batch_size | 1024 |
| pin_activations | True |
| Expected time | ~10-15 min/layer |

### ⚡ Balanced (Recommended)
| Setting | Value |
|---------|-------|
| num_workers | 8 |
| batch_size | 512 |
| pin_activations | True |
| Expected time | ~5-10 min/layer |

### 🛡️ Conservative (Stable)
| Setting | Value |
|---------|-------|
| num_workers | 4 |
| batch_size | 256 |
| pin_activations | False |
| Expected time | ~10-15 min/layer |

### 🧪 Testing/Debug
| Setting | Value |
|---------|-------|
| num_workers | 2 |
| batch_size | 128 |
| pin_activations | False |
| layer_start | 27 |
| layer_end | 27 |

---

## 📈 Performance Expectations

### Sequential Training
```
Time per layer:    20-30 minutes
Total (25 layers): 8-12 hours
GPU utilization:   20-30%
```

### Parallel Training (8 workers)
```
Time per layer:    5-10 minutes
Total (25 layers): 2-3 hours
GPU utilization:   70-90%
Speedup:          8x ⚡
```

---

## 🔍 Monitoring

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Training Progress
Training shows live progress:
```
[01/45] ✓ action1 | AUC: 0.9234 | F1: 0.8765 | ETA: 8.3m
[02/45] ✓ action2 | AUC: 0.9156 | F1: 0.8821 | ETA: 7.9m
...
```

---

## 🐛 Troubleshooting

### Out of Memory?
```bash
# Reduce workers and batch size
--num-workers 4 --batch-size 256 --no-pin-activations
```

### Too Slow?
```bash
# Increase workers and batch size
--num-workers 12 --batch-size 1024
```

### Want to Test First?
```bash
--num-workers 2 --batch-size 128 --layer 27
```

---

## 📁 Output Structure

```
data/probes_binary/
├── probe_action1.pth
├── probe_action2.pth
├── ... (45 probe files)
├── aggregate_metrics.json     # Includes num_workers, speedup
└── metrics_action*.json       # Per-action metrics
```

### Metrics Include:
- `num_workers`: Parallel workers used
- `total_training_time_seconds`: Actual time taken
- `average_auc_roc`: Average performance
- All standard accuracy metrics

---

## 💡 Tips

1. **Start with balanced settings** (8 workers, batch 512)
2. **Monitor GPU usage** - should be 70-90%
3. **Test with one layer first** if unsure
4. **Pin activations** if you have >30GB VRAM
5. **More workers = faster** but needs more VRAM

---

## ⚡ Quick Commands

```bash
# Default (recommended)
./train_probe_pipeline_parallel.sh

# Max speed
./train_probe_pipeline_parallel.sh --num-workers 12 --batch-size 1024

# Test one layer
./train_probe_pipeline_parallel.sh --layer 27 --num-workers 4

# Conservative
./train_probe_pipeline_parallel.sh --num-workers 4 --batch-size 256

# Help
./train_probe_pipeline_parallel.sh --help
```

---

**🚀 Speedup: 8x faster than sequential training!**

