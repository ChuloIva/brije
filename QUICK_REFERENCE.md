# Quick Reference Card

## 🚀 One-Line Training

```bash
# Complete pipeline (recommended)
./train_probe_pipeline.sh

# Or in Python
cd src/probes && python run_full_pipeline.py
```

## 📝 Common Commands

### Train with Custom Options
```bash
# Train multihead probe on layer 21 for 30 epochs
./train_probe_pipeline.sh --layer 21 --probe-type multihead --epochs 30

# Quick test run (5 epochs)
python run_full_pipeline.py --epochs 5 --skip-test

# Use CPU instead of GPU
./train_probe_pipeline.sh --device cpu
```

### Test Existing Probe
```bash
cd src/probes
python probe_inference.py \
    --probe ../../data/probes/best_probe.pth \
    --text "Your text here"
```

### View Metrics
```bash
# JSON format
cat data/probes/test_metrics.json

# Pretty print
python3 -m json.tool data/probes/test_metrics.json
```

### Analyze Dataset
```bash
cd src/probes
python dataset_utils.py \
    ../../third_party/datagen/generated_data/stratified_combined_31500.jsonl
```

## 📊 Pipeline Stages

| Stage | Time (GPU) | Output |
|-------|-----------|--------|
| Activation Capture | 2-3 hours | `layer_27_activations.h5` |
| Probe Training | 15-30 min | `best_probe.pth` |
| Testing | 1-2 min | Performance metrics |

## 🎯 Expected Performance

| Metric | Linear Probe | MultiHead Probe |
|--------|-------------|----------------|
| Accuracy | 70-80% | 75-85% |
| Macro F1 | 0.65-0.75 | 0.70-0.80 |
| Training Time | 15 min | 25 min |

## 🔧 File Locations

```
brije/
├── train_probe_pipeline.sh          # Main pipeline script
├── src/probes/
│   ├── run_full_pipeline.py         # Python pipeline script
│   ├── capture_activations.py       # Step 1
│   ├── train_probes.py              # Step 2
│   └── probe_inference.py           # Step 3
├── data/
│   ├── activations/                 # Cached activations
│   │   └── layer_27_activations.h5
│   └── probes/                      # Trained models
│       ├── best_probe.pth           # Best checkpoint
│       ├── final_probe.pth          # Final model
│       ├── test_metrics.json        # Test performance
│       └── training_history.json    # Training curves
└── third_party/datagen/generated_data/
    └── stratified_combined_31500.jsonl  # Training data
```

## ⚡ Quick Debugging

### Check if activations exist
```bash
ls -lh data/activations/layer_27_activations.h5
```

### Check if probe trained
```bash
ls -lh data/probes/best_probe.pth
```

### View last training metrics
```bash
tail -20 data/probes/test_metrics.json
```

### Test specific cognitive action
```bash
cd src/probes
python probe_inference.py \
    --probe ../../data/probes/best_probe.pth \
    --text "She was reconsidering her approach" \
    --top-k 3
```

## 🎨 Using in Liminal Backrooms

```bash
cd third_party/liminal_backrooms
python main.py
```

1. Select: **"Gemma 3 4B (with Probes)"**
2. Choose: **"Cognitive Roles - Analyst vs Creative"**
3. Set turns: **5-10**
4. Click: **Propagate**
5. Watch cognitive actions in console!

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| Out of memory | `--batch-size 16` or `--device cpu` |
| Activations not found | Check path or run capture step |
| Low accuracy | Try `--probe-type multihead` or `--epochs 50` |
| Slow inference | Use `--layer 27` (last layer only) |

## 💡 Tips

- **First time?** Run `./train_probe_pipeline.sh` and follow prompts
- **Testing changes?** Use `--skip-capture` to reuse activations
- **Production?** Use `--probe-type multihead` for better accuracy
- **Quick experiment?** Try `--epochs 5` for faster iteration
- **Multiple layers?** Run pipeline separately for each layer

## 📚 Help Commands

```bash
# Pipeline help
./train_probe_pipeline.sh --help
python run_full_pipeline.py --help

# Individual scripts
python capture_activations.py --help
python train_probes.py --help
python probe_inference.py --help
```

## 🎯 Example Workflow

```bash
# 1. Train probe (one command)
./train_probe_pipeline.sh

# 2. Test with your text
cd src/probes
python probe_inference.py \
    --probe ../../data/probes/best_probe.pth \
    --text "I'm reconsidering my approach to this problem"

# 3. Use in GUI
cd ../../third_party/liminal_backrooms
python main.py
```

## 📊 Performance Checklist

After training, check:
- [ ] Test accuracy > 70%
- [ ] Macro F1 > 0.65
- [ ] Top-3 accuracy > 85%
- [ ] No class with F1 < 0.40
- [ ] Training converged (loss plateaued)

If any fail:
- Increase epochs: `--epochs 50`
- Try multihead: `--probe-type multihead`
- Check data balance: `python dataset_utils.py [dataset]`