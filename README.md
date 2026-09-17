# motionnet

### Dependencies
```bash
uv venv --python 3.12
source .venv/bin/activate  
uv sync
uv pip install -e .
```

Package structure 

```bash
motionnet/
├── src/motionnet/
│   ├── dataset/                # Mostly done!
│   ├── models/
│   │   └── cnn.py              # WIP: CNN, basis functions
│   ├── train/
│   │   ├── loop.py             # run_epoch, fit(cfg) -> checkpoint path
│   │   └── losses.py           # rate costs, penalties
│   └── analysis/
│       ├── tuning.py           # layer2_tuning, edge_test, dsi
│       └── layer1.py           # layer1_summary, snr_report
├── scripts/
│   ├── train.py                # parse config, call motionnet.train.fit
│   ├── sweep.py                # loop over λ, seeds
│   └── analyze.py              # load checkpoints, write tables/figures
├── configs/
│   └── base.yaml
└── notebooks/                  # exploration and development; import from motionnet
```