```
tron-tgnn-forensics/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw_seeds/
│   ├── stage1_raw_logs/
│   └── stage2_processed/
│
├── src/
│   ├── __init__.py
│   │
│   ├── cacher/ 
│   │   ├── __init__.py
│   │   ├── rpc_client.py
│   │   └── log_extractor.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── temporal_feat.py
│   │   └── graph_builder.py
│   │
│   └── models/
│       ├── __init__.py
│       ├── tgnn_model.py
│       ├── train.py
│       └── evaluate.py
│
└── .gitignore
```