TEXASHOLDEMAI/
│
├── ai/                     # All AI-related logic goes here
│   ├── __init__.py
│   ├── Kane/               # Kane's rule-based AI logic
│   │   ├── __init__.py
│   │   ├── kane_rule_based.py
│   │   └── strategies/     # Folder for Kane's advanced strategies
│   │       ├── __init__.py
│   │       ├── aggressive.py
│   │       └── defensive.py
│   │
│   └── Abel/               # Abel's ML-based AI logic
│       ├── __init__.py
│       ├── abel_rl_based.py
│       ├── abel_q_table.json
│       └── training/       # Abel's training scripts and utilities
│           ├── __init__.py
│           ├── abel_gcp_10000.py
│           └── abel_training_utils.py
│
├── game/                   # Game environment and utilities
│   ├── __init__.py
│   ├── holdem_env.py
│   └── holdem_env_abel.py
│
├── tests/                  # Unit and integration tests for both AIs
│   ├── __init__.py
│   ├── test_kane.py
│   └── test_abel.py
│
├── logs/                   # Centralized logging
│   ├── kane_log.txt
│   ├── abel_training_log.txt
│   └── encoded_states_log.txt
│
├── notebooks/              # Jupyter Notebooks for Analysis & Experimentation
│   └── analysis.ipynb
│
├── models/                 # Folder to store trained models
│   ├── checkpoints/        # Checkpoints for Abel's model
│   │   └── abel_checkpoint.h5
│   └── abel_model.h5
│
├── utils/                  # Utility scripts and helper functions
│   ├── __init__.py
│   └── logger.py
│
├── main.py                 # Main entry point for running simulations
├── requirements.txt        # Python dependencies
└── README.md                # Project Documentation

