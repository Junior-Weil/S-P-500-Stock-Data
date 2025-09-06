# S-P-500-Stock-Data

Data Source: https://www.kaggle.com/datasets/camnugent/sandp500

Repo Structure:
nn-playground/
├─ data/ # small CSVs or NPZ you actually commit
│ ├─ raw
│ └─ processed
├─ py/ # EDA + driver scripts
│ ├─ requirements.txt
│ └─ eda.ipynb # quick EDA & baseline plots
├─ cpp/ # C++ NN (standalone CLI)
│ ├─ CMakeLists.txt
│ └─ src/
├─ rust/ # Rust NN (standalone CLI)
│ ├─ Cargo.toml
│ └─ src/
├─ specs/ # one-page notes: IO format, metrics, CLI args
└─ README.md
