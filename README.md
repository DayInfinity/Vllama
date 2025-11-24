# Vllama: Vision Models Made Easy 🚀

Vllama is a comprehensive CLI tool that simplifies working with vision models and machine learning workflows. Whether you're preprocessing datasets, training models with AutoML, or generating images with state-of-the-art diffusion models, Vllama makes it easy - locally or on cloud GPUs.

---

## ✨ Key Features

- **🔧 Autonomous Data Preprocessing**: Intelligent data cleaning, encoding, scaling, and feature selection
- **🤖 AutoML Training**: Train and compare multiple ML models automatically with hyperparameter tuning
- **🎨 Vision Model Inference**: Generate images using pre-trained diffusion models (Stable Diffusion, SD-Turbo)
- **☁️ Cloud GPU Integration**: Seamlessly offload computation to Kaggle GPUs
- **📊 Rich Visualizations**: Automatic generation of insights, correlations, and performance metrics
- **💾 Smart Output Management**: Organized folder structure with logs, models, and visualizations

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ManvithGopu13/Vllama.git
cd Vllama
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Vllama CLI
```bash
pip install -e .
```

Now you can use `vllama` from anywhere in your terminal!

---

## 🚀 Quick Start Guide

### Data Preprocessing & Model Training Workflow

#### Step 1: Preprocess Your Dataset
Clean and prepare your data for machine learning:

```bash
vllama data --path dataset.csv --target price --test_size 0.2 --output_dir ./outputs
```

**What it does:**
- Automatically detects column types (numerical/categorical)
- Handles missing values intelligently (KNN imputation, median/mode filling)
- Removes duplicates and handles outliers
- Encodes categorical variables (label encoding, one-hot encoding, frequency encoding)
- Scales features using RobustScaler
- Performs feature selection (removes zero-variance and highly correlated features)
- Generates visualizations (missing values heatmap, correlation matrix, etc.)
- Splits data into train/test sets
- Saves processed data as `train_data.csv` and `test_data.csv`

**Parameters:**
- `--path`: Path to your dataset (supports CSV, Excel, JSON, Parquet)
- `--target`: Target column name (auto-detected if not specified)
- `--test_size` or `-t`: Test set proportion (default: 0.2)
- `--output_dir` or `-o`: Output directory (default: current directory)

**Output Structure:**
```
output_folder_YYYYMMDD_HHMMSS/
├── train_data.csv
├── test_data.csv
├── processed_full_data.csv
├── preprocessing_log.json
├── preprocessing_log.txt
├── summary_report.json
├── transformation_metadata.json
└── visualizations/
    ├── 01_missing_initial.png
    ├── 02_dtypes.png
    ├── 03_corr_processed.png
    ├── 04_target_processed.png
    └── 05_mi.png
```

#### Step 2: Train Models with AutoML
Automatically train and compare multiple ML models:

```bash
vllama train --path ./outputs/output_folder_YYYYMMDD_HHMMSS --target price
```

**What it does:**
- Auto-detects task type (classification or regression)
- Trains multiple models with hyperparameter tuning:
  - **Classification**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP, Naive Bayes
  - **Regression**: Random Forest, XGBoost, LightGBM, CatBoost, SVR, KNN, MLP
- Uses RandomizedSearchCV for efficient hyperparameter optimization
- Evaluates models on test set with comprehensive metrics
- Generates visualizations (confusion matrices, ROC curves, prediction plots)
- Saves all models and creates a leaderboard
- Identifies and saves the best performing model

**Parameters:**
- `--path` or `-p`: Path to folder containing `train_data.csv` and `test_data.csv`
- `--target` or `-t`: Target column name

**Output Structure:**
```
results/
├── model_summary.csv          # Leaderboard of all models
├── best_model.pkl             # Best performing model
├── best_model.txt             # Best model details
├── report.html                # HTML report with all results
└── per_model/
    ├── RandomForest/
    │   ├── RandomForest_best_model.pkl
    │   ├── RandomForest_tuning_results.csv
    │   ├── RandomForest_confusion_matrix.png
    │   └── RandomForest_roc_curve.png
    ├── XGBoost/
    └── ...
```

---

### Vision Model Inference Workflow

#### Step 1: Show Available Models
```bash
vllama show models
```

Lists all supported vision models with descriptions.

#### Step 2: Install a Model (Optional)
Pre-download model weights to cache:

```bash
vllama install stabilityai/sd-turbo
```

#### Step 3: Generate Images Locally

**Single Prompt Mode:**
```bash
vllama run stabilityai/sd-turbo --prompt "A serene mountain landscape at sunset" --output_dir ./images
```

**Interactive Mode:**
```bash
vllama run stabilityai/sd-turbo
```
Then enter prompts interactively. Type `exit` or `quit` to stop.

**Parameters:**
- `model`: Model name (e.g., `stabilityai/sd-turbo`)
- `--prompt` or `-p`: Text prompt for image generation
- `--output_dir` or `-o`: Directory to save generated images (default: current directory)
- `--service` or `-s`: Offload to cloud service (e.g., `kaggle`)

**Features:**
- Automatic GPU/CPU detection
- Low VRAM optimization (for GPUs with ≤3GB VRAM)
- Memory-efficient attention (xformers)
- Attention slicing and VAE tiling for better performance

#### Step 4: Generate Images on Kaggle GPU

```bash
vllama run stabilityai/sd-turbo --service kaggle --prompt "A cyberpunk city at night"
```

**What it does:**
- Creates a Kaggle kernel with GPU enabled
- Installs dependencies automatically
- Runs the model on Kaggle's GPU
- Downloads the generated image to your local machine

---

## 📚 Complete Command Reference

### Data & ML Commands

#### `vllama data`
Autonomous data preprocessing and cleaning.

```bash
vllama data --path <dataset> --target <column> [--test_size <float>] [--output_dir <dir>]
```

**Examples:**
```bash
# Basic usage with auto-detected target
vllama data --path sales_data.csv

# Specify target column and test size
vllama data --path housing.csv --target price --test_size 0.25

# Custom output directory
vllama data --path data.csv --target label -t 0.3 -o ./processed_data
```

#### `vllama train`
AutoML model training with hyperparameter tuning.

```bash
vllama train --path <data_folder> --target <column>
```

**Examples:**
```bash
# Train on preprocessed data
vllama train --path ./output_folder_20231124_143022 --target SalePrice

# Short form
vllama train -p ./data -t label
```

---

### Vision Model Commands

#### `vllama show models`
List all supported vision models.

```bash
vllama show models
```

#### `vllama install`
Download and cache a model.

```bash
vllama install <model_name>
```

**Example:**
```bash
vllama install stabilityai/sd-turbo
```

#### `vllama run`
Run a vision model for image generation.

```bash
vllama run <model_name> [--prompt <text>] [--service <service>] [--output_dir <dir>]
```

**Examples:**
```bash
# Single prompt
vllama run stabilityai/sd-turbo --prompt "A beautiful sunset"

# Interactive mode
vllama run stabilityai/sd-turbo

# Run on Kaggle GPU
vllama run stabilityai/sd-turbo --service kaggle --prompt "A dragon flying"

# Custom output directory
vllama run stabilityai/sd-turbo -p "A forest" -o ./my_images
```

#### `vllama post`
Send a prompt to an already running model session.

```bash
vllama post <prompt> [--output_dir <dir>]
```

**Example:**
```bash
vllama post "A magical castle" --output_dir ./outputs
```

#### `vllama stop`
Stop the currently running model session.

```bash
vllama stop
```

---

### Cloud Integration Commands

#### `vllama login`
Authenticate with a cloud GPU service.

```bash
vllama login --service <service> [--username <user>] [--key <api_key>]
```

**Examples:**
```bash
# Login to Kaggle with credentials
vllama login --service kaggle --username myusername --key abc123xyz

# Use existing Kaggle credentials from ~/.kaggle/kaggle.json
vllama login --service kaggle
```

#### `vllama init gpu`
Initialize a GPU session on a cloud service.

```bash
vllama init gpu --service <service>
```

**Example:**
```bash
vllama init gpu --service kaggle
```

#### `vllama logout`
Remove cloud service credentials.

```bash
vllama logout
```

---

## 🎯 Common Workflows

### Workflow 1: Complete ML Pipeline
```bash
# 1. Preprocess data
vllama data --path raw_data.csv --target price

# 2. Train models (use the output folder from step 1)
vllama train --path ./output_folder_20231124_143022 --target price

# 3. Review results in the results/ folder
```

### Workflow 2: Local Image Generation
```bash
# 1. Install model (optional, first-time only)
vllama install stabilityai/sd-turbo

# 2. Generate images interactively
vllama run stabilityai/sd-turbo

# Enter prompts:
# Prompt> A serene lake with mountains
# Prompt> A futuristic city
# Prompt> exit
```

### Workflow 3: Cloud GPU Image Generation
```bash
# 1. Login to Kaggle
vllama login --service kaggle --username myuser --key myapikey

# 2. Generate image on Kaggle GPU
vllama run stabilityai/sd-turbo --service kaggle --prompt "A magical forest"

# Image will be downloaded automatically
```

---

## 📊 Understanding Outputs

### Data Preprocessing Outputs

**Logs:**
- `preprocessing_log.json`: Detailed JSON log of all preprocessing steps
- `preprocessing_log.txt`: Human-readable text log
- `summary_report.json`: Summary statistics and metadata

**Data Files:**
- `train_data.csv`: Training dataset (80% by default)
- `test_data.csv`: Testing dataset (20% by default)
- `processed_full_data.csv`: Complete processed dataset
- `transformation_metadata.json`: Encoders and scalers metadata for future use

**Visualizations:**
- Missing values heatmap
- Data types distribution
- Correlation matrix (top 20 features)
- Target distribution
- Mutual information scores

### Model Training Outputs

**Model Files:**
- `best_model.pkl`: Best performing model (can be loaded with joblib)
- `model_summary.csv`: Comparison of all trained models
- `report.html`: Interactive HTML report

**Per-Model Outputs:**
- `{model}_best_model.pkl`: Saved model
- `{model}_tuning_results.csv`: Hyperparameter search results
- `{model}_confusion_matrix.png`: Confusion matrix (classification)
- `{model}_roc_curve.png`: ROC curve (binary classification)
- `{model}_pred_vs_true.png`: Scatter plot (regression)

### Vision Model Outputs

Generated images are saved as:
```
vllama_output_{timestamp}.png          # Local generation
vllama_kaggle_{timestamp}.png          # Kaggle generation
```

---

## 🔧 Advanced Configuration

### Environment Variables

Create a `.env` file for configuration:

```env
# Kaggle API Credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Model Cache Directory (optional)
HF_HOME=/path/to/cache
```

### GPU Optimization

Vllama automatically optimizes for your GPU:
- **High VRAM (>3GB)**: Uses float16, full resolution (512x512), more inference steps
- **Low VRAM (≤3GB)**: Uses float32, reduced steps, memory-efficient attention
- **CPU**: Falls back to CPU inference (slower but works)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

## 🆘 Troubleshooting

### Common Issues

**Issue: "Kaggle API credentials not found"**
```bash
# Solution: Set up Kaggle credentials
vllama login --service kaggle --username YOUR_USERNAME --key YOUR_API_KEY
```

**Issue: "CUDA out of memory"**
```bash
# Solution: The tool automatically handles low VRAM, but you can also:
# 1. Close other GPU applications
# 2. Use CPU mode (automatic fallback)
# 3. Use Kaggle GPU instead
vllama run model --service kaggle --prompt "your prompt"
```

**Issue: "Target column not found"**
```bash
# Solution: Specify the target column explicitly
vllama data --path data.csv --target your_column_name
```

---

## 📞 Support

- **Documentation**: [GitHub Repository](https://github.com/ManvithGopu13/Vllama)
- **Issues**: [GitHub Issues](https://github.com/ManvithGopu13/Vllama/issues)
- **Email**: manvithgopu1394@gmail.com

---

## 🌟 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)

---

**Made with ❤️ by Gopu Manvith**
