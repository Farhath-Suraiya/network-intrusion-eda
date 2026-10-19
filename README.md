# Network Intrusion Detection and Cyberattack Pattern Analysis Using EDA, Statistical Visualization and Deep Learning

**Course**: U21ADP05 – Exploratory Data Analysis and Visualization (Assignment II)  
**Institution**: KPR Institute of Engineering and Technology  
**Department**: Artificial Intelligence and Data Science  
**Student Name**: Farhath Suraiya S D  
**Roll Number / Register Number**: 24AD035  
**Degree & Branch**: B.Tech. Artificial Intelligence and Data Science  
**Academic Year**: 2026–2027  
**Date of Submission**: 15.10.2026  

**Curriculum Mapping**:
- **CO3**: Data Visualization (8 Marks)
- **CO4**: Statistical and Distribution-Based Visualization (16 Marks)
- **CO5**: Application of EDA and Visualization (16 Marks)

---
## Data Preprocessing

See [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md) for the dataset source,
cleaning workflow, preprocessing steps, and reproducibility notes.

Implementation: `src/data_loader.py` and `src/preprocessing.py`.

## 1. Project Overview & Objective

Modern enterprise and cloud networks are subjected to automated, high-velocity, and evasive cyber threats, including Distributed Denial of Service (DDoS), Port Scanning reconnaissance, Brute-Force credentials stuffing, and Web Infiltration exploits. Traditional signature-based detection systems struggle to detect novel attack variants and zero-day intrusions.

### Verified Experimental Performance Highlights (Held-Out Test Set):
- **Overall Accuracy**: **99.92%** (14,860 / 14,872 flows correctly classified)
- **Attack Recall (Sensitivity)**: **99.97%** (7,157 of 7,159 DDoS attacks intercepted; **only 2 missed**)
- **Benign Precision**: **99.97%** (**only 10 false alarms** across 7,713 benign flows)
- **Macro F1-Score**: **0.9992** | **Weighted F1-Score**: **0.9992**
- **ROC-AUC**: **0.9997** | **Average Precision (PR-AUC)**: **0.9997**
- **Unit Test Suite**: **7 / 7 automated tests passing** (100% verified)

### Primary Objectives:
1. **Exploratory Data Analysis (CO3)**: Uncover empirical network flow dynamics, detect collinear attributes, and expose traffic skewness through 8 publication-grade visualizations.
2. **Statistical & Distribution-Based Analysis (CO4)**: Formulate parametric and non-parametric statistical descriptors, evaluate empirical probability distributions, test Gaussian normality using the Kolmogorov-Smirnov test, and perform two-sided Mann-Whitney U hypothesis testing with Rank-Biserial effect sizes.
3. **Deep Learning Classification (CO5)**: Design a Multilayer Perceptron (MLP) deep neural network using TensorFlow/Keras with Batch Normalization, Dropout regularization, cost-sensitive class weights, and leak-free stratified preprocessing.
4. **Diagnostic Evaluation**: Produce held-out test diagnostics, including Confusion Matrices, ROC-AUC, and Precision-Recall Curves (PR-AUC) tailored for extreme class imbalances.

---

## 2. Real-World Applications

- **Enterprise Security Operations Centers (SOC)**: Automated triage and prioritization of high-volume network telemetry to reduce analyst alert fatigue.
- **Next-Generation Firewalls (NGFW)**: In-line classification of non-linear traffic bursts into benign sessions versus volumetric DDoS exploits.
- **Zero-Trust Network Architecture**: Behavioral baseline validation and dynamic access policy enforcement based on flow duration and packet transmission profiles.

---

## 3. Dataset Source & Setup Instructions

This project evaluates the official **CIC-IDS2017 (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset)** developed by the University of New Brunswick (UNB).

### Dataset Metadata:
- **Provider**: Canadian Institute for Cybersecurity (UNB)
- **Official Portal**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Kaggle Mirror**: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/cicdataset/cicids2017)
- **Capture Window**: 5 consecutive weekdays (Monday, July 3, 2017 – Friday, July 7, 2017)
- **Flow Feature Extraction**: ~79 statistical network flow metrics extracted via `CICFlowMeter`.

### Available Daily Capture Files:
1. `Monday-WorkingHours.pcap_ISCX.csv` — Purely benign normal traffic baseline.
2. `Tuesday-WorkingHours.pcap_ISCX.csv` — FTP-Patator and SSH-Patator brute-force attacks.
3. `Wednesday-workingHours.pcap_ISCX.csv` — DoS GoldenEye, Slowloris, Slowhttptest, Hulk, and Heartbleed.
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` — Web Attacks (Brute Force, XSS, SQL Injection).
5. `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` — Infiltration exploits.
6. `Friday-WorkingHours-Morning.pcap_ISCX.csv` — Botnet ARES infrastructure.
7. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` — Network port scanning reconnaissance.
8. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` — Volumetric DDoS LOIC traffic.

### Step-by-Step Dataset Download Instructions:
1. Navigate to the [Kaggle CIC-IDS2017 Mirror](https://www.kaggle.com/datasets/cicdataset/cicids2017) or the [UNB Dataset Portal](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Download any single daily CSV for evaluation (recommended starting files):
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (~225,000 flows, balanced binary & DDoS).
   - `Wednesday-workingHours.pcap_ISCX.csv` (Diverse DoS attacks).
3. Place the downloaded `.csv` file directly into the `data/` directory inside this project:
   ```
   d:/Network_Intrusion_EDA/data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```
4. *Notice*: The software automatically detects any placed CSV file in `data/`. If no CSV file is present, the code halts with an informative diagnostic message rather than fabricating data.

### Data Collection and Preprocessing Overview:
1. **Dataset Source**: This project uses the CIC-IDS2017 dataset, which contains network traffic records representing benign activities and different cyberattacks.
   - Official dataset: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
   - Evaluated file: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
2. **Raw Dataset**:
   - Raw records analyzed: 50,000
   - Benign traffic: 26,135 records (52.27%)
   - Attack traffic: 23,865 records (47.73%)
   - Data type: Numerical network traffic features and a target class label (`Binary_Label`: `BENIGN` vs `ATTACK`)
3. **Data Cleaning**:
   - **Duplicate records**: Duplicate rows were identified and removed. The recorded cleaning summary reports 427 duplicates removed.
   - **Infinite values**: 23 infinite values (`+inf` in `Flow Bytes/s` and `Flow Packets/s`) were identified and resolved.
   - **Missing values**: Three missing values were handled during cleaning. Missing-value checks and median imputation are also part of the preprocessing workflow.
   - **Constant features**: Features with no variation (12 zero-variance columns) were removed because they provide no useful information for distinguishing traffic classes.
   - **Feature and target separation**: The target class is separated from the input features before model training. Features that could leak target information are excluded where specified by the training pipeline.
   - **Feature preprocessing**: Numerical features are imputed and standardized (`StandardScaler`) as configured in the training pipeline. Categorical features, when present, are imputed and one-hot encoded.
4. **Cleaned Dataset**:
   - The recorded cleaned dataset contains **49,573** network traffic records:
   - Benign traffic: 25,709 records (51.86%)
   - Attack traffic: 23,864 records (48.14%)
   - The cleaned data is used for exploratory data analysis, statistical testing, visualization, and deep learning experiments.
5. **Model Evaluation and Reproducibility**:
   - The project contains preprocessing, training, and evaluation scripts under the `src/` directory. The repository also contains the generated visualizations, evaluation results, and project report.
   - To reproduce the workflow, download the original dataset from the official source, place the relevant CSV file in the `data/` directory expected by the project, install the dependencies listed in `requirements.txt`, and follow the execution instructions in `README.md`.
   - *Note: The dataset is not included directly in this repository. The recorded results depend on the dataset version, preprocessing configuration, and model-training procedure used for the experiment.*

---

## 4. Technologies & Tools

- **Programming Language**: Python 3.10+
- **Data Wrangling & Mathematics**: Pandas, NumPy, SciPy (`scipy.stats`)
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning & Preprocessing**: Scikit-learn (`StandardScaler`, `train_test_split`, `LabelEncoder`, `metrics`)
- **Deep Learning**: TensorFlow 2.x, Keras (Multilayer Perceptron, Callbacks, Layers)
- **Documentation & Reporting**: Python-docx (Word Academic Report generation), Jupyter Notebook
- **Testing**: Pytest

---

## 5. Project Directory Structure

```
network-intrusion-eda/
├── .gitignore                      # Git exclusion rules (CSVs, venv, checkpoints)
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
├── main.py                         # End-to-end command line execution script
├── build_notebook.py               # Notebook build utility
├── data/
│   └── README.md                   # Download instructions & daily file mappings
├── notebooks/
│   └── network_intrusion_eda.ipynb # Step-by-step interactive research workflow
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── data_loader.py              # File discovery, loading, and sanity inspection
│   ├── preprocessing.py            # Leakage-free scaling, IQR audit, imputation
│   ├── eda.py                      # 8 publication-grade visualization routines (CO3)
│   ├── statistical_analysis.py     # Non-parametric hypothesis testing & stats (CO4)
│   ├── train_model.py              # Keras MLP classifier, class weights, HP tuning
│   ├── evaluate_model.py           # Test diagnostics, ROC-AUC, PR-AUC, metrics
│   └── generate_academic_report.py # Automated Word (.docx) report generator
├── results/
│   ├── eda/                        # 12 publication-grade EDA figures (.png)
│   ├── model_evaluation/           # 9 diagnostic figures, metrics JSON, model .keras
│   ├── tables/                     # 5 summary CSV tables (descriptive, KS, Mann-Whitney U)
│   └── report/
│       ├── project_report.docx     # Complete 11-section academic assignment report (Word)
│       └── project_report.pdf      # Print-ready compiled PDF report
└── tests/
    ├── __init__.py                 # Test package initializer
    └── test_preprocessing.py       # Pytest unit tests (7/7 passing)
```

---

## 6. Installation & Execution Guide

### Step 1: Clone or Open Repository
Open VS Code in the project root directory:
```powershell
cd d:\Network_Intrusion_EDA
```

### Step 2: Install Required Dependencies
Install the required packages using pip:
```powershell
pip install -r requirements.txt
```

### Step 3: Run Unit Tests
Verify pipeline components using isolated synthetic unit test fixtures:
```powershell
python -m pytest tests/test_preprocessing.py -v
```
*(All 7 unit tests validate cleaning, infinity resolution, leak-free fitting, and MLP model forward pass.)*

### Step 4: Run the Complete Pipeline
Place your downloaded CIC-IDS2017 CSV file into `data/` and execute:
```powershell
# Run with default settings (loads available CSV in data/)
python main.py

# Rapid prototyping with 50,000 rows in binary mode (BENIGN vs ATTACK):
python main.py --file_name "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" --nrows 50000 --binary

# Multi-class full dataset training:
python main.py --epochs 25 --batch_size 256
```

### Step 5: Interactive Analysis in Jupyter Notebook
Launch VS Code's built-in notebook viewer or Jupyter:
```powershell
jupyter notebook notebooks/network_intrusion_eda.ipynb
```

### Step 6: Generate or Update Academic Reports (Word & PDF)
The Word and PDF reports are automatically compiled and exported with 1.0-inch margins, zero running headers/footers, and verified experimental metrics:
```powershell
# 1. Run full class balance, data leakage audit, and imbalance strategy benchmark:
python src/audit_and_compare_strategies.py

# 2. Compile the Word (.docx) document and export the PDF report:
python src/generate_academic_report.py

# 3. Programmatically verify formatting, margins, student details, and PDF integrity:
python src/verify_deliverables.py
```
Output Deliverables:
- Word Document: [`results/report/project_report.docx`](file:///D:/Network_Intrusion_EDA/results/report/project_report.docx) (3.67 MB)
- Formatted PDF: [`results/report/project_report.pdf`](file:///D:/Network_Intrusion_EDA/results/report/project_report.pdf) (1.92 MB, 27 pages)

---

## 7. Statistical & Exploratory Analysis Methodology (CO3 & CO4)

### Data Cleaning & Anti-Leakage Protocol:
- **Duplicate Flows**: Identical network flows recorded by monitoring mirrors (427 rows) are pruned.
- **Infinities & NaNs**: Flow rate features (`Flow Bytes/s`, `Flow Packets/s`) with positive/negative infinity (23 instances) are transformed and imputed using training set medians.
- **Outlier Policy**: Analyzed via Interquartile Range ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$). *Outliers are preserved* because high packet rates and large payload bursts represent genuine attack signatures.
- **Strict Anti-Leakage Separation**: Scalers (`StandardScaler`) and encoders fit **exclusively** on the 35% training split (17,350 flows) and are subsequently applied to validation (35%, 17,351 flows) and held-out test (30%, 14,872 flows) sets.

### Statistical Tests (CO4):
- **Descriptive Descriptors**: Mean, Median, Mode, Variance, Standard Deviation, IQR, Skewness, Kurtosis.
- **Normality Evaluation**: Two-sample Kolmogorov-Smirnov goodness-of-fit test against theoretical Gaussian and Log-Normal models. (Gaussian fit rejected at $p < 0.001$, $D > 0.34$).
  - *Methodological Limitation Note*: Standard KS p-values are conservative when parameters (mean and std) are estimated from the sample data (Lilliefors effect); however, test statistics vastly exceed critical bounds ($D_{crit} \approx 0.023$), conclusively refuting Gaussian normality.
- **Comparative Non-Parametric Testing**: Two-sided **Mann-Whitney U Test** ($n_1 = 5,000, n_2 = 5,000$) comparing continuous distributions of Benign vs Attack flows.
- **Effect Size Metric**: **Rank-Biserial Correlation ($r$)** calculating the practical magnitude of distributional differences (e.g. Forward Packet Length Mean: $r = -0.5234$, large effect).

---

## 8. Deep Learning Model Architecture (CO5)

The neural network is formulated as a deep Multilayer Perceptron (MLP) implemented in TensorFlow/Keras:

```
Input Features (66 Standardized Continuous Flow Metrics)
       │
       ▼
[Dense Layer 1: 128 Neurons, He-Normal, L2 Regularization (1e-4)]
       │
       ▼
[Batch Normalization + ReLU Activation + Dropout (0.30)]
       │
       ▼
[Dense Layer 2: 64 Neurons + Batch Normalization + ReLU + Dropout (0.20)]
       │
       ▼
[Dense Layer 3: 32 Neurons + ReLU Activation + Dropout (0.10)]
       │
       ▼
[Output Layer: 1 Unit with Sigmoid Activation (Binary Classification)]
```

### Training & Regularization Highlights:
- **Balanced Class Weighting**: Computed via $w_j = \frac{N}{K \times n_j}$ to prevent minority attack neglect.
- **Early Stopping**: Restores optimal weights after 5 epochs of validation loss stagnation.
- **Learning Rate Annealing**: Dynamically reduces learning rate by 50% on loss plateaus.
- **Permutation Feature Importance**: Evaluates post-hoc macro F1 drop on held-out test data.

---

## 9. Evaluation Metrics & Imbalance-Handling Comparison

### Audited Class Proportions Across Partitions:
| Partition Stage | Total Flows | BENIGN Flows (%) | ATTACK Flows (%) | Data Quality Notes |
|:---|:---:|:---:|:---:|:---|
| **Raw Dataset Sample** | 50,000 | 26,135 (52.27%) | 23,865 (47.73%) | 427 duplicates, 3 NaNs, 23 $\pm\infty$ |
| **Cleaned Dataset** | 49,573 | 25,709 (51.86%) | 23,864 (48.14%) | 100% duplicates purged, infs imputed |
| **Training Set (35.0%)** | 17,350 | 8,998 (51.86%) | 8,352 (48.14%) | Scalers/imputers fit strictly here |
| **Validation Set (35.0%)** | 17,351 | 8,998 (51.86%) | 8,353 (48.14%) | Model checkpoint & tuning only |
| **Held-Out Test Set (30.0%)** | 14,872 | 7,713 (51.86%) | 7,159 (48.14%) | Quarantined; evaluated once |

### Systematic Comparison of Imbalance Mitigation Strategies:
Evaluated on identical splits (imputers, scalers, and resampling fit strictly on Train only):

| Strategy | Val Accuracy | Val Macro F1 | Test Accuracy | Balanced Acc | Test Macro F1 | ROC-AUC | Attack Recall | Confusion Matrix [TN, FP, FN, TP] | Decision Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline MLP** | 99.86% | 0.9986 | 99.93% | 99.93% | 0.9993 | 0.9997 | 99.97% | [7704, 9, 2, 7157] | High precision, standard loss |
| **Class-Weighted MLP** | **99.86%** | **0.9986** | **99.92%** | **99.92%** | **0.9992** | **0.9997** | **99.97%** | **[7703, 10, 2, 7157]** | **Selected Optimal Model** |
| **SMOTE + MLP** | 99.81% | 0.9981 | 99.88% | 99.88% | 0.9988 | 0.9997 | 99.96% | [7698, 15, 3, 7156] | Synthetic interpolation artifacts |

**Rationale for Retaining Class-Weighted MLP**:
Because the dataset is naturally balanced (~52% BENIGN to 48% ATTACK), synthetic oversampling via SMOTE generates interpolated points along decision boundaries, slightly increasing false negatives (15 vs 10). Class weighting provides balanced gradient backpropagation without distorting the empirical data manifold.

---

## 10. Limitations & Future Scope

### Current Limitations:
- The MLP processes tabular flow summaries produced by `CICFlowMeter` rather than raw packet payload byte sequences.
- Evaluated on single-day DDoS capture (50,000 flows); multi-day composite captures (Slowloris, Infiltration, Web Attacks) require multi-class extensions.
- Evaluated in offline batch mode rather than live streaming in-line network sensors.

### Future Directions:
- Incorporate sequential architectures (LSTM / Temporal Convolutional Networks / Transformers) to model packet inter-arrival time series.
- Implement live streaming intrusion detection integrating Zeek or Suricata network sensors.
- Deploy adversarial robustness training against evasion and payload perturbation attacks.

---

## 11. Instructions for Creating & Pushing to GitHub

To publish this project to your personal GitHub account:

1. **Initialize Git** in the project directory:
   ```powershell
   git init
   ```
2. **Add Files to Staging**:
   ```powershell
   git add .
   ```
3. **Commit Your Code**:
   ```powershell
   git commit -m "feat: complete academic project for network intrusion EDA, statistical analysis, and MLP deep learning"
   ```
4. **Create a New Repository on GitHub**:
   - Log into [GitHub](https://github.com).
   - Name it: `network-intrusion-eda`.
5. **Link and Push**:
   ```powershell
   git branch -M main
   git remote add origin https://github.com/<your-username>/network-intrusion-eda.git
   git push -u origin main
   ```

---

## 12. Academic Attribution & Candidate Information

- **Student Name**: Farhath Suraiya S D
- **Roll Number / Register Number**: 24AD035
- **Department**: Artificial Intelligence and Data Science
- **Degree & Branch**: B.Tech. Artificial Intelligence and Data Science
- **Academic Year**: 2026–2027
- **Course Code & Title**: U21ADP05 – Exploratory Data Analysis and Visualization (Assignment II)
- **Institution**: KPR Institute of Engineering and Technology
- **Date of Submission**: 15.10.2026
- **References**: IEEE format citations included in Section 10 of the generated academic report.


