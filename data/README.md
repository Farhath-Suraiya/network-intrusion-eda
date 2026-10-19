# CIC-IDS2017 Dataset Directory

This directory is designated for storing the official **CIC-IDS2017 (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset)** CSV files.

---

## 1. About the Dataset
- **Provider**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)
- **Official URL**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Kaggle Mirror**: [CIC-IDS2017 on Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017)
- **Format**: Comma-Separated Values (CSV), generated from 5 days of captured network traffic using `CICFlowMeter`.
- **Total Features**: ~79 network flow features (Flow Duration, Total Fwd Packets, Flow Bytes/s, Packet Length stats, TCP Flags, etc.) plus a target `Label` column.

---

## 2. Available Daily Flow Files

The complete dataset consists of 8 CSV files corresponding to network traffic captured from Monday, July 3, 2017, to Friday, July 7, 2017:

1. `Monday-WorkingHours.pcap_ISCX.csv`  
   *Contents*: Normal benign network traffic.
2. `Tuesday-WorkingHours.pcap_ISCX.csv`  
   *Attacks*: FTP-Patator, SSH-Patator (Brute Force).
3. `Wednesday-workingHours.pcap_ISCX.csv`  
   *Attacks*: DoS GoldenEye, DoS Slowloris, DoS Slowhttptest, DoS Hulk, Heartbleed.
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`  
   *Attacks*: Web Attack – Brute Force, Web Attack – XSS, Web Attack – Sql Injection.
5. `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`  
   *Attacks*: Infiltration.
6. `Friday-WorkingHours-Morning.pcap_ISCX.csv`  
   *Attacks*: Botnet ARES.
7. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`  
   *Attacks*: PortScan.
8. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`  
   *Attacks*: DDoS LOIC.

---

## 3. How to Download and Setup

### Option A: From Official UNB Portal
1. Navigate to [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Request download access to "GeneratedLabelledFlows.zip" or direct CSV files.
3. Extract one or more CSV files (e.g., `Wednesday-workingHours.pcap_ISCX.csv` or `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) into this `data/` folder.

### Option B: From Kaggle (Fastest)
1. Visit [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/cicdataset/cicids2017).
2. Download any single daily CSV (recommended for quick evaluation: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` or `Wednesday-workingHours.pcap_ISCX.csv`).
3. Place the downloaded `.csv` file directly inside this `data/` folder.

---

## 4. Data Collection and Preprocessing Summary

### 1. Dataset Source
This project uses the CIC-IDS2017 dataset, which contains network traffic records representing benign activities and different cyberattacks.
- **Official Dataset**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Analyzed File**: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

### 2. Raw Dataset
- **Raw Records Analyzed**: 50,000
- **Benign Traffic**: 26,135 records (52.27%)
- **Attack Traffic**: 23,865 records (47.73%)
- **Data Type**: Numerical network traffic features and a target class label (`Binary_Label`: `BENIGN` vs `ATTACK`)

### 3. Data Cleaning
The preprocessing workflow addresses the following data-quality issues:
- **Duplicate Records**: Duplicate rows were identified and removed. The recorded cleaning summary reports 427 duplicates removed.
- **Infinite Values**: 23 infinite values (`+inf` in `Flow Bytes/s` and `Flow Packets/s`) were identified and resolved.
- **Missing Values**: Three missing values were handled during cleaning. Missing-value checks and median imputation are part of the preprocessing workflow.
- **Constant Features**: Features with no variation (12 zero-variance columns) were removed because they provide no useful information for distinguishing traffic classes.
- **Feature and Target Separation**: The target class is separated from the input features before model training. Features that could leak target information (metadata and identifiers) are excluded where specified by the training pipeline.
- **Feature Preprocessing**: Numerical features are imputed and standardized (`StandardScaler`) as configured in the training pipeline, fitted strictly on the training partition to prevent leakage. Categorical features, when present, are imputed and one-hot encoded.

### 4. Cleaned Dataset
The recorded cleaned dataset contains **49,573** network traffic records:
- **Benign Traffic**: 25,709 records (51.86%)
- **Attack Traffic**: 23,864 records (48.14%)

The cleaned data is used for exploratory data analysis, statistical testing, visualization, and deep learning experiments.

### 5. Model Evaluation and Reproducibility
The project contains preprocessing, training, and evaluation scripts under the `src/` directory. The repository also contains the generated visualizations, evaluation results, and project report.

To reproduce the workflow:
1. Download the original dataset from the official source ([UNB Portal](https://www.unb.ca/cic/datasets/ids-2017.html) or [Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017)).
2. Place the relevant CSV file (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) in this `data/` directory.
3. Install the dependencies listed in `requirements.txt`.
4. Follow the execution instructions in `README.md`.

*Note: The dataset is not included directly in this repository. The recorded results depend on the dataset version, preprocessing configuration, and model-training procedure used for the experiment.*

