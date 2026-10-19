\# Data Collection and Preprocessing



\## 1. Dataset Source



This project uses the CIC-IDS2017 dataset, which contains network traffic records representing benign activities and different cyberattacks.



\*\*Official dataset:\*\* https://www.unb.ca/cic/datasets/ids-2017.html



The experiment analyzed the `Friday-WorkingHours-Afternoon-DDos.pcap\_ISCX.csv` file.



\## 2. Raw Dataset



\- Raw records analyzed: 50,000

\- Benign traffic: 26,135 records

\- Attack traffic: 23,865 records

\- Data type: Numerical network traffic features and a target class label



\## 3. Data Cleaning



The preprocessing workflow addresses the following data-quality issues:



1\. \*\*Duplicate records:\*\* Duplicate rows were identified and removed. The recorded cleaning summary reports 427 duplicates removed.

2\. \*\*Infinite values:\*\* 23 infinite values were identified and resolved.

3\. \*\*Missing values:\*\* Three missing values were handled during cleaning. Missing-value checks and imputation are also part of the preprocessing workflow.

4\. \*\*Constant features:\*\* Features with no variation were removed because they provide no useful information for distinguishing traffic classes.

5\. \*\*Feature and target separation:\*\* The target class is separated from the input features before model training. Features that could leak target information are excluded where specified by the training pipeline.

6\. \*\*Feature preprocessing:\*\* Numerical features are imputed and standardized as configured in the training pipeline. Categorical features, when present, are imputed and one-hot encoded.



\## 4. Cleaned Dataset



The recorded cleaned dataset contains 49,573 network traffic records.



\- Benign traffic: 25,709 records

\- Attack traffic: 23,864 records



The cleaned data is used for exploratory data analysis, statistical testing, visualization, and deep learning experiments.



\## 5. Model Evaluation and Reproducibility



The project contains preprocessing, training, and evaluation scripts under the `src/` directory. The repository also contains the generated visualizations, evaluation results, and project report.



To reproduce the workflow, download the original dataset from the official source, place the relevant CSV file in the data directory expected by the project, install the dependencies listed in `requirements.txt`, and follow the execution instructions in `README.md`.



\*\*Note:\*\* The dataset is not included directly in this repository. The recorded results depend on the dataset version, preprocessing configuration, and model-training procedure used for the experiment.

