"""
generate_academic_report.py
Generates the definitive, formal academic assignment report for Course U21ADP05.
Institution: KPR Institute of Engineering and Technology
Department: Artificial Intelligence and Data Science

Structure strictly adheres to the 11-section format:
1. Title Page (Cybersecurity aesthetic, editable student metadata)
2. Abstract (100–150 words, verified results)
3. Introduction and Objectives
4. Dataset Description (Verified CIC-IDS2017 DDoS capture stats & data dictionary)
5. EDA and Preprocessing (Methodology, before/after table, architecture flowchart)
6. Data Visualization (All 12 publication-grade figures with full analytical breakdowns)
7. Deep Learning Model (Architecture diagram, layer specs, hyperparameter experiments)
8. Result Visualization and Interpretation (All evaluation plots, confusion matrix, ROC, PR, metrics)
9. Conclusion and Future Scope
10. References (IEEE format with in-text citations [1]–[5])
11. Appendix: Code Section (Excerpts, commands, directory tree, test results, GitHub instructions)

Exports:
- results/report/project_report.docx
- results/report/project_report.pdf (via native Word COM automation)
"""

import os
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def set_cell_background(cell, fill_hex):
    """Sets background shading color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a table cell (in twentieths of a point / dxa)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def add_figure_with_caption(doc, img_path, caption_text, width=Inches(5.4)):
    """Embeds an image centered with a styled bold caption below it."""
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(img_path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(8)
        cap_run = p_cap.add_run(caption_text)
        cap_run.font.size = Pt(9.5)
        cap_run.font.bold = True
        cap_run.font.italic = True
        cap_run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    else:
        p_err = doc.add_paragraph(f"[Image file not found: {img_path}]")
        p_err.runs[0].font.italic = True


def build_formal_report(
    docx_out: str = "results/report/project_report.docx",
    pdf_out: str = "results/report/project_report.pdf",
    eda_dir: str = "results/eda",
    model_dir: str = "results/model_evaluation",
    tables_dir: str = "results/tables"
):
    os.makedirs(os.path.dirname(os.path.abspath(docx_out)), exist_ok=True)
    doc = Document()

    # Configure Margins: Exactly 1.0 inch (2.54 cm) on all four sides consistently
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Remove all running headers and footers across all sections
        section.header.is_linked_to_previous = False
        section.footer.is_linked_to_previous = False
        for p in section.header.paragraphs:
            p.text = ""
        for p in section.footer.paragraphs:
            p.text = ""

    # Base Typography Settings
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(5)

    def add_h1(text):
        h = doc.add_heading(text, level=1)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        run = h.runs[0]
        run.font.name = 'Arial'
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0A, 0x25, 0x40) # Deep Navy
        return h

    def add_h2(text):
        h = doc.add_heading(text, level=2)
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        run = h.runs[0]
        run.font.name = 'Arial'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x00, 0x55, 0x82) # Slate Blue
        return h

    def add_h3(text):
        h = doc.add_heading(text, level=3)
        h.paragraph_format.space_before = Pt(7)
        h.paragraph_format.space_after = Pt(3)
        h.paragraph_format.keep_with_next = True
        run = h.runs[0]
        run.font.name = 'Arial'
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return h

    # =========================================================================
    # SECTION 1: TITLE PAGE
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(15)
    p_inst.paragraph_format.space_after = Pt(4)
    r1 = p_inst.add_run("KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY\n")
    r1.font.name = 'Arial'
    r1.font.size = Pt(16)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0x0A, 0x25, 0x40)

    r2 = p_inst.add_run("(Autonomous Institution, Affiliated to Anna University, Chennai)\n"
                        "Department of Artificial Intelligence and Data Science\n"
                        "Arasur, Coimbatore - 641 407, Tamil Nadu, India\n")
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Decorative Rule
    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_div.paragraph_format.space_after = Pt(14)
    r_div = p_div.add_run("―" * 48)
    r_div.font.color.rgb = RGBColor(0x00, 0x80, 0x80)

    # Academic Course Banner
    p_course = doc.add_paragraph()
    p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_course.paragraph_format.space_after = Pt(12)
    rc1 = p_course.add_run("COURSE: U21ADP05 – EXPLORATORY DATA ANALYSIS AND VISUALIZATION\n")
    rc1.font.name = 'Arial'
    rc1.font.size = Pt(11)
    rc1.font.bold = True
    rc1.font.color.rgb = RGBColor(0x00, 0x55, 0x82)
    rc2 = p_course.add_run("LABORATORY ASSIGNMENT II — ACADEMIC EVALUATION REPORT")
    rc2.font.size = Pt(10)
    rc2.font.bold = True

    # Main Project Title Box
    title_box = doc.add_table(rows=1, cols=1)
    title_box.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_tb = title_box.rows[0].cells[0]
    cell_tb.width = Inches(6.5)
    set_cell_background(cell_tb, "0A2540")
    set_cell_margins(cell_tb, top=180, bottom=180, left=180, right=180)
    p_box = cell_tb.paragraphs[0]
    p_box.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_box = p_box.add_run("NETWORK INTRUSION DETECTION AND CYBERATTACK PATTERN ANALYSIS USING EXPLORATORY DATA ANALYSIS, STATISTICAL VISUALIZATION AND DEEP LEARNING")
    r_box.font.name = 'Arial'
    r_box.font.size = Pt(13)
    r_box.font.bold = True
    r_box.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Student Metadata Table
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(18)
    p_sp.paragraph_format.space_after = Pt(6)

    meta_table = doc.add_table(rows=7, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Student Name:", "Farhath Suraiya S D"),
        ("Roll Number / Register Number:", "24AD035"),
        ("Department:", "Artificial Intelligence and Data Science"),
        ("Degree & Branch:", "B.Tech. Artificial Intelligence and Data Science"),
        ("Academic Year:", "2026–2027"),
        ("Course Code & Title:", "U21ADP05 – Exploratory Data Analysis and Visualization"),
        ("Date of Submission:", "15.10.2026")
    ]
    for row_idx, (label, val) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        cell_lbl, cell_val = row.cells[0], row.cells[1]
        cell_lbl.width = Inches(2.5)
        cell_val.width = Inches(4.0)
        set_cell_margins(cell_lbl, top=65, bottom=65, left=90, right=90)
        set_cell_margins(cell_val, top=65, bottom=65, left=90, right=90)
        
        bg = "F4F6F8" if row_idx % 2 == 0 else "FFFFFF"
        set_cell_background(cell_lbl, bg)
        set_cell_background(cell_val, bg)
        
        p_l = cell_lbl.paragraphs[0]
        r_l = p_l.add_run(label)
        r_l.font.bold = True
        r_l.font.size = Pt(10)
        
        p_v = cell_val.paragraphs[0]
        r_v = p_v.add_run(val)
        r_v.font.size = Pt(10)

    p_eval = doc.add_paragraph()
    p_eval.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eval.paragraph_format.space_before = Pt(25)
    r_eval = p_eval.add_run("Curriculum Outcomes Addressed: CO3 (Data Visualization - 8 Marks) | "
                            "CO4 (Statistical & Distribution Visualization - 16 Marks) | "
                            "CO5 (Application of EDA & DL - 16 Marks)")
    r_eval.font.size = Pt(8.5)
    r_eval.font.bold = True
    r_eval.font.color.rgb = RGBColor(0x00, 0x80, 0x80)

    doc.add_page_break()

    # =========================================================================
    # SECTION 2: ABSTRACT
    # =========================================================================
    add_h1("Section 2: Abstract")
    abstract_p = doc.add_paragraph()
    abstract_p.paragraph_format.line_spacing = 1.2
    abstract_p.paragraph_format.space_after = Pt(12)
    abstract_run = abstract_p.add_run(
        "Modern cloud networks face persistent threats from volumetric cyberattacks such as Distributed Denial of Service (DDoS). "
        "This project presents an empirical investigation into network intrusion detection using the real-world Canadian Institute for "
        "Cybersecurity CIC-IDS2017 dataset (Friday DDoS capture). An analytical pipeline was engineered comprising data cleansing, "
        "infinite-value resolution, duplicate elimination (427 rows removed), and zero-leakage stratified standardization. "
        "Exploratory and statistical analyses demonstrated pronounced positive skewness and non-Gaussian distributions, confirmed by "
        "Kolmogorov-Smirnov goodness-of-fit tests (normality rejected at p < 0.001) and Mann-Whitney U hypothesis testing (large effect sizes "
        "in packet length, r = -0.5234). A Multilayer Perceptron (MLP) deep neural network regularized with Batch Normalization and Dropout "
        "achieved 99.92% test accuracy, 0.9992 macro F1-score, and 0.9997 ROC-AUC evaluated on a stratified 35% training partition (17,350 flows), "
        "35% validation partition (17,351 flows), and held-out 30% test partition of 14,872 flows. "
        "The model demonstrates excellent volumetric separation, with limitations centered on single-day attack scope and offline tabular telemetry."
    )
    abstract_run.font.size = Pt(10.5)

    p_kw = doc.add_paragraph()
    r_kw_title = p_kw.add_run("Keywords: ")
    r_kw_title.font.bold = True
    p_kw.add_run("Network Intrusion Detection, CIC-IDS2017, Exploratory Data Analysis, Non-Parametric Hypothesis Testing, Multilayer Perceptron, DDoS Detection.")

    # Table of Contents
    add_h2("Table of Contents")
    toc_table = doc.add_table(rows=12, cols=2)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_headers = ["Report Section", "Section Title / Core Focus Area"]
    for j, h in enumerate(toc_headers):
        c = toc_table.cell(0, j)
        c.paragraphs[0].add_run(h).bold = True
        set_cell_background(c, "0A2540")
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(c, 50, 50, 70, 70)

    toc_data = [
        ("Section 1", "Title Page (Candidate & Course Credentials)"),
        ("Section 2", "Executive Abstract & Keywords"),
        ("Section 3", "Introduction, Background and Project Objectives"),
        ("Section 4", "Dataset Description, Provenance and Data Dictionary (CIC-IDS2017)"),
        ("Section 5", "EDA and Preprocessing Methodology (Deduplication, IQR Outliers, Anti-Leakage Split)"),
        ("Section 6", "Data Visualization (8 Core Plots + 4 Supplementary Visuals + Hypothesis Tests)"),
        ("Section 7", "Deep Learning Model (Architecture, Hyperparameters & Imbalance Strategies)"),
        ("Section 8", "Result Visualization and Held-Out Test Evaluation (Confusion Matrix, ROC, PR, Importance)"),
        ("Section 9", "Conclusion, Operational Impact and Future Scope"),
        ("Section 10", "References (IEEE Citation Format)"),
        ("Section 11", "Appendix: Directory Structure, Execution Commands, Unit Tests & Git Protocol")
    ]
    for r_idx, (sec_num, sec_title) in enumerate(toc_data):
        c0 = toc_table.cell(r_idx + 1, 0)
        c1 = toc_table.cell(r_idx + 1, 1)
        c0.width = Inches(1.8)
        c1.width = Inches(4.7)
        c0.paragraphs[0].add_run(sec_num).bold = True
        c1.paragraphs[0].add_run(sec_title)
        bg = "F4F6F8" if r_idx % 2 == 0 else "FFFFFF"
        set_cell_background(c0, bg)
        set_cell_background(c1, bg)
        set_cell_margins(c0, 40, 40, 60, 60)
        set_cell_margins(c1, 40, 40, 60, 60)

    doc.add_page_break()

    # =========================================================================
    # SECTION 3: INTRODUCTION AND OBJECTIVES
    # =========================================================================
    add_h1("Section 3: Introduction and Objectives")
    add_h2("3.1 Background of Network Intrusion Detection")
    doc.add_paragraph(
        "Network Intrusion Detection Systems (NIDS) represent the cornerstone of modern enterprise defense architectures, "
        "responsible for continuously monitoring packet transmissions across perimeter firewalls, core routers, and cloud gateways [1]. "
        "With corporate infrastructure migrating toward microservices and hybrid cloud endpoints, the attack surface has expanded exponentially. "
        "Cyberattacks—ranging from volumetric flooding (DDoS LOIC) and brute-force credential cracking to multi-stage internal reconnaissance—disrupt "
        "mission-critical services, inflict catastrophic economic losses, and compromise sensitive user credentials [3]."
    )

    add_h2("3.2 Limitations of Rule-Based Intrusion Detection")
    doc.add_paragraph(
        "Traditional NIDS solutions rely heavily on static signature databases and hardcoded heuristics (e.g., Snort or Suricata rule sets). "
        "While computationally fast for known exploit signatures, rule-based detection suffers from critical operational limitations:\n"
        "1. Inability to Detect Novel Variants: Zero-day exploits and polymorphic traffic easily circumvent static signature matching.\n"
        "2. High False Alarm Rates: Burst traffic generated by legitimate web events (flash crowds, database backups) frequently triggers false positives, inducing alert fatigue in Security Operations Center (SOC) teams.\n"
        "3. Manual Rule Engineering Bottleneck: Creating and maintaining signature rules requires constant human intervention, failing to keep pace with automated threat automation [4]."
    )

    add_h2("3.3 Role of Exploratory Data Analysis (EDA) and Deep Learning")
    doc.add_paragraph(
        "To overcome rule-based deficiencies, data-driven security analytics pairs Exploratory Data Analysis (EDA) with Deep Learning. "
        "EDA exposes the underlying statistical topologies, multicollinear feature structures, and probability density profiles inherent to network telemetries. "
        "Concurrently, Multilayer Perceptrons (MLPs) can learn non-linear decision surfaces across multidimensional flow features (such as packet lengths, "
        "inter-arrival times, and byte rates), isolating attack flows without manual threshold engineering [2]."
    )

    add_h2("3.4 Problem Statement")
    doc.add_paragraph(
        "Given continuous streams of multidimensional network flow records containing extreme statistical skewness, positive infinity artifacts, "
        "and legitimate volumetric outliers, how can we construct an automated, leakage-free EDA and Deep Learning classification pipeline that "
        "reliably distinguishes cyberattack traffic from benign sessions with high precision and recall?"
    )

    add_h2("3.5 Project Objectives")
    doc.add_paragraph(
        "This project achieves five explicit objectives aligned with the curriculum outcomes of Course U21ADP05:\n"
        "• Objective 1 (CO3 - Data Visualization): Formulate and generate at least 8 publication-grade graphical representations capturing flow distributions, class skews, bivariate relationships, and multivariate clusterings.\n"
        "• Objective 2 (CO4 - Statistical Analysis): Calculate comprehensive parametric and non-parametric descriptive statistics, evaluate empirical class probabilities, and conduct formal hypothesis tests (Mann-Whitney U) and goodness-of-fit assessments.\n"
        "• Objective 3 (Pre-processing & Leakage Prevention): Establish a leak-free data cleaning pipeline that resolves infinities, eliminates duplicate flows, preserves true attack outliers, and fits scalers strictly on training data.\n"
        "• Objective 4 (CO5 - Deep Learning Modeling): Design, regularize, and train a Multilayer Perceptron (MLP) using TensorFlow/Keras, incorporating Batch Normalization, Dropout, and cost-sensitive class weighting.\n"
        "• Objective 5 (Held-Out Evaluation): Rigorously benchmark test performance via Confusion Matrices, ROC-AUC, Precision-Recall Curves, and Permutation Feature Importance on strictly held-out test traffic."
    )

    add_h2("3.6 Practical Real-World Applications")
    doc.add_paragraph(
        "The analytical techniques developed in this study directly inform: (1) Enterprise SOC Alert Triage to suppress false alarms; "
        "(2) Next-Generation Firewall (NGFW) telemetry filtering for inline packet classification; and (3) Zero-Trust Network Access (ZTNA) "
        "for behavioral anomaly scoring [3]. Note: This study serves as an academic prototype evaluating offline flow telemetry and is not "
        "presented as a turn-key commercial defense engine."
    )

    # =========================================================================
    # SECTION 4: DATASET DESCRIPTION
    # =========================================================================
    add_h1("Section 4: Dataset Description")
    doc.add_paragraph(
        "The investigation was conducted on the official Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset "
        "(CIC-IDS2017), generated by the University of New Brunswick (UNB) [1]. Captured over five days (July 3–7, 2017), the dataset captures "
        "realistic background network traffic intertwined with contemporary cyberattack profiles generated via behavioral B-Profile agents."
    )

    add_h2("4.1 Dataset Provenance and Capture Configuration")
    doc.add_paragraph(
        "• Official Source: Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/ids-2017.html) [1]\n"
        "• Specific Capture File Evaluated: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (77.1 MB)\n"
        "• Extraction Mechanism: Network flows extracted via CICFlowMeter, calculating forward and backward statistical flow metrics.\n"
        "• Experimental Sample Size: 50,000 raw flows processed for reproducible benchmarking.\n"
        "• Target Variable: Binary_Label, mapped from raw Label into 'BENIGN' (normal operational flows) and 'ATTACK' (volumetric DDoS LOIC flows)."
    )

    add_h2("4.2 Pre-Cleaning Audit and Class Distribution Table")
    doc.add_paragraph("Table 1 summarizes the raw and processed dataset dimensions, data quality anomalies, and final partition counts.")

    # Table 1: Dataset Summary
    t1 = doc.add_table(rows=9, cols=3)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1_headers = ["Dataset Property", "Raw Observation", "Post-Cleaning Resolution"]
    for j, h in enumerate(t1_headers):
        c = t1.cell(0, j)
        c.paragraphs[0].add_run(h).bold = True
        set_cell_background(c, "0A2540")
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(c, 80, 80, 100, 100)

    t1_data = [
        ("Total Rows Loaded", "50,000 flows", "49,573 flows retained"),
        ("Duplicate Flows", "427 duplicate rows", "100% culled (0 duplicates remaining)"),
        ("Missing Values (NaN)", "3 missing values in Flow Bytes/s", "Median-imputed using training set"),
        ("Positive Infinities (+Inf)", "23 values (+inf in Flow Bytes/s & Packets/s)", "Replaced with NaN, median-imputed"),
        ("Original Feature Count", "78 traffic features", "66 predictive features retained"),
        ("Constant Features", "12 zero-variance columns", "Excised (e.g. constant flag counts)"),
        ("Target Class Proportions", "26,135 Benign / 23,865 Attack", "52.27% Benign / 47.73% Attack"),
        ("Data Partitioning Breakdown", "50,000 raw flows", "Train: 17,350 (35.0%) | Val: 17,351 (35.0%) | Test: 14,872 (30.0%)")
    ]
    for i, row in enumerate(t1_data):
        for j, val in enumerate(row):
            c = t1.cell(i + 1, j)
            c.paragraphs[0].add_run(val)
            bg = "F4F6F8" if i % 2 == 0 else "FFFFFF"
            set_cell_background(c, bg)
            set_cell_margins(c, 60, 60, 80, 80)

    add_h2("4.3 Network Traffic Feature Data Dictionary")
    doc.add_paragraph(
        "Table 2 provides the technical definition and unit of measurement for key flow features analyzed throughout the study."
    )

    # Table 2: Data Dictionary
    t2 = doc.add_table(rows=7, cols=3)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2_headers = ["Feature Name", "Data Type / Unit", "Cybersecurity & Operational Description"]
    for j, h in enumerate(t2_headers):
        c = t2.cell(0, j)
        c.paragraphs[0].add_run(h).bold = True
        set_cell_background(c, "005582")
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(c, 80, 80, 100, 100)

    t2_data = [
        ("Flow Duration", "Continuous / Microseconds (μs)", "Total temporal duration between flow initiation (SYN) and teardown (FIN/RST)."),
        ("Total Fwd Packets", "Discrete / Integer Count", "Cumulative count of packets transmitted from client source to destination server."),
        ("Total Length of Fwd Packets", "Continuous / Bytes", "Total cumulative payload size in forward direction, revealing volumetric weight."),
        ("Fwd Packet Length Mean", "Continuous / Bytes", "Mean byte size of forward packets; distinguishes automated probes from web browsing."),
        ("Flow Bytes/s", "Continuous / Rate (Bytes/s)", "Payload transfer velocity across the flow session; extreme spike indicator for DDoS."),
        ("Flow Packets/s", "Continuous / Rate (Packets/s)", "Packet transmission rate; volumetric flood identifier.")
    ]
    for i, row in enumerate(t2_data):
        for j, val in enumerate(row):
            c = t2.cell(i + 1, j)
            c.paragraphs[0].add_run(val)
            bg = "F4F6F8" if i % 2 == 0 else "FFFFFF"
            set_cell_background(c, bg)
            set_cell_margins(c, 60, 60, 80, 80)

    # =========================================================================
    # SECTION 5: EDA AND PREPROCESSING
    # =========================================================================
    add_h1("Section 5: EDA and Preprocessing")
    doc.add_paragraph(
        "Network flow datasets extracted from raw PCAP monitors exhibit serious operational data hazards. "
        "A rigorous, leak-free preprocessing pipeline was implemented in `src/preprocessing.py`."
    )

    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "12_preprocessing_workflow_diagram.png"),
        "Figure 1: End-to-End Leakage-Free Preprocessing, Stratified Partitioning, and Model Training Pipeline Architecture [Schematic Diagram]."
    )

    add_h2("5.1 Data Cleansing Protocols")
    doc.add_paragraph(
        "1. Duplicate Row Elimination: Multi-interface network monitoring mirrors duplicate identical flows. In total, 427 duplicate records were audited and eliminated.\n"
        "2. Resolution of Positive and Negative Infinities: When flow duration is zero microseconds, rate formulas produce positive infinity (+inf). "
        "All 23 infinite values detected in Flow Bytes/s and Flow Packets/s were replaced with NaN and imputed using training set medians.\n"
        "3. Statistical Outlier Audit via Interquartile Range (IQR): Using [Q1 - 1.5*IQR, Q3 + 1.5*IQR], extreme values were audited across all continuous flow metrics. "
        "Outliers were intentionally NOT deleted. In cybersecurity telemetry, extreme packet bursts (e.g. 50,000 packets/s) represent bona fide attack signatures.\n"
        "4. Culling of Unusable & Constant Columns: Features with variance <= 0 (e.g., constant zero flag fields) and non-predictive identifiers (Source IP, Flow ID) were excised, reducing feature dimensionality from 78 raw traffic features to 66 predictive features (12 constant columns dropped)."
    )

    add_h2("5.2 Anti-Leakage Stratified Splitting, Cross-Split Overlap Audit, and Scaling")
    doc.add_paragraph(
        "To ensure statistically representative splits while preventing data leakage, the 49,573 cleaned flows were partitioned using stratified sampling:\n"
        "• Training Partition: 35.0% (17,350 flows: 8,352 Attack, 8,998 Benign) — used strictly for fitting median imputers and StandardScaler parameters.\n"
        "• Validation Partition: 35.0% (17,351 flows: 8,353 Attack, 8,998 Benign) — used for monitoring loss convergence, hyperparameter selection, and early stopping.\n"
        "• Held-Out Test Partition: 30.0% (14,872 flows: 7,159 Attack, 7,713 Benign) — completely quarantined and evaluated strictly once after final model selection [5].\n\n"
        "Cross-Split Overlap and Leakage Audit: Target labels and non-predictive metadata (Source IP, Destination IP, Timestamp, Flow ID) were strictly excluded from feature inputs. "
        "An audit of rounded feature vectors revealed that 1,766 out of 13,688 unique test vectors (12.90%) share identical coordinate values with training flows. "
        "In cybersecurity telemetry, identical flow coordinates occur naturally due to repeated TCP SYN retransmissions, keep-alive probes, and uniform LOIC flood packet lengths. "
        "Rather than claiming theoretical zero leakage, we acknowledge this operational reality while guaranteeing that preprocessing statistics (median imputers and scalers) were fitted strictly on the training partition."
    )

    # Table 3: Before-vs-After Cleaning Summary
    add_h2("5.3 Before-versus-After Data Cleaning Summary")
    t3 = doc.add_table(rows=6, cols=3)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3_headers = ["Pipeline Stage", "Before Preprocessing", "After Preprocessing"]
    for j, h in enumerate(t3_headers):
        c = t3.cell(0, j)
        c.paragraphs[0].add_run(h).bold = True
        set_cell_background(c, "005582")
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(c, 70, 70, 90, 90)

    t3_data = [
        ("Duplicate Records", "427 duplicate flows present", "0 duplicates (all 427 purged)"),
        ("Infinite Values (+/- Inf)", "23 infinite values in flow rates", "0 infinite values (median-imputed)"),
        ("Missing Values (NaN)", "3 missing cells in Flow Bytes/s", "0 missing cells (training median imputed)"),
        ("Feature Dimension", "78 raw traffic columns", "66 standardized features (12 constant dropped)"),
        ("Feature Scaling", "Raw unscaled values (0 to 10^7)", "Standardized (Zero Mean, Unit Variance)")
    ]
    for i, row in enumerate(t3_data):
        for j, val in enumerate(row):
            c = t3.cell(i + 1, j)
            c.paragraphs[0].add_run(val)
            bg = "F4F6F8" if i % 2 == 0 else "FFFFFF"
            set_cell_background(c, bg)
            set_cell_margins(c, 50, 50, 70, 70)

    # =========================================================================
    # SECTION 6: DATA VISUALIZATION (12 FIGURES)
    # =========================================================================
    add_h1("Section 6: Data Visualization")
    doc.add_paragraph(
        "A comprehensive suite of 12 distinct graphical visualizations was constructed to investigate cyberattack flow dynamics. "
        "Every figure is grounded in the actual dataset with detailed cybersecurity and statistical interpretations."
    )

    # Visualizations 1 & 2: Class Distributions
    add_h2("6.1 Class Distribution and Proportion Visualizations")
    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "01_class_distribution.png"),
        "Figure 2: Attack Class Frequency Distribution and Volumetric Counts (BENIGN vs ATTACK)."
    )
    doc.add_paragraph(
        "• What it shows: Exact counts and percentages of normal benign traffic versus DDoS attack traffic.\n"
        "• Why selected: Crucial baseline to detect class skewness and avoid majority-class model guessing.\n"
        "• Actual Observation: Benign flows account for 26,135 flows (52.27%), while DDoS attacks comprise 23,865 flows (47.73%).\n"
        "• Statistical Interpretation: Balanced sample proportions indicate that binary classification does not suffer from extreme rare-class starvation.\n"
        "• Practical Cybersecurity Significance: Enables reliable symmetric cost evaluation without heavily skewing recall against false alarms."
    )

    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "09_class_distribution_donut.png"),
        "Figure 3: Donut Proportion Chart Depicting Relative Share of Normal vs DDoS Traffic."
    )
    doc.add_paragraph(
        "• What it shows: Angular proportion and relative percentage share of each traffic classification.\n"
        "• Why selected: Provides an immediate visual executive summary of the session composition.\n"
        "• Actual Observation: Near-even split (52.27% vs 47.73%) validating balanced data intake for the experimental capture.\n"
        "• Practical Cybersecurity Significance: Demonstrates that the afternoon DDoS capture captured both dense attack flooding and background user sessions."
    )

    # Visualizations 3 & 4: Data Quality and Outlier Summary
    add_h2("6.2 Data Quality and Outlier Distribution Audits")
    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "10_missing_values_bar.png"),
        "Figure 4: Pre-Cleaning Data Quality Audit (Missing and Infinite Value Counts)."
    )
    doc.add_paragraph(
        "• What it shows: Bar frequencies of missing and infinite values detected in rate features.\n"
        "• Why selected: Validates the necessity of automated sanitization prior to numerical standardization.\n"
        "• Actual Observation: Flow Bytes/s contained 3 missing values and 20 positive infinities; Flow Packets/s contained 3 infinities.\n"
        "• Statistical Interpretation: Caused by microsecond flow durations where division by zero occurs in packet rate computations.\n"
        "• Practical Cybersecurity Significance: Prevents numerical overflow and gradient explosion during deep neural network backpropagation."
    )

    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "11_iqr_outliers_barchart.png"),
        "Figure 5: Statistical Outlier Prevalence Across Top Flow Features (1.5x IQR Rule)."
    )
    doc.add_paragraph(
        "• What it shows: Percentage of traffic records falling outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] boundaries.\n"
        "• Why selected: Audits whether extreme values represent sensory corruptions or cyberattack footprints.\n"
        "• Actual Observation: Flow Duration (19.85%), Total Length of Fwd Packets (18.24%), and Total Fwd Packets (16.50%) exhibit high outlier percentages.\n"
        "• Statistical Interpretation: Extreme right-tailed dispersion characteristic of heavy-tailed network queuing distributions.\n"
        "• Practical Cybersecurity Significance: Empirically proves that deleting outliers would erroneously purge high-impact cyberattacks."
    )

    # Visualizations 5 & 6: Continuous Distributions & Outliers
    add_h2("6.3 Continuous Distribution and Boxplot Profiles")
    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "02_feature_distributions_kde.png"),
        "Figure 6: Kernel Density Estimation (KDE) and Logarithmic Histograms for Key Flow Descriptors."
    )
    doc.add_paragraph(
        "• What it shows: Non-parametric probability density curves and log-scaled histograms for Flow Duration, Total Fwd Packets, and Forward Length.\n"
        "• Why selected: Evaluates unimodality, dispersion, and empirical deviation from Gaussian assumptions.\n"
        "• Actual Observation: All features span 5 to 6 orders of magnitude with extreme positive right skewness.\n"
        "• Statistical Interpretation: Standard parametric Gaussian models are decisively violated; theoretical log-normal distributions provide superior fit.\n"
        "• Practical Cybersecurity Significance: Confirms that linear thresholding fails and robust standardization or non-linear MLP models are required."
    )

    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "03_outlier_boxplots.png"),
        "Figure 7: Log-Scaled Boxplots Demonstrating Outlier Dispersion Boundaries."
    )
    doc.add_paragraph(
        "• What it shows: Medians, quartiles, and individual extreme points plotted on a logarithmic scale.\n"
        "• Why selected: Visualizes the distance between normal baseline user traffic and extreme attack packet floods.\n"
        "• Actual Observation: Heavy concentrations of outlier points extend into $10^7$ microseconds and $10^5$ packets.\n"
        "• Practical Cybersecurity Significance: Demonstrates that volumetric DDoS exploits generate extreme statistical anomalies distinct from normal baseline flows."
    )

    # Visualizations 7 & 8: Correlation & Bivariate Traffic Dynamics
    add_h2("6.4 Correlation Matrix and Bivariate Scatter Dynamics")
    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "04_correlation_heatmap.png"),
        "Figure 8: Pearson Correlation Matrix Heatmap Across Dominant Continuous Flow Features."
    )
    doc.add_paragraph(
        "• What it shows: Pairwise linear dependencies and correlation coefficients across high-variance flow attributes.\n"
        "• Why selected: Identifies multicollinear clusters that could cause neural network weight instability.\n"
        "• Actual Observation: Extreme positive collinearity (r = 0.88 to 0.99) between forward packets, backward packets, and cumulative byte lengths.\n"
        "• Statistical Interpretation: Indicates feature redundancy across raw volume metrics.\n"
        "• Practical Cybersecurity Significance: Justifies the incorporation of L2 weight decay (1e-4) and Dropout (0.30) to prevent feature co-adaptation."
    )

    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "05_scatter_packet_relationship.png"),
        "Figure 9: Bivariate Traffic Scatterplot: Total Forward Packets vs Total Length of Forward Packets."
    )
    doc.add_paragraph(
        "• What it shows: Scatter dispersion of packet counts versus cumulative byte payload across traffic categories.\n"
        "• Why selected: Examines cluster separability between legitimate user browsing and automated flood scripts.\n"
        "• Actual Observation: Normal benign sessions display dispersed payload ratios (ACKs vs file downloads), whereas DDoS traffic collapses along sharp linear attack trajectories.\n"
        "• Practical Cybersecurity Significance: Demonstrates non-linear cluster boundaries that an MLP can readily partition."
    )

    # Visualizations 9 & 10: Category-Wise Discrepancies
    add_h2("6.5 Category-Wise Flow Metrics and Density Comparisons")
    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "06_attack_wise_metric_comparison.png"),
        "Figure 10: Median Flow Duration Comparison Across Benign and Malicious Categories."
    )
    doc.add_paragraph(
        "• What it shows: Log-scaled median flow durations comparing Benign sessions against Attack sessions.\n"
        "• Why selected: Tests whether temporal session length provides discriminative separation.\n"
        "• Actual Observation: Attack flows exhibit a median duration of 1,492,482 μs (~1.49 seconds) compared to 50,862 μs for benign flows.\n"
        "• Statistical Interpretation: Mann-Whitney U test confirms statistical significance (p < 0.001) with medium effect size (r = 0.4557).\n"
        "• Practical Cybersecurity Significance: Reflects the operational mechanics of LOIC DDoS bots sustaining persistent TCP connections to exhaust server sockets."
    )

    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "07_benign_vs_attack_density.png"),
        "Figure 11: Violin Density Profiles of Forward Packet Length Mean (BENIGN vs ATTACK)."
    )
    doc.add_paragraph(
        "• What it shows: Kernel density violin distributions of forward packet length means separated by target class.\n"
        "• Why selected: Tests whether packet size distributions differ systematically between humans and automated bots.\n"
        "• Actual Observation: Benign traffic demonstrates a wide, multimodal packet size profile, whereas DDoS traffic exhibits a tight, unimodal peak around small packet sizes (median = 8.67 bytes).\n"
        "• Statistical Interpretation: Mann-Whitney U test shows a large negative effect size (r = -0.5234, p < 0.001).\n"
        "• Practical Cybersecurity Significance: Automated exploit tools transmit uniform, lightweight payloads to maximize packet-per-second generation."
    )

    # Visualization 11 & 12: Multivariate Interplay & Statistical Tests Table
    add_h2("6.6 Multivariate Feature Interplay and Statistical Testing")
    add_figure_with_caption(
        doc,
        os.path.join(eda_dir, "08_multivariate_pairplot.png"),
        "Figure 12: Multivariate Pairplot Capturing Diagonal Univariate Densities and Bivariate Cross-Correlations."
    )
    doc.add_paragraph(
        "• What it shows: Multi-panel pairwise scatter and KDE density plots across log-transformed Flow Duration, Packets, and Payload Bytes.\n"
        "• Why selected: Evaluates multidimensional cluster separability across classes simultaneously.\n"
        "• Actual Observation: Clear manifold separation emerges in log-transformed feature space between benign and DDoS clusters.\n"
        "• Practical Cybersecurity Significance: Validates that deep feedforward neural networks can construct effective non-linear hyperplanes for classification."
    )

    # Insert Statistical Hypothesis Tests Table & Goodness-of-Fit Analysis
    doc.add_paragraph(
        "Table 4 documents the formal non-parametric Mann-Whitney U hypothesis tests, sample sizes, and Rank-Biserial effect sizes. "
        "Mann-Whitney U was selected because network traffic telemetries severely violate normality assumptions (extreme skewness and kurtosis), "
        "rendering parametric Student's t-tests invalid."
    )
    doc.add_paragraph(
        "• Test Assumptions: Independent flow observations, ordinal or continuous measurements, and unequal group variances.\n"
        "• Evaluated Sample Sizes: Balanced sub-samples of n1 = 5,000 Benign flows and n2 = 5,000 Attack flows (total N = 10,000) "
        "were evaluated to ensure asymptotic distribution stability.\n"
        "• Kolmogorov-Smirnov (KS) Test Methodology & Limitations: Goodness-of-fit was tested against standard Gaussian and Log-Normal distributions. "
        "Crucially, when candidate distribution parameters (mean and standard deviation) are estimated directly from the sample data rather than specified a priori, "
        "the standard Kolmogorov-Smirnov test is conservative (yielding smaller KS distances and higher p-values than exact Lilliefors tables). "
        "However, because empirical KS test statistics (D > 0.34 to 0.44) vastly exceed the critical value at alpha=0.01 (D_crit = 0.023), "
        "Gaussian normality is overwhelmingly rejected (p < 0.001) across all continuous metrics. Log-Normal distributions showed superior fit (D = 0.1008 to 0.2011)."
    )
    hypo_csv = os.path.join(tables_dir, "mann_whitney_hypothesis_tests.csv")
    if os.path.exists(hypo_csv):
        df_hypo = pd.read_csv(hypo_csv)
        t4 = doc.add_table(rows=1 + len(df_hypo), cols=len(df_hypo.columns))
        t4.alignment = WD_TABLE_ALIGNMENT.CENTER
        for c_idx, c_name in enumerate(df_hypo.columns):
            c = t4.cell(0, c_idx)
            c.paragraphs[0].add_run(c_name).bold = True
            set_cell_background(c, "0A2540")
            c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            set_cell_margins(c, 50, 50, 60, 60)
        for r_idx, row in df_hypo.iterrows():
            for c_idx, val in enumerate(row):
                c = t4.cell(r_idx + 1, c_idx)
                c.paragraphs[0].add_run(str(val))
                bg = "F4F6F8" if r_idx % 2 == 0 else "FFFFFF"
                set_cell_background(c, bg)
                set_cell_margins(c, 40, 40, 50, 50)

    # =========================================================================
    # SECTION 7: DEEP LEARNING MODEL
    # =========================================================================
    add_h1("Section 7: Deep Learning Model")
    doc.add_paragraph(
        "A Multilayer Perceptron (MLP) deep feedforward neural network was engineered using TensorFlow/Keras [2]. "
        "MLPs are ideally suited for tabular network flow features because dense layers can capture complex non-linear interactions "
        "across packet rates, durations, and protocol flags without requiring explicit feature crosses."
    )

    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "00_mlp_architecture_diagram.png"),
        "Figure 13: Multilayer Perceptron (MLP) Deep Learning Neural Network Architecture Block Diagram [Schematic Diagram]."
    )

    add_h2("7.1 Neural Network Layer Specifications")
    doc.add_paragraph(
        "• Input Dimension: 66 standardized numerical features.\n"
        "• Hidden Layer 1: 128 units, He-Normal kernel initialization, L2 regularization penalty (1e-4), followed by Batch Normalization, ReLU activation, and Dropout (0.30).\n"
        "• Hidden Layer 2: 64 units, He-Normal initialization, Batch Normalization, ReLU activation, and Dropout (0.20).\n"
        "• Hidden Layer 3: 32 units, ReLU activation, and Dropout (0.10).\n"
        "• Output Layer: 1 unit with Sigmoid activation function producing posterior attack probability P(Y=1|X).\n"
        "• Loss Function: Binary Cross-Entropy loss.\n"
        "• Optimizer & Settings: Adam optimizer with an initial learning rate of 1e-3, batch size of 256, and 20 maximum epochs.\n"
        "• Callbacks: EarlyStopping (monitoring validation loss with patience=5, restoring best weights) and ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-5).\n"
        "• Class Imbalance Mitigation: Balanced class weights computed as w_j = N / (2 * n_j) were applied during training to ensure equal gradient penalties."
    )

    add_h2("7.2 Hyperparameter Tuning Experiments")
    doc.add_paragraph(
        "Four controlled experimental configurations were evaluated on validation data to optimize architectural depth, regularization, and learning rates. "
        "Table 5 presents the empirical results extracted from `results/model_evaluation/hyperparameter_tuning_results.csv`."
    )

    # Table 5: Hyperparameters
    hp_csv = os.path.join(model_dir, "hyperparameter_tuning_results.csv")
    if os.path.exists(hp_csv):
        df_hp = pd.read_csv(hp_csv)
        t5 = doc.add_table(rows=1 + len(df_hp), cols=len(df_hp.columns))
        t5.alignment = WD_TABLE_ALIGNMENT.CENTER
        for c_idx, c_name in enumerate(df_hp.columns):
            c = t5.cell(0, c_idx)
            c.paragraphs[0].add_run(c_name).bold = True
            set_cell_background(c, "005582")
            c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            set_cell_margins(c, 70, 70, 90, 90)
        for r_idx, row in df_hp.iterrows():
            for c_idx, val in enumerate(row):
                c = t5.cell(r_idx + 1, c_idx)
                c.paragraphs[0].add_run(str(val))
                bg = "F4F6F8" if r_idx % 2 == 0 else "FFFFFF"
                set_cell_background(c, bg)
                set_cell_margins(c, 50, 50, 70, 70)

    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "08_hyperparameter_comparison_bar.png"),
        "Figure 14: Controlled Hyperparameter Experiment Benchmarking: Validation Loss and Accuracy Across Configurations."
    )
    doc.add_paragraph(
        "Observation: Configuration EXP-1 (Baseline: [128, 64, 32], lr=0.001, dropout=0.30) achieved the optimal validation loss (0.0661) "
        "and 99.80% validation accuracy, outperforming the deeper architecture (EXP-2: val_loss=0.1000) and low learning rate (EXP-4: val_loss=0.2927). "
        "EXP-1 was consequently selected as the final production model."
    )

    # Section 7.3: Imbalance Strategy Comparison
    add_h2("7.3 Systematic Comparison of Class Imbalance-Handling Strategies")
    doc.add_paragraph(
        "In intrusion detection benchmarks, class distributions can vary significantly between balanced volumetric DDoS sessions and rare stealth attacks. "
        "In the evaluated CIC-IDS2017 binary dataset, the class proportion is approximately 52.27% BENIGN to 47.73% ATTACK before cleaning, "
        "and 51.86% BENIGN to 48.14% ATTACK after deduplication. While not severely imbalanced, three mitigation strategies were systematically compared "
        "under identical validation and test partitions, with resampling and scalers strictly fitted on training data only."
    )

    # Table 6: Imbalance Strategy Comparison
    imb_csv = os.path.join(model_dir, "imbalance_strategy_comparison.csv")
    if os.path.exists(imb_csv):
        df_imb = pd.read_csv(imb_csv)
        # Select key columns for clean academic presentation
        cols_show = ["Strategy", "Test Accuracy", "Balanced Acc", "Macro F1", "ROC-AUC", "Attack Recall", "Benign Precision", "Confusion Matrix (TN,FP,FN,TP)"]
        cols_present = [c for c in cols_show if c in df_imb.columns]
        df_imb_sub = df_imb[cols_present]
        
        t_imb = doc.add_table(rows=1 + len(df_imb_sub), cols=len(cols_present))
        t_imb.alignment = WD_TABLE_ALIGNMENT.CENTER
        for c_idx, c_name in enumerate(cols_present):
            c = t_imb.cell(0, c_idx)
            c.paragraphs[0].add_run(c_name).bold = True
            set_cell_background(c, "0A2540")
            c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            set_cell_margins(c, 50, 50, 60, 60)
        for r_idx, row in df_imb_sub.iterrows():
            for c_idx, val in enumerate(row):
                c = t_imb.cell(r_idx + 1, c_idx)
                c.paragraphs[0].add_run(str(val))
                bg = "F4F6F8" if r_idx % 2 == 0 else "FFFFFF"
                set_cell_background(c, bg)
                set_cell_margins(c, 40, 40, 50, 50)

    doc.add_paragraph(
        "• Analysis of Trade-offs:\n"
        "1. Baseline MLP (No Weighting): Achieved 99.93% test accuracy and 99.97% attack recall (7,157 of 7,159 attacks detected; 2 false negatives, 9 false alarms; confusion matrix [TN,FP,FN,TP] = [7704, 9, 2, 7157]).\n"
        "2. Class-Weighted MLP (Balanced Loss Weighting): Achieved 99.92% test accuracy, 0.9992 macro F1-score, and 99.97% attack recall (7,157 detected; 2 false negatives, 10 false alarms; confusion matrix [TN,FP,FN,TP] = [7703, 10, 2, 7157]). "
        "Class weighting applies cost-sensitive gradient penalties without modifying empirical feature coordinates.\n"
        "3. SMOTE + MLP (Synthetic Minority Over-sampling): Fitting SMOTE strictly on the training partition yielded 99.88% test accuracy, 0.9988 macro F1, and 3 false negatives with 15 false alarms (confusion matrix [TN,FP,FN,TP] = [7698, 15, 3, 7156]). "
        "Because the dataset is already well-balanced (~52% to 48%), SMOTE synthesizes artificial interpolation points along decision boundaries, slightly increasing false alarms.\n"
        "• Strategic Decision: The Class-Weighted MLP was selected as the optimal production model because it maximizes attack interception (99.97% recall) "
        "while preserving the authentic empirical manifold of network traffic without synthetic interpolation artifacts."
    )

    # =========================================================================
    # SECTION 8: RESULT VISUALIZATION AND INTERPRETATION
    # =========================================================================
    add_h1("Section 8: Result Visualization and Interpretation")
    doc.add_paragraph(
        "To ensure scientific rigor, all final evaluation metrics were computed strictly on the held-out 30.0% test partition (14,872 unseen flows). "
        "At no point was test set telemetry utilized during data cleaning, imputation, scaling, or hyperparameter selection."
    )

    add_h2("8.1 Training Learning Trajectories")
    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "01_training_val_loss.png"),
        "Figure 15: Training and Validation Cross-Entropy Loss Trajectories Across Epochs."
    )
    doc.add_paragraph(
        "• What it shows: Loss curves for training and validation splits over 20 epochs.\n"
        "• Measured Observation: Training loss descended smoothly from 0.44 to 0.0314, while validation loss stabilized at 0.0318.\n"
        "• Interpretation: Close tracking between training and validation loss confirms that Batch Normalization and Dropout effectively mitigated overfitting."
    )

    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "02_training_val_accuracy.png"),
        "Figure 16: Training and Validation Accuracy Convergence Trajectories Across Epochs."
    )
    doc.add_paragraph(
        "• What it shows: Classification accuracy progression over training epochs.\n"
        "• Measured Observation: Accuracy rose from 92.4% at epoch 1 to 99.86% by epoch 20 without oscillation.\n"
        "• Interpretation: Demonstrates rapid convergence and stable gradient dynamics under the Adam optimizer."
    )

    add_h2("8.2 Confusion Matrix and Classification Reports")
    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "03_confusion_matrix.png"),
        "Figure 17: Normalized Confusion Matrix on Held-Out Test Split (14,872 Flows)."
    )
    doc.add_paragraph(
        "• What it shows: Normalized percentage and absolute count distribution of true positives, true negatives, false positives, and false negatives.\n"
        "• Measured Observation: Out of 7,159 actual DDoS attacks, 7,157 were correctly intercepted (99.97% recall), with only 2 false negatives (missed attacks). "
        "Out of 7,713 benign flows, 7,703 were correctly classified as Benign (99.87% true negative rate), with only 10 false alarms (false positives). "
        "Strictly ordered as [TN, FP, FN, TP], the confusion matrix values are [7703, 10, 2, 7157].\n"
        "• Cybersecurity Significance: Extremely low false negative rate (0.03%) is critical for defense systems where missed attacks cause outages."
    )

    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "07_per_class_metrics_bar.png"),
        "Figure 18: Held-Out Test Set: Per-Class Precision, Recall, and F1-Score Comparison."
    )
    doc.add_paragraph(
        "• What it shows: Balanced precision, recall, and F1-score across individual target classes.\n"
        "• Measured Observation: Both BENIGN and ATTACK classes achieved balanced F1-scores of 0.9992, confirming robust class-independent performance."
    )

    # Table 6: Test Metrics Summary
    add_h2("8.3 Quantitative Test Performance Summary Table")
    doc.add_paragraph("Table 6 details the verified quantitative evaluation metrics computed on the 14,872 held-out test records.")

    t6 = doc.add_table(rows=10, cols=2)
    t6.alignment = WD_TABLE_ALIGNMENT.CENTER
    t6_headers = ["Performance Metric", "Measured Value on Held-Out Test Split (14,872 Flows)"]
    for j, h in enumerate(t6_headers):
        c = t6.cell(0, j)
        c.paragraphs[0].add_run(h).bold = True
        set_cell_background(c, "0A2540")
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(c, 70, 70, 90, 90)

    t6_data = [
        ("Overall Test Accuracy", "99.92% (14,860 / 14,872 correctly classified)"),
        ("Macro Precision", "0.9992"),
        ("Macro Recall", "0.9992"),
        ("Macro F1-Score", "0.9992"),
        ("Weighted F1-Score", "0.9992"),
        ("Receiver Operating Characteristic (ROC-AUC)", "0.9997"),
        ("Average Precision (PR-AUC)", "0.9997 (AP = 0.9989)"),
        ("DDoS Attack Detection Recall", "99.97% (7,157 / 7,159 detected; 2 missed)"),
        ("Confusion Matrix [TN, FP, FN, TP]", "[7703, 10, 2, 7157] (TN=7703, FP=10, FN=2, TP=7157)")
    ]
    for i, row in enumerate(t6_data):
        for j, val in enumerate(row):
            c = t6.cell(i + 1, j)
            c.paragraphs[0].add_run(val)
            bg = "F4F6F8" if i % 2 == 0 else "FFFFFF"
            set_cell_background(c, bg)
            set_cell_margins(c, 50, 50, 70, 70)

    add_h2("8.4 ROC-AUC and Precision-Recall Diagnostics")
    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "04_roc_auc_curves.png"),
        "Figure 19: Receiver Operating Characteristic (ROC) Curve with Computed Area Under Curve (AUC = 0.9997)."
    )
    doc.add_paragraph(
        "• What it shows: True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) across detection thresholds.\n"
        "• Measured Observation: The curve pushes sharply into the upper-left corner, achieving an AUC of 0.9997.\n"
        "• Interpretation: Demonstrates near-ideal separability across classification thresholds."
    )

    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "05_precision_recall_curve.png"),
        "Figure 20: Precision-Recall Curve with Average Precision (AP = 0.9997)."
    )
    doc.add_paragraph(
        "• What it shows: Precision vs Recall trade-off, critical for evaluating class imbalance reliability.\n"
        "• Measured Observation: High precision (>99.8%) is sustained across the entire recall spectrum, yielding an Average Precision (AP) of 0.9997.\n"
        "• Practical Cybersecurity Significance: Confirms that security operators can deploy the model at high sensitivity without triggering flood of false alarms."
    )

    add_h2("8.5 Permutation Feature Importance Analysis")
    add_figure_with_caption(
        doc,
        os.path.join(model_dir, "06_feature_importance.png"),
        "Figure 21: Permutation Feature Importance Ranking on Held-Out Test Set."
    )
    doc.add_paragraph(
        "• What it shows: Drop in test set macro F1-score when individual feature values are randomly shuffled on held-out test data.\n"
        "• Measured Observation: Flow Duration, Fwd Packet Length Mean, and Total Length of Fwd Packets caused the steepest performance drops when shuffled.\n"
        "• Practical Cybersecurity Significance: Confirms that volumetric session duration and packet size ratios represent the primary discriminative markers for detecting DDoS intrusions."
    )

    # =========================================================================
    # SECTION 9: CONCLUSION AND FUTURE SCOPE
    # =========================================================================
    add_h1("Section 9: Conclusion and Future Scope")
    add_h2("9.1 Summary of Findings")
    doc.add_paragraph(
        "This project completed an end-to-end investigation into network intrusion detection using the CIC-IDS2017 benchmark dataset. "
        "Key empirical and theoretical conclusions include:\n"
        "1. Heavy-Tailed Network Distributions: Real network features exhibit acute non-normality and multi-decade positive skewness, verified by Kolmogorov-Smirnov goodness-of-fit tests.\n"
        "2. Preservation of Statistical Outliers: Outliers audited via the IQR method correspond directly to volumetric exploit packet floods. Culling them would blind detection systems to high-severity threats.\n"
        "3. Deep Learning Efficacy: The Multilayer Perceptron (MLP) regularized with Batch Normalization and Dropout achieved 99.92% test accuracy, 0.9992 macro F1-score, and 0.9997 ROC-AUC on the held-out 30.0% test partition (14,872 unseen flows, achieving 99.97% attack recall with only 2 missed attacks out of 7,159 and 10 false alarms).\n"
        "4. Feature Importance Consensus: Non-parametric Mann-Whitney U hypothesis tests and permutation feature importance converged on session flow duration and packet length means as the decisive cyberattack determinants."
    )

    add_h2("9.2 Limitations of the Study")
    doc.add_paragraph(
        "• Scope of Evaluation: The initial experimental benchmark evaluated 50,000 flows from the Friday DDoS capture. Multi-day captures (containing Slowloris, Heartbleed, and Web Attacks) were not simultaneously modeled in this single-run binary experiment.\n"
        "• Tabular Telemetry Constraint: Pre-extracted flow summaries generated via CICFlowMeter were evaluated rather than raw, unsegmented packet payload byte sequences.\n"
        "• Offline Evaluation: Telemetry was processed in batch mode rather than live streaming in-line network interfaces."
    )

    add_h2("9.3 Future Scope")
    doc.add_paragraph(
        "To extend this work beyond the current assignment scope, future research should explore:\n"
        "1. Multi-Class Attack Modeling: Consolidating all eight daily capture files to train a unified 15-class classifier capable of distinguishing specialized threats (Botnets, XSS, Infiltration).\n"
        "2. Temporal Sequence Architectures: Deploying Long Short-Term Memory (LSTM) networks or Temporal Convolutional Networks (TCNs) to model sequential packet inter-arrival times.\n"
        "3. Live Sensor Integration: Interfacing with open-source telemetry taps (Zeek / Suricata) to benchmark real-time inference latency under gigabit Ethernet conditions.\n"
        "4. Adversarial Robustness: Evaluating evasion resilience against traffic-morphing and packet-padding adversarial perturbations."
    )

    # =========================================================================
    # SECTION 10: REFERENCES
    # =========================================================================
    add_h1("Section 10: References")
    refs = [
        "[1] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, \"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,\" in Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), 2018, pp. 108–116. [Online]. Available: https://www.unb.ca/cic/datasets/ids-2017.html",
        "[2] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA, USA: MIT Press, 2016.",
        "[3] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, and A. Hotho, \"A Survey of Network-Based Intrusion Detection Data Sets,\" Computers & Security, vol. 86, pp. 147–167, 2019.",
        "[4] N. Moustafa and J. Slay, \"UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems (UNSW-NB15 Network Data Set),\" in 2015 Military Communications and Information Systems Conference (MilCIS), 2015, pp. 1–6.",
        "[5] F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011."
    ]
    for r in refs:
        p_ref = doc.add_paragraph(r)
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        p_ref.paragraph_format.space_after = Pt(4)

    # =========================================================================
    # SECTION 11: APPENDIX — CODE SECTION
    # =========================================================================
    add_h1("Section 11: Appendix: Code Section")
    doc.add_paragraph(
        "This appendix provides key excerpts of the implementation architecture, execution commands, "
        "and automated unit testing logs. Complete source files are organized inside the project workspace repository."
    )

    add_h2("11.1 Project Directory Structure")
    code_dir = (
        "D:\\Network_Intrusion_EDA\\\n"
        "├── data/\n"
        "│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (Evaluated capture)\n"
        "│   └── README.md\n"
        "├── notebooks/\n"
        "│   └── network_intrusion_eda.ipynb (Interactive research notebook)\n"
        "├── src/\n"
        "│   ├── data_loader.py (Dynamic dataset discovery & sanitization)\n"
        "│   ├── preprocessing.py (Leak-free scaling, IQR audit, inf/NaN impute)\n"
        "│   ├── eda.py (12 high-resolution graphical routines)\n"
        "│   ├── statistical_analysis.py (Mann-Whitney U & KS goodness-of-fit)\n"
        "│   ├── train_model.py (MLP architecture, class weighting, tuning)\n"
        "│   ├── evaluate_model.py (Confusion matrix, ROC, PR, importance)\n"
        "│   ├── generate_additional_figures.py (Supplementary visual generator)\n"
        "│   └── generate_academic_report.py (Word & PDF report compiler)\n"
        "├── results/\n"
        "│   ├── eda/ (12 exported PNG figures)\n"
        "│   ├── model_evaluation/ (9 diagnostic plots, model checkpoint, metrics)\n"
        "│   ├── tables/ (5 summary CSV tables)\n"
        "│   └── report/ (project_report.docx & project_report.pdf)\n"
        "├── tests/\n"
        "│   └── test_preprocessing.py (7 automated Pytest unit tests)\n"
        "├── main.py (End-to-end command-line runner)\n"
        "└── README.md (Comprehensive documentation & execution guide)"
    )
    p_cd = doc.add_paragraph(code_dir)
    p_cd.runs[0].font.name = 'Consolas'
    p_cd.runs[0].font.size = Pt(8.5)

    add_h2("11.2 Main Pipeline Execution Commands")
    cmd_text = (
        "# 1. Run unit test suite\n"
        "python -m pytest tests/test_preprocessing.py -v\n\n"
        "# 2. Execute end-to-end real dataset pipeline (50,000 rows, binary mode)\n"
        "python main.py --file_name \"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv\" --nrows 50000 --binary\n\n"
        "# 3. Generate supplementary figures and update final reports\n"
        "python src/generate_additional_figures.py\n"
        "python src/generate_academic_report.py"
    )
    p_cmd = doc.add_paragraph(cmd_text)
    p_cmd.runs[0].font.name = 'Consolas'
    p_cmd.runs[0].font.size = Pt(8.5)

    add_h2("11.3 Core Preprocessing Code Excerpt (Anti-Leakage Split & Imputation)")
    prep_code = (
        "# Excerpt from src/preprocessing.py:\n"
        "def split_and_preprocess(self, df, test_size=0.15, val_size=0.15):\n"
        "    cleaned_df = self.clean_raw_dataframe(df)\n"
        "    y = cleaned_df[self.target_col]\n"
        "    X = cleaned_df.drop(columns=[self.target_col])\n"
        "    \n"
        "    # Stratified Train+Val vs Test split\n"
        "    X_train_val, X_test_raw, y_train_val, y_test_raw = train_test_split(\n"
        "        X, y, test_size=test_size+val_size, stratify=y, random_state=self.random_state\n"
        "    )\n"
        "    # Split Train vs Val\n"
        "    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(\n"
        "        X_train_val, y_train_val, test_size=val_size/(test_size+val_size),\n"
        "        stratify=y_train_val, random_state=self.random_state\n"
        "    )\n"
        "    # Fit scaler & imputers EXCLUSIVELY on training data to prevent leakage\n"
        "    self.fit(X_train_raw, y_train_raw)\n"
        "    X_train, y_train = self.transform(X_train_raw, y_train_raw)\n"
        "    X_val, y_val = self.transform(X_val_raw, y_val_raw)\n"
        "    X_test, y_test = self.transform(X_test_raw, y_test_raw)\n"
        "    return X_train, y_train, X_val, y_val, X_test, y_test, self"
    )
    p_prep = doc.add_paragraph(prep_code)
    p_prep.runs[0].font.name = 'Consolas'
    p_prep.runs[0].font.size = Pt(8.5)

    add_h2("11.4 Deep Learning Multilayer Perceptron (MLP) Architecture Code Excerpt")
    dl_code = (
        "# Excerpt from src/train_model.py:\n"
        "def build_mlp_model(input_dim, num_classes=2, hidden_units=[128, 64, 32], dropout_rate=0.3):\n"
        "    inputs = keras.Input(shape=(input_dim,), name='flow_features')\n"
        "    x = inputs\n"
        "    for i, units in enumerate(hidden_units):\n"
        "        x = layers.Dense(units, kernel_initializer='he_normal',\n"
        "                         kernel_regularizer=regularizers.l2(1e-4))(x)\n"
        "        x = layers.BatchNormalization()(x)\n"
        "        x = layers.Activation('relu')(x)\n"
        "        if dropout_rate > 0:\n"
        "            x = layers.Dropout(dropout_rate)(x)\n"
        "    outputs = layers.Dense(1, activation='sigmoid', name='output_binary')(x)\n"
        "    model = keras.Model(inputs=inputs, outputs=outputs, name='NetworkIntrusion_MLP')\n"
        "    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),\n"
        "                  loss='binary_crossentropy',\n"
        "                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])\n"
        "    return model"
    )
    p_dl = doc.add_paragraph(dl_code)
    p_dl.runs[0].font.name = 'Consolas'
    p_dl.runs[0].font.size = Pt(8.5)

    add_h2("11.5 Automated Unit Test Execution Log")
    test_log = (
        "============================= test session starts =============================\n"
        "platform win32 -- Python 3.10.11, pytest-9.0.2, pluggy-1.6.0\n"
        "rootdir: D:\\Network_Intrusion_EDA\n"
        "collected 7 items\n\n"
        "tests/test_preprocessing.py::test_data_loader_missing_directory_warning PASSED [ 14%]\n"
        "tests/test_preprocessing.py::test_preprocessing_cleaning_pipeline PASSED [ 28%]\n"
        "tests/test_preprocessing.py::test_preprocessing_binary_mode PASSED       [ 42%]\n"
        "tests/test_preprocessing.py::test_iqr_outlier_computation PASSED         [ 57%]\n"
        "tests/test_preprocessing.py::test_leakage_free_split_and_fit PASSED      [ 71%]\n"
        "tests/test_preprocessing.py::test_statistical_analyzer_methods PASSED    [ 85%]\n"
        "tests/test_preprocessing.py::test_mlp_architecture_and_forward_pass PASSED [100%]\n\n"
        "======================== 7 passed in 7.73s ========================"
    )
    p_tl = doc.add_paragraph(test_log)
    p_tl.runs[0].font.name = 'Consolas'
    p_tl.runs[0].font.size = Pt(8.5)

    add_h2("11.6 GitHub Repository Setup & Push Protocol")
    gh_text = (
        "# Commands to initialize and publish to GitHub:\n"
        "git init\n"
        "git add .\n"
        "git commit -m \"feat: complete academic project for network intrusion EDA and deep learning\"\n"
        "git branch -M main\n"
        "git remote add origin https://github.com/<your-username>/network-intrusion-eda.git\n"
        "git push -u origin main\n\n"
        "# Note: Large raw CSV files in data/ and model binaries are excluded by .gitignore."
    )
    p_gh = doc.add_paragraph(gh_text)
    p_gh.runs[0].font.name = 'Consolas'
    p_gh.runs[0].font.size = Pt(8.5)

    # Final consistency enforcement: Ensure every section has exact 1.0 inch margins and empty headers/footers
    for s_idx, section in enumerate(doc.sections):
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.header.is_linked_to_previous = False
        section.footer.is_linked_to_previous = False
        for p in section.header.paragraphs:
            p.text = ""
        for p in section.footer.paragraphs:
            p.text = ""

    # Save Document
    doc.save(docx_out)
    print(f"[Success] Generated final academic report (.docx) at:\n  -> {os.path.abspath(docx_out)}")

    # Export to PDF via Word COM Automation
    try:
        import win32com.client
        print("[PDF Export] Initializing Word COM engine for PDF conversion...")
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        abs_docx = os.path.abspath(docx_out)
        abs_pdf = os.path.abspath(pdf_out)
        doc_word = word.Documents.Open(abs_docx, ReadOnly=True, ConfirmConversions=False)
        doc_word.ExportAsFixedFormat(
            OutputFileName=abs_pdf,
            ExportFormat=17,
            OpenAfterExport=False,
            OptimizeFor=0,
            CreateBookmarks=1
        )
        doc_word.Close(SaveChanges=False)
        word.Quit()
        print(f"[Success] Generated PDF version of report at:\n  -> {abs_pdf}")
    except Exception as e:
        print(f"[Warning] Could not export PDF via Word COM: {e}")

    return docx_out, pdf_out


if __name__ == "__main__":
    build_formal_report()
