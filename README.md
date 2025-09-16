# TR-ENF: A Trust-Region Optimization Framework for Peak-Agnostic ENF Detection in Audio

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the implementation of **TR-ENF**, a Trust-Region (TR) framework for **Electric Network Frequency (ENF) detection**.

## 🔍 Overview

**TR-ENF** introduces a **trust-region optimizer** for estimation of ENF parameters, particularly under **short audio durations (5–10s)**. The framework deals with peak-agnostic ENF detection by leveraging advanced optimization techniques to improve detection aacuracy in noisy conditions.

## 📊 Dataset

Experiments were performed on the [**ENF-WHU Dataset**](https://github.com/ghua-ac/ENF-WHU-Dataset/tree/master/ENF-WHU-Dataset), which provides **real-world recordings** under diverse acoustic conditions.

### Dataset Structure
After downloading the ENF-WHU dataset, organize your data as follows:

```
recordings/
├── H1/                 # ENF-present signals
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── H0/                 # ENF-free signals
    ├── file1.wav
    ├── file2.wav
    └── ...
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Required packages listed in `requirements.txt`

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/TR-ENF.git
   cd TR-ENF
   ```

2. Create the virtual environment and install dependencies:

   ```bash
   conda create -n trENF python=3.10
   conda activate trENF

   ```
   
   ```bash
   pip install -r requirements.txt
   ```

3. Download and organize the ENF-WHU dataset as described above. Refer to the /h1_files.txt and /h0_files.txt


## 📖 Usage

Run the TR-ENF detection pipeline with:

```bash
python main.py
```

Generate only the ROC/DET curves, using the summary_tr.scv, without running the main file:

```bash
python print_curves.py
```

## 📈 Output Files

Results are automatically saved to `./results/` and include:

| File | Description |
|------|-------------|
| `roc_trenf_5s.png` | ROC curve for TR-ENF (5 s only), with AUC |
| `det_curves_trenf_5_10s.png` | DET curves for TR-ENF (5–10 s), with EER values |
| `per_segment_tr.csv` | Detailed per-segment statistics |
| `summary_tr.csv` | Aggregated per-duration summary |
| `tr_convergence_perseg.csv` | Trust-region convergence metrics per segment |
| `tr_convergence_summary.csv` | Aggregated convergence statistics |

## 🏗️ Repository Structure

```
TR-ENF/
├── recordings/
│   ├── H1/                  # Place ENF-present signals here
│   ├── H0/                  # Place ENF-free signals here
│   ├── h1_files.txt         # Provided list of H1 WAV files
│   └── h0_files.txt         # Provided list of H0 WAV files
├── tr_optimizer.py          # Trust-region optimization module
├── main.py    		         # Main detection pipeline
├── print_curves.py          # Generation of ROC/DET curves without running main.py
├── requirements.txt         # Python dependencies
├── results/                 # Results folder
├── README.md                # This file
└── LICENSE                  # MIT License
```


## 📧 Contact

For questions, issues:

- **Email**: [ckorgial@csd.auth.gr], [tsingalis@csd.auth.gr], and [costas@csd.auth.gr]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This work is currently under review for **2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)**. Hence, the code files will be released publicly upon paper acceptance.

---
