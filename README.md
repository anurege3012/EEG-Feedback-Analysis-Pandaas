# EEG-Feedback-Analysis-Pandaas
We are creating this repo as an intermediate analysis and review of our project.
# Team Pandaas: EEG Feedback Processing Analysis
**Members:** Adarsh Panda, Abriti Chakraborty, Anushree Rege  
**Dataset:** [Average Task Value (OpenNeuro ds004147)](https://openneuro.org/datasets/ds004147/)

## 📌 Project Overview
This project investigates how reward-related feedback influences neural responses, specifically focusing on the **Feedback-Related Negativity (FRN)** and **P300** components. We aim to replicate and extend findings regarding whether the brain's response to reward is modulated by the contextual value (high vs. low expectation) of the task.

### Main Research Question
> Does the brain react differently to rewards depending on the internal visual predictions and the contextual value of the task?

---

## 🛠 Analysis Pipeline
Our team implemented an improved version of the original authors' pipeline to ensure robustness and better noise reduction.

| Step | Authors' Pipeline | Our Updated Pipeline | Justification |
| :--- | :--- | :--- | :--- |
| **Filtering** | 0.1–40 Hz | **0.1–30 Hz** | Removes high-frequency muscle noise while preserving P300 content. |
| **Referencing** | Average | **Average** | No mastoid electrodes present in dataset; average is most stable. |
| **Artifact Removal** | ICA + EOG | **ICA + Pseudo-EOG** | Used Fp1/Fp2 as proxies since dedicated EOG sensors were missing. |
| **Epoching** | -200 to 800 ms | **-200 to 800 ms** | Standard window for FRN and P300 analysis. |
| **Statistics** | TRF Regression | **ERP Difference Waves** | prioritized robustness and clear visualization of components. |

---

## 📂 Repository Structure
* `EEG.py` & `EEG_2.py`: Initial data loading and event visualization for subjects 28 and 29.
* `EEG_3.py`: Implementation of the correlation-based ICA artifact removal.
* `EEG_4.py`: Final processing script generating sanity check plots (PSD, ICA Topographies, and ERPs).
* `milestone4_outputs/`: Contains all generated figures including:
    * **PSD Plots**: Comparison of signal power before and after filtering.
    * **ICA Topographies**: Visualization of identified independent components (e.g., eye blinks).
    * **Butterfly Plots**: Overlaid evoked responses across all 31 EEG channels.

## 🚀 How to Run
1. Install dependencies: `pip install mne numpy matplotlib`
2. Ensure the BrainVision data files (`.vhdr`, `.vmrk`, `.eeg`) are in the correct subject folders (e.g., `sub-28/eeg/`).
3. Run the final analysis script:
   ```bash
   python EEG_4.py
