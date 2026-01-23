import mne
import numpy as np

# -------------------------------------------------------------
# 1) LOAD RAW BRAINVISION DATA
# -------------------------------------------------------------
raw = mne.io.read_raw_brainvision(
    "sub-28/eeg/sub-28_task-casinos_eeg.vhdr",
    preload=True
)

# -------------------------------------------------------------
# 2) FILTERING (corrected based on professor feedback)
# -------------------------------------------------------------
# Authors used 0.1–40 Hz. We use 0.1–30 Hz:
# - keeps P300 low-frequency content
# - removes high-frequency muscle noise
raw_filt = raw.copy().filter(l_freq=0.1, h_freq=30)

# -------------------------------------------------------------
# 3) REFERENCING (justified)
# -------------------------------------------------------------
# Dataset has NO mastoid electrodes → average reference is best choice
raw_filt.set_eeg_reference('average')

# -------------------------------------------------------------
# 4) ICA ARTIFACT REMOVAL (Fully working for MNE 1.11)
# -------------------------------------------------------------
ica = mne.preprocessing.ICA(n_components=20, random_state=42)
ica.fit(raw_filt)

# ---- Extract ICA sources ----
ica_sources = ica.get_sources(raw_filt).get_data()

# ---- Use Fp1 & Fp2 as pseudo-EOG channels (no EOG exists) ----
try:
    pseudo_eog = raw_filt.copy().pick_channels(['Fp1', 'Fp2']).get_data()
except Exception:
    raise RuntimeError("Fp1 or Fp2 not found in the dataset!")

blink_signal = np.mean(pseudo_eog, axis=0)  # Combine both

# ---- Correlate each ICA component with blink signal ----
corr_values = []
for comp in range(ica_sources.shape[0]):
    r = np.corrcoef(ica_sources[comp], blink_signal)[0, 1]
    corr_values.append(r)

corr_values = np.array(corr_values)

# ---- Mark components with |corr| > 0.3 as blink components ----
blink_like_components = list(np.where(np.abs(corr_values) > 0.3)[0])
print("Blink-like ICA components:", blink_like_components)

# ---- Remove them ----
ica.exclude = blink_like_components
raw_clean = raw_filt.copy()
ica.apply(raw_clean)

print("Removed ICA components:", ica.exclude)

# -------------------------------------------------------------
# 5) EVENT EXTRACTION
# -------------------------------------------------------------
events, event_id = mne.events_from_annotations(raw_clean)

print("Event IDs found:", event_id)

# Use one reward-related event for Milestone 3 (example: Stimulus/S 11)
event_code = "Stimulus/S 11"

if event_code not in event_id:
    raise ValueError(f"{event_code} not found in event_id. Available: {event_id}")

# -------------------------------------------------------------
# 6) EPOCHING
# -------------------------------------------------------------
epochs = mne.Epochs(
    raw_clean,
    events,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2, 0),
    preload=True
)

print(len(epochs), "epochs extracted.")

# -------------------------------------------------------------
# 7) AVERAGE ERP
# -------------------------------------------------------------
# -------------------------------------------------------------
# 7) AVERAGE ERP (Corrected for MNE 1.11)
# -------------------------------------------------------------
erp = epochs.average()

fig = erp.plot(spatial_colors=True)
fig.figure.suptitle("ERP – Subject 28")



print("ERP plotted successfully!")
