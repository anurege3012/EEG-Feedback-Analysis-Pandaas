import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

# -------------------------------
# PATHS
# -------------------------------
data_path = "sub-28/eeg/sub-28_task-casinos_eeg.vhdr"
out_dir = "milestone4_outputs"
os.makedirs(out_dir, exist_ok=True)

# -------------------------------
# 1. LOAD DATA
# -------------------------------
raw = mne.io.read_raw_brainvision(data_path, preload=True)

# Set standard montage (IMPORTANT for topoplots)
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="ignore")

# -------------------------------
# 2. PSD BEFORE FILTERING
# -------------------------------
fig_psd_before = raw.compute_psd(fmax=50).plot(show=False)
fig_psd_before.savefig(f"{out_dir}/01_psd_before_filtering.png", dpi=150)
plt.close(fig_psd_before)

# -------------------------------
# 3. FILTERING (ERP-safe)
# -------------------------------
raw_filt = raw.copy().filter(l_freq=0.1, h_freq=30)

# PSD AFTER FILTERING
fig_psd_after = raw_filt.compute_psd(fmax=50).plot(show=False)
fig_psd_after.savefig(f"{out_dir}/02_psd_after_filtering.png", dpi=150)
plt.close(fig_psd_after)

# -------------------------------
# 4. REFERENCING
# -------------------------------
raw_filt.set_eeg_reference("average")

# -------------------------------
# 5. ICA
# -------------------------------
ica = ICA(n_components=20, random_state=97, max_iter="auto")
ica.fit(raw_filt)

# ICA TOPOGRAPHIES
fig_ica = ica.plot_components(show=False)
fig_ica.savefig(f"{out_dir}/03_ica_topographies.png", dpi=150)
plt.close(fig_ica)

# -------------------------------
# 6. ICA ARTIFACT REMOVAL (Fp1 blink proxy)
# -------------------------------
blink_inds, scores = ica.find_bads_eog(raw_filt, ch_name="Fp1")
ica.exclude = blink_inds

raw_clean = raw_filt.copy()
ica.apply(raw_clean)

# -------------------------------
# 7. RAW vs CLEAN EEG (STATIC)
# -------------------------------
raw_static = raw.copy().pick("eeg").get_data()[:, :5000]
clean_static = raw_clean.copy().pick("eeg").get_data()[:, :5000]

plt.figure(figsize=(10, 4))
plt.plot(raw_static.T, color="black", linewidth=0.3)
plt.title("Raw EEG (Static)")
plt.savefig(f"{out_dir}/04_raw_static.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(clean_static.T, color="black", linewidth=0.3)
plt.title("Clean EEG (After ICA)")
plt.savefig(f"{out_dir}/05_clean_static.png", dpi=150)
plt.close()

# -------------------------------
# 8. EPOCHING
# -------------------------------
events, event_id = mne.events_from_annotations(raw_clean)

epochs = mne.Epochs(
    raw_clean,
    events,
    event_id={"Stimulus/S 11": 11},
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2, 0),
    preload=True
)

# -------------------------------
# 9. ERP
# -------------------------------
evoked = epochs.average()
fig_erp = evoked.plot(spatial_colors=True, show=False)
fig_erp.savefig(f"{out_dir}/06_erp.png", dpi=150)
plt.close(fig_erp)

# -------------------------------
# 10. BUTTERFLY PLOT
# -------------------------------
plt.figure(figsize=(8, 4))
plt.plot(evoked.times, evoked.data.T, linewidth=0.7)
plt.axvline(0, color="k", linestyle="--")
plt.title("Butterfly Plot – Subject 28")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.savefig(f"{out_dir}/07_butterfly.png", dpi=150)
plt.close()

print("\n✅ ALL MILESTONE-4 SANITY CHECK PLOTS GENERATED")
print(f"📁 Saved in folder: {out_dir}")
