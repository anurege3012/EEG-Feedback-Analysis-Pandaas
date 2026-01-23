#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 10:11:35 2025

@author: anushree
"""

import mne
import os


subject = "sub-29"

vhdr_path = os.path.join(
    subject,
    "eeg",
    f"{subject}_task-casinos_eeg.vhdr"
)

# Load raw EEG
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
print(raw)

# Extract events
events, event_id = mne.events_from_annotations(raw)
print("Event IDs:", event_id)

# Plot continuous EEG with events marked
raw.plot(events=events, event_color='red')
