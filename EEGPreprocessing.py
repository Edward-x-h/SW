import mne
import numpy as np
import glob
import os

data_path = 'edward'
file_list = sorted(glob.glob(os.path.join(data_path, '*.edf')))
if len(file_list) != 18:
    print("Warning: Expected 18 participant files, found", len(file_list))
cleaned_data_list = []
erp_list = []
band_power_alpha_list = []
band_power_beta_list = []
band_power_gamma_list = []
psd_list = []
freqs_list = []

for file in file_list:
    raw = mne.io.read_raw_edf(file, preload=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    raw.notch_filter(freqs=50, fir_design='firwin')
    ica = mne.preprocessing.ICA(n_components=16, random_state=42, max_iter='auto')
    ica.fit(raw)
    eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
    ica.exclude.extend(eog_indices)
    if 'ECG' in raw.ch_names:
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, threshold=3.0)
        ica.exclude.extend(ecg_indices)
    ica.apply(raw)
    events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
    event_id = {'task': 1}
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True, reject_by_annotation=True)
    epochs.interpolate_bads(reset_bads=True)
    evoked = epochs.average()
    psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=1, fmax=40, n_fft=256, average='mean')
    psd_avg = np.mean(psds, axis=0)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    gamma_band = (30, 40)
    epochs_alpha = epochs.copy().filter(l_freq=alpha_band[0], h_freq=alpha_band[1], fir_design='firwin')
    epochs_beta = epochs.copy().filter(l_freq=beta_band[0], h_freq=beta_band[1], fir_design='firwin')
    epochs_gamma = epochs.copy().filter(l_freq=gamma_band[0], h_freq=gamma_band[1], fir_design='firwin')
    psds_alpha, _ = mne.time_frequency.psd_welch(epochs_alpha, fmin=alpha_band[0], fmax=alpha_band[1], n_fft=256, average='mean')
    psds_beta, _ = mne.time_frequency.psd_welch(epochs_beta, fmin=beta_band[0], fmax=beta_band[1], n_fft=256, average='mean')
    psds_gamma, _ = mne.time_frequency.psd_welch(epochs_gamma, fmin=gamma_band[0], fmax=gamma_band[1], n_fft=256, average='mean')
    band_alpha = np.mean(psds_alpha, axis=(0, 2))
    band_beta = np.mean(psds_beta, axis=(0, 2))
    band_gamma = np.mean(psds_gamma, axis=(0, 2))
    cleaned_data_list.append(epochs.get_data())
    erp_list.append(evoked.data)
    band_power_alpha_list.append(band_alpha)
    band_power_beta_list.append(band_beta)
    band_power_gamma_list.append(band_gamma)
    psd_list.append(psd_avg)
    freqs_list.append(freqs)

cleaned_data = np.array(cleaned_data_list)
erp_array = np.array(erp_list)
band_power_alpha = np.array(band_power_alpha_list)
band_power_beta = np.array(band_power_beta_list)
band_power_gamma = np.array(band_power_gamma_list)
psd_array = np.array(psd_list)
freqs_array = np.array(freqs_list)

np.save('cleaned_eeg_data.npy', cleaned_data)
np.save('erp_data.npy', erp_array)
np.savez('frequency_band_power.npz', alpha=band_power_alpha, beta=band_power_beta, gamma=band_power_gamma)
np.save('psd_data.npy', psd_array)
np.save('freqs_data.npy', freqs_array)
