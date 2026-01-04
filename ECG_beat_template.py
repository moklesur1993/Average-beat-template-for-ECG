import numpy as np
from wfdb import processing
from scipy.signal import butter, filtfilt, correlate, correlation_lags
from typing import List, Optional, Tuple


def bandpass_filter(
    ecg: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 3,
) -> np.ndarray:
    """
    Bandpass filter an ECG signal (multi-lead supported).

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal of shape (n_samples,) or (n_samples, n_leads).
    fs : float
        Sampling frequency in Hz.
    low_hz : float
        Low cutoff frequency in Hz.
    high_hz : float
        High cutoff frequency in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered ECG with same shape as input.
    """
    ecg = np.asarray(ecg)
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, ecg, axis=0)


def woody_correct_qrs_indices(
    ecg: np.ndarray,
    q_indices: np.ndarray,
    i_left: int,
    i_right: int,
    max_shift_samples: int = 20,
) -> np.ndarray:
    """
    Woody-like alignment to correct QRS indices by cross-correlating beat segments
    with a running template.

    This implementation:
    - Builds beat segments around each provided QRS index using vector magnitude across leads.
    - Forms a template as the mean of beat segments.
    - For each beat, finds the lag of max cross-correlation with the template.
    - Adjusts that beat's QRS index by the estimated lag (clipped to max_shift_samples).

    IMPORTANT:
    - Beats that would be out of bounds are skipped and not corrected.

    Parameters
    ----------
    ecg : np.ndarray
        ECG array of shape (n_samples, n_leads).
    q_indices : np.ndarray
        Candidate QRS indices (sample indices).
    i_left : int
        Samples to include to the LEFT of q (segment start = q - i_left).
    i_right : int
        Samples to include to the RIGHT of q (segment end = q + i_right).
    max_shift_samples : int
        Maximum allowed correction shift (in samples). Correlation lags are clipped.

    Returns
    -------
    np.ndarray
        Corrected QRS indices for the subset of beats that were valid (in bounds).
        If no beats are valid, returns an empty array.
    """
    ecg = np.asarray(ecg)
    q_indices = np.asarray(q_indices, dtype=int)

    if ecg.ndim != 2:
        raise ValueError("ecg must be (n_samples, n_leads)")
    if q_indices.size == 0:
        return np.array([], dtype=int)

    n_samples = ecg.shape[0]
    seg_len = i_left + i_right

    # Vector magnitude across leads (robust single reference signal for alignment)
    vm = np.sqrt(np.sum(ecg ** 2, axis=1))  # shape: (n_samples,)

    beats = []
    kept_q = []

    for q in q_indices:
        start = q - i_left
        end = q + i_right
        if start < 0 or end > n_samples:
            continue
        segment = vm[start:end]
        if segment.shape[0] == seg_len:
            beats.append(segment)
            kept_q.append(q)

    if len(beats) == 0:
        return np.array([], dtype=int)

    beats = np.asarray(beats)  # (n_beats, seg_len)
    kept_q = np.asarray(kept_q, dtype=int)

    # Template (mean beat)
    template = np.nanmean(beats, axis=0)

    # Cross-correlation for each beat
    lags = correlation_lags(template.size, seg_len, mode="full")
    delays = np.empty(beats.shape[0], dtype=int)

    for i, beat in enumerate(beats):
        c = correlate(template, beat, mode="full")
        lag = int(lags[np.argmax(c)])
        # Clip lag to avoid insane shifts
        lag = int(np.clip(lag, -max_shift_samples, max_shift_samples))
        delays[i] = lag

    # If beat appears delayed by +lag samples relative to template, shift q backward by lag
    corrected = kept_q - delays

    # Keep corrected indices still within bounds for later extraction windows
    corrected = corrected[(corrected - i_left >= 0) & (corrected + i_right <= n_samples)]
    return corrected


def extract_average_beat(
    ecg: np.ndarray,
    q_indices: np.ndarray,
    pre: int,
    post: int,
    pad_with_nan: bool = True,
) -> np.ndarray:
    """
    Extract and average multi-lead beats around each QRS index.

    Parameters
    ----------
    ecg : np.ndarray
        ECG array of shape (n_samples, n_leads).
    q_indices : np.ndarray
        QRS indices (sample indices) used as beat centers.
    pre : int
        Samples before QRS to include.
    post : int
        Samples after QRS to include.
    pad_with_nan : bool
        If True: allow edge beats and pad missing samples with NaN.
        If False: skip beats that don't fully fit in [q-pre, q+post).

    Returns
    -------
    np.ndarray
        Average beat template of shape (n_leads, pre + post).
        Values are NaN if no valid beats exist for a lead.
    """
    ecg = np.asarray(ecg)
    q_indices = np.asarray(q_indices, dtype=int)

    n_samples, n_leads = ecg.shape
    win_len = pre + post

    avg = np.full((n_leads, win_len), np.nan)

    if q_indices.size == 0:
        return avg

    for lead in range(n_leads):
        beat_stack = []

        for q in q_indices:
            start = q - pre
            end = q + post

            if not pad_with_nan:
                if start < 0 or end > n_samples:
                    continue
                beat_stack.append(ecg[start:end, lead])
            else:
                # NaN-padded extraction
                out = np.full(win_len, np.nan)
                s = max(0, start)
                e = min(n_samples, end)
                out[(s - start):(s - start) + (e - s)] = ecg[s:e, lead]
                beat_stack.append(out)

        if len(beat_stack) > 0:
            beat_stack = np.asarray(beat_stack)
            if not np.all(np.isnan(beat_stack)):
                avg[lead, :] = np.nanmean(beat_stack, axis=0)

    return avg


def process_ecg_batch(
    ecg_data: np.ndarray,
    fs: float,
    i_left_ms: float = 20.0,
    i_right_ms: float = 50.0,
    pre_ms: float = 80.0,
    post_ms: float = 100.0,
    filter_before_qrs: bool = False,
    max_shift_ms: float = 20.0,
) -> List[Optional[np.ndarray]]:
    """
    Process a batch of ECG records to compute an average beat template per record.

    Expected input shape
    --------------------
    ecg_data: (n_records, n_samples, n_leads)

    Pipeline
    --------
    - Optionally bandpass filter each record.
    - Detect QRS on lead 0 with gqrs_detect.
    - Refine peaks with correct_peaks.
    - Align QRS indices with Woody-style cross-correlation (using vector magnitude).
    - Extract and average multi-lead beats around corrected QRS.

    Parameters
    ----------
    ecg_data : np.ndarray
        ECG batch of shape (n_records, n_samples, n_leads).
    fs : float
        Sampling frequency (Hz).
    i_left_ms, i_right_ms : float
        Window for Woody alignment segment around QRS (ms).
    pre_ms, post_ms : float
        Window for beat template extraction around QRS (ms).
    filter_before_qrs : bool
        Whether to bandpass filter before QRS detection and template extraction.
    max_shift_ms : float
        Max allowed Woody correction (ms).

    Returns
    -------
    List[Optional[np.ndarray]]
        Each item is:
        - avg_beat of shape (n_leads, pre+post) in samples, or
        - None if no QRS was detected / no valid beats.
    """
    ecg_data = np.asarray(ecg_data)
    if ecg_data.ndim != 3:
        raise ValueError("ecg_data must have shape (n_records, n_samples, n_leads)")

    i_left = int(round((i_left_ms / 1000.0) * fs))
    i_right = int(round((i_right_ms / 1000.0) * fs))
    pre = int(round((pre_ms / 1000.0) * fs))
    post = int(round((post_ms / 1000.0) * fs))
    max_shift_samples = int(round((max_shift_ms / 1000.0) * fs))

    results: List[Optional[np.ndarray]] = []

    for idx, ecg in enumerate(ecg_data):
        if idx % 1000 == 0:
            print(f"Processing instance {idx}")

        x = bandpass_filter(ecg, fs) if filter_before_qrs else ecg

        # QRS detection on lead 0
        qrs_inds = processing.gqrs_detect(x[:, 0], fs=fs)

        if qrs_inds.size == 0:
            print(f"No QRS detected for instance {idx}")
            results.append(None)
            continue

        # Peak correction on lead 0
        corrected_peaks = processing.peaks.correct_peaks(
            x[:, 0],
            peak_inds=qrs_inds,
            search_radius=int(round(0.10 * fs)),   # 100 ms search radius (more interpretable)
            smooth_window_size=int(round(0.13 * fs)),  # ~130 ms smoothing window
        )

        # Woody alignment (returns only valid corrected peaks)
        q_corrected = woody_correct_qrs_indices(
            x, corrected_peaks, i_left=i_left, i_right=i_right, max_shift_samples=max_shift_samples
        )

        if q_corrected.size == 0:
            results.append(None)
            continue

        avg_beat = extract_average_beat(
            x, q_corrected, pre=pre, post=post, pad_with_nan=True
        )
        results.append(avg_beat)

    return results


# Example usage:
# ECG should be shaped (n_records, n_samples, n_leads)
# average_beats = process_ecg_batch(ECG, fs=400, filter_before_qrs=False)
