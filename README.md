## Methodology

The proposed pipeline generates robust ECG beat templates from multi-lead recordings by combining
QRS detection, peak refinement, and Woody-style alignment to improve beat-to-beat temporal consistency
prior to averaging.

### 1. Optional Bandpass Filtering
ECG signals may be filtered using a Butterworth bandpass filter to suppress baseline wander
(low-frequency drift) and reduce high-frequency noise. This step can improve QRS detection
stability depending on the recording conditions.

### 2. QRS Detection
QRS complexes are detected using the `wfdb.processing.gqrs_detect` algorithm on a reference lead
(commonly lead 0). The detector returns candidate sample indices corresponding to QRS locations.

### 3. Peak Refinement
Detected QRS indices are refined using a local correction step (e.g., `processing.peaks.correct_peaks`),
which searches within a defined neighborhood around each candidate index to locate a more accurate peak
position based on the waveform characteristics.

### 4. Woody-Style Beat Alignment
To improve beat alignment, a Woody-style approach is applied using cross-correlation:

- **Segment Extraction:**  
  For each detected QRS index, a short segment is extracted using a fixed window
  (e.g., 20 ms before and 50 ms after the QRS index).

- **Vector Magnitude Reference:**  
  A single alignment reference signal is formed by computing the vector magnitude across leads,
  reducing sensitivity to lead-specific morphology differences.

- **Template Formation:**  
  A preliminary template is computed as the mean of all extracted segments.

- **Delay Estimation:**  
  Each beat segment is cross-correlated with the template to estimate the time lag that maximizes
  similarity. The lag is optionally constrained to a maximum allowed shift to prevent unrealistic
  corrections.

- **Index Correction:**  
  QRS indices are shifted according to the estimated delays, yielding corrected beat centers for
  subsequent extraction.

### 5. Beat Window Extraction and Template Averaging
Using the corrected QRS indices, a wider window is extracted for each lead (e.g., 80 ms pre-QRS and
100 ms post-QRS). Beats that fall near boundaries can be handled via NaN padding or by skipping incomplete
segments. A final per-lead beat template is obtained by averaging across extracted beats using NaN-safe
statistics.

The final output is a multi-lead average beat template per ECG record, suitable for morphology analysis,
signal quality assessment, feature extraction, or machine learning preprocessing.
