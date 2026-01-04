<section id="methodology">
  <h2>Methodology</h2>

  <p>
    The proposed pipeline generates robust ECG beat templates from multi-lead recordings by combining
    QRS detection, peak refinement, and Woody-style alignment to improve beat-to-beat temporal consistency
    prior to averaging.
  </p>

  <ol>
    <li>
      <h3>Optional Bandpass Filtering</h3>
      <p>
        ECG signals may be filtered using a Butterworth bandpass filter to suppress baseline wander
        (low-frequency drift) and reduce high-frequency noise. This step can improve QRS detection
        stability depending on the recording conditions.
      </p>
    </li>

    <li>
      <h3>QRS Detection</h3>
      <p>
        QRS complexes are detected using the <code>wfdb.processing.gqrs_detect</code> algorithm on a reference lead
        (commonly lead 0). The detector returns candidate sample indices corresponding to QRS locations.
      </p>
    </li>

    <li>
      <h3>Peak Refinement</h3>
      <p>
        Detected QRS indices are refined using a local correction step (e.g., <code>processing.peaks.correct_peaks</code>),
        which searches within a defined neighborhood around each candidate index to locate a more accurate peak
        position based on the waveform characteristics.
      </p>
    </li>

    <li>
      <h3>Woody-Style Beat Alignment</h3>
      <p>
        To improve beat alignment, a Woody-style approach is applied using cross-correlation:
      </p>
      <ul>
        <li>
          <strong>Segment Extraction:</strong>
          For each detected QRS index, a short segment is extracted using a fixed window
          (e.g., 20 ms before and 50 ms after the QRS index).
        </li>
        <li>
          <strong>Vector Magnitude Reference:</strong>
          A single alignment reference signal is formed by computing the vector magnitude across leads,
          reducing sensitivity to lead-specific morphology differences.
        </li>
        <li>
          <strong>Template Formation:</strong>
          A preliminary template is computed as the mean of all extracted segments.
        </li>
        <li>
          <strong>Delay Estimation:</strong>
          Each beat segment is cross-correlated with the template to estimate the time lag that maximizes
          similarity. The lag is optionally constrained to a maximum allowed shift to prevent unrealistic corrections.
        </li>
        <li>
          <strong>Index Correction:</strong>
          QRS indices are shifted according to the estimated delays, yielding corrected beat centers for
          subsequent extraction.
        </li>
      </ul>
    </li>

    <li>
      <h3>Beat Window Extraction and Template Averaging</h3>
      <p>
        Using the corrected QRS indices, a wider window is extracted for each lead (e.g., 80 ms pre-QRS and
        100 ms post-QRS). Beats that fall near boundaries can be handled via NaN padding or by skipping incomplete
        segments. A final per-lead beat template is obtained by averaging across extracted beats using NaN-safe
        statistics.
      </p>
    </li>
  </ol>

  <p>
    The final output is a multi-lead average beat template per ECG record, suitable for morphology analysis,
    signal quality assessment, feature extraction, or machine learning preprocessing.
  </p>
</section>
