# Figure contract: validation evidence sensitivity

Backend: Python/matplotlib exclusively. White backgrounds; 183 mm quantitative grids; editable SVG text (`svg.fonttype=none`), embedded TrueType PDF (`pdf.fonttype=42`), 300 dpi PNG. No simulated values in experimental figures. Source data: saved validation summary/classwise/missingness CSVs with paired dialogue bootstrap intervals. One anchor seed per dataset; exploratory comparisons, no generalization claim.

Scientific argument: attenuating or masking stored representations quantifies model dependence on those representations. A change is not evidence that the original prediction was unreliable, and a weak response is not proof that a modality lacks real emotion evidence. T1–T4 are stored arrays, not established encoder layer numbers. No semantic text or physical audio/video intervention is represented.

Panels and distinct evidence:

1. `modality_acc_curves`: accuracy vs rho including unmodified rho0, two datasets; T-All/A/V family comparison. Show both V all and V present, retaining missing rows in all.
2. `trueprob_curves`: true-label probability change vs rho; direction and magnitude of representation dependence, with the same visual strata.
3. `text_layer_comparison`: T1–T4 plus T-All accuracy curves, separating stored-array contributions without interpreting them as known encoder layers.
4. `flip_curves`: flip rate vs rho; distinguishes changed decisions from accuracy change.
5. `class_heatmap`: per-true-class accuracy change at rho1 across T-All/A/V. Annotate values and n; use a diverging zero-centered scale. Exploratory class comparisons.
6. `meld_visual_missingness`: retained zero-visual vs present sample counts and original accuracy, alongside V rho1 accuracy (no intervention for missing rows). The missingness group is observational, not a causal comparison.

Uncertainty: 2000 paired dialogue bootstrap draws from the original dialogue pool, shared across all conditions within dataset; fixed original-correctness and missingness subgroup membership. Lines show percentile 95% bands; heatmap annotations refer to point estimates and export CI source tables. Random masks use mean of per-repeat metrics, never probability ensembles.

Visual encoding: text blue family, audio orange, visual green; T1–T4 graded blues and T-All darkest. Missing/all visual line styles differ. Axes and legends use plain English to avoid font dependence; reports explain interpretation in Chinese. Every panel indicates dataset, units, and subgroup definitions. No spline interpolation, no unreported condition selection, no truncated accuracy axes without explicit limits.

Review risks: dialogue dependence; IEMOCAP only 12 dialogues; feature missingness; varying class supports; nonmonotonic response; sparse classes or empty bootstrap subgroups; multiple exploratory comparisons; interventions toward training means differ from physical removal. Preserve NA rather than turning empty groups into zero. Record exports and perform Python-based file/render QA after real data exist.
