# Perturbation feasibility matrix

| Perturbation | Required data | Available now? | Recommended implementation point | Decision |
|---|---|---|---|---|
| Irrelevant question | raw/full text + pinned encoder | text yes; exact historical encoder no | text before re-encoding | CONDITIONAL |
| Assertion | raw/full text + pinned encoder | text yes; exact historical encoder no | text before re-encoding | CONDITIONAL |
| Probe | raw/full text + pinned encoder | text yes; exact historical encoder no | text before re-encoding | CONDITIONAL |
| Sample Corruption | raw/full text + pinned encoder | text yes; exact historical encoder no | text before re-encoding | CONDITIONAL |
| Text Gaussian noise | T1-T4 arrays | yes | feature space, before BN | READY |
| Text Feature Dropout | T1-T4 arrays | yes | feature space, before BN | READY |
| Audio Pitch Shift | attributable waveform + extractor | no proven XP04 raw/extractor chain | audio signal, then re-extract | BLOCKED |
| Audio Timbre | attributable waveform + extractor | no proven XP04 raw/extractor chain | audio signal, then re-extract | BLOCKED |
| Audio Feature Dropout | utterance audio vector | yes | feature space | READY |
| Facial dynamics smoothing | frame sequence + face track + extractor | no | video/frame space, then re-extract | BLOCKED |
| Local facial occlusion | frames + region geometry + extractor | no | video/frame space, then re-extract | BLOCKED |
| Visual Feature Dropout | utterance visual vector | yes | feature space | READY |

“READY” means mechanically possible on frozen inputs, not scientifically validated as beneficial. Gaussian noise/dropout must remain separate from semantic/signal-space claims. T1-T4 can be perturbed independently, but the standard model independently normalizes and then averages them; layer-specific studies must record the pre-BN intervention point and preserve untouched control arrays.

Before semantic text work, create a new immutable feature namespace and manifest rather than overwriting XP04 pickles. Before A/V signal work, restore or establish raw-data provenance, define the raw-ID mapping, pin the extractor and all assets, reproduce a clean baseline, and only then generate perturbed features.

The row-level machine-readable version is `perturbation_feasibility.csv`.
