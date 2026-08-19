# Data sources & the path to generalization

**TL;DR.** Project Vimaan's training data is **100% synthetic** (templates →
Pegasus/Flan-T5 paraphrase → clean → merge). A learning-curve experiment
(`ML/experiments/learning_curve.py`) shows slot quality is **flat from 10 %→100 %**
of the data — so *more of the same synthetic data does nothing*. The gains come
from **label quality** (already fixed — see `vimaan_nlu/postprocessor.py`) and,
for real-world generalization, from **real spoken phrasing**. There is **no public
corpus of pilot→copilot cockpit commands** matching our schema; every real
aviation-speech corpus is controller↔pilot radiotelephony (R/T). Their value to
us is **ASR-realism** (how numbers/frequencies/headings are actually spoken), not
intents.

## The generalization gap

- Our "held-out" test set is still synthetic template output, so metrics
  (intent 98.9 %, slot-F1 0.99 after the postprocessor fixes) likely **overstate**
  quality on real ASR transcripts of real speakers.
- Real speech has fillers ("uh, set heading to…"), self-corrections ("altitude one
  zero— sorry, one one thousand"), politeness, partial commands, and accent/ASR
  artifacts the templates never produce.

## Real aviation-speech corpora

| Source | Content | Size | License | HF id / URL | Fetchable now? | Use to Vimaan |
|---|---|---|---|---|---|---|
| **ATCOSIM** | ATC sim speech, controllers only, orthographic transcripts | ~2.4 GB / 9.5k rows | Free for research; HF mirror open | `Jzuluaga/atcosim_corpus` | ✅ open (audio+text parquet) | ASR realism only — clean number/heading/FL phrasing |
| **UWB-ATCC** | Controller **and** pilot R/T, role labels (`_PI`/`_AT`) | ~711 MB / 14k rows | **CC BY-NC-SA 4.0 (non-commercial)** | `Jzuluaga/uwb_atcc` | ✅ open | ASR realism + pilot-side speech |
| **ATCO2-ASR 1h** | Real radio ATC, transcripts + **command/value NER** | 113 MB / 871 rows | **EULA-gated** (atco2.org) | `Jzuluaga/atco2_corpus_1h` | ⚠️ accept EULA first | Closest to slots; still R/T schema |
| **jlvdoorn / jacktol merges** | ATCO2+ATCOSIM/UWB, Whisper-ready | ~2.5 GB | inherits ATCO2 EULA + CC-NC | `jlvdoorn/atco2-asr-atcosim`, `jacktol/ATC-ASR-Dataset` | ✅ open, but encumbered upstream | ASR realism, convenience |
| **TartanAviation (CMU)** | Paired ATC audio + ADS-B, live tower | ~398 h / 531k utt | Open (verify terms) | project / HF | mostly ✅ | Large ASR-realism source |
| **LDC94S14A (ATC0)** | US ATC complete, controller+pilot, timed transcripts | ~70 h | **LDC paid / membership** | catalog.ldc.upenn.edu/LDC94S14A | ❌ paywalled | ASR realism (US accents) |
| **HIWIRE** | Non-native **cockpit command-and-control** (closest task type!) | 8,099 utt | **ELRA paid** (ELRA-S0293) | catalogue.elra.info/…/ELRA-S0293 | ❌ paid | Conceptually closest; different grammar |
| **MALORCA** | >100 h clean ATC, controllers only | >100 h | project-gated | — | ❌ | ASR realism only |

**Honest relevance verdict**
- **Usable as-is for our intents/slots: none.** No corpus uses our schema.
- **Usable after relabeling/filtering:** ATCO2-1h (NER), UWB-ATCC (roles) — but you must strip callsign/clearance structure.
- **ASR-realism only:** ATCOSIM, TartanAviation, LDC ATC0, the merges.
- **Closest task type but paid:** HIWIRE.

## Fetching (open, no gating)

Requires `pip install datasets` (pulls pyarrow). See `scripts/fetch_atc_corpora.py`,
which streams **transcripts only** into a phrasing bank (it does not keep audio):

```bash
pip install datasets
python scripts/fetch_atc_corpora.py --dataset Jzuluaga/atcosim_corpus --limit 5000
python scripts/fetch_atc_corpora.py --dataset Jzuluaga/uwb_atcc      --limit 5000
```

**License caveats:** UWB-ATCC and the jacktol/Tabys/jlvdoorn merges are
**CC BY-NC-SA (non-commercial)**; ATCO2 carries a **EULA**. If Vimaan ever ships
commercially, keep these OUT of the shipped training set — use them only to
*inform* generation (mine surface phrasings), not as shipped labels.

## Recommended plan (highest leverage first)

1. **Record a real gold set — do this first.** Have the team + a few outside voices
   speak each of the 17 intents in many natural variations (headset + cockpit
   noise), transcribe with **Whisper large-v3**, hand-correct. Target **~1,000–2,000
   utterances** (≈50–120/intent, weighted to the common ones). Even ~500 as a
   **held-out real eval set** exposes the synthetic→real gap immediately. Capture
   fillers, self-corrections, politeness, partial commands, spoken number readbacks.
   *This is the single highest-leverage step and it does not require any of the
   corpora above.*
2. **Upgrade synthetic generation: templates → LLM.** Replace the fixed templates
   in `ML/data/generate_slot_dataset.py` with an LLM prompted for many diverse
   paraphrases per intent, emitting the intent+slots as structured JSON (so labels
   are never mis-derived). Constrain slot values to valid ranges. This is the
   cheapest way to gain phrasing diversity while keeping gold labels.
3. **Mine real ATC corpora for surface phrasings (not whole utterances).** From
   ATCOSIM/UWB-ATCC transcripts, extract how numbers/frequencies/headings are
   actually spoken ("flight level two five zero", "one two one decimal five") and
   feed them as a phrasing bank into the generator's slot-realization step. Do
   **not** relabel raw ATC utterances wholesale — the callsign/clearance framing
   pollutes the schema.
4. **Purge synthetic label noise.** ~15 % of the current set is degenerate Flan-T5
   "key: value" echoes (`"altitude = 34200"`); add a cleaner rule in
   `ML/data/clean_flan_t5_dataset.py`.

**Sequence:** record gold eval set → LLM generator seeded with mined phrasings →
train → measure on the real gold set (not synthetic-on-synthetic) → iterate.

## Sources
- ATCO2 paper https://arxiv.org/abs/2211.04054 · data/EULA https://www.atco2.org/data
- HF: https://huggingface.co/datasets/Jzuluaga/atcosim_corpus · https://huggingface.co/datasets/Jzuluaga/uwb_atcc · https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h · https://huggingface.co/datasets/jlvdoorn/atco2-asr-atcosim · https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset
- ATCOSIM home https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
- LDC ATC0 https://catalog.ldc.upenn.edu/LDC94S14A · HIWIRE https://catalogue.elra.info/en-us/repository/browse/ELRA-S0293/
- TartanAviation https://twango.dev/writing/tartanaviation-atc-labels
