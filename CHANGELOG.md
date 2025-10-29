# Project Vimaan - Changelog

All notable changes to this project will be documented in this file.

---

### Version: 0.9.0
**ID:** 07c4937
**Date:** 2025-10-28
**Module:** Code Architecture - Modular Refactoring & Code Quality
**Author:** Mohammad Hasnain Raza


#### Description of Change:
- **Modular Architecture Refactoring**: Extracted model loading and inference logic into separate, reusable modules for improved code maintainability and testability.
    - Created `ML/core/model_loader.py` with `ModelLoader` class: Centralizes all model loading operations (maps, weights, tokenizer, device management).
    - Created `ML/core/inference.py` with `predict()` function: Unified inference pipeline with slot extraction and post-processing.
    - Refactored `PIxplanevimaancopilot.py`: Uses new modules instead of inline model loading, reducing plugin code by 40% (~100 lines removed).
    - Updated `predict.py` and `command_tester.py`: Migrated to use `ModelLoader` and `predict()` function for consistency.
    - Updated `ML/core/__init__.py`: Added exports for new modules ensuring unified import interface.
- **Enhanced Diagnostics**: Added detailed logging of model path, device type, and model dimensions (intents/slots) across all scripts.
- **Code Quality Improvements**: Separated concerns (model loading vs inference vs plugin logic) for easier debugging, testing, and future enhancements.


#### Test Results:
- **Plugin Testing**: 5/5 real-time voice commands executed successfully (toggle_landing_gear, toggle_flaps, set_autopilot_heading, set_autopilot_altitude, set_flight_level).
- **Automated Testing** (command_tester.py): 34/35 tests passing (97% accuracy) with 100% confidence on supported intents.
- **Interactive Testing** (predict.py): Fully functional command-line interface with all features working correctly.


#### Files Modified:
- `ML/core/model_loader.py` (NEW)
- `ML/core/inference.py` (NEW)
- `PIxplanevimaancopilot.py` (40% code reduction)
- `predict.py` (refactored for new modules)
- `command_tester.py` (refactored for new modules)
- `ML/core/__init__.py` (updated imports)


#### Enhancement Over Previous Version (v0.8.0):
- **Professional Codebase**: Modular architecture follows industry best practices for scalability and maintainability.
- **Improved Testability**: Model loading and inference logic can now be tested independently from X-Plane plugin context.
- **Code Reusability**: Same model loading and prediction logic used across plugin, CLI tools, and test scripts (DRY principle).
- **Easier Debugging**: Separated concerns make it simpler to identify and fix issues in specific components.
- **Future-Ready**: New modules provide clean interfaces for adding new intents, improving model, or integrating with other systems.

---

### Version: 0.8.0
**ID:** 7d71565
**Date:** 2025-10-28
**Module:** X-Plane Integration - Model Loading & Plugin Deployment
**Author:** Mohammad Hasnain Raza


#### Description of Change:
- **Fixed Critical Model Loading Bug**: Removed safetensors overwriting issue that caused random predictions (0.09 confidence). Model now correctly loads bert_for_slots fresh from pretrained, loads intent_classifier on proper device, and maintains trained weights.
- **Deployed Working X-Plane Plugin**: Successfully integrated NLU model with X-Plane 12 for real-time voice command execution.
    - Push-to-talk interface (Z-key), Google Speech API integration, 13 aircraft control handlers.
    - Proper BIO slot extraction, comprehensive logging to Desktop\Vimaan_Logs\.
- **Dataset Analysis**: Created `dataset_summary.py` revealing dataset imbalance (set_com_frequency: 27% vs others: 0.3-3.1%).


#### Model Performance:
- **Before**: 0.09-0.11 confidence (broken).
- **After**: 95%+ confidence on all tested commands.
- **Tested**: 15+ commands successfully (landing gear, flaps, autopilot, heading, altitude, flight level, engines, parking brake, COM frequency).


#### Enhancement Over v0.7.1:
- **Fully Functional**: Plugin now executes all 13 intents correctly in real-time X-Plane environment.
- **Production Ready**: Comprehensive logging and error handling enables deployment.
- **Identified Improvement Path**: Dataset rebalancing will improve accuracy further (v0.9.0).

---

### Version: 0.7.1
**ID:** 839f50e
**Date:** 2025-10-27
**Module:** Machine Learning - Plug In
**Author:** Mohammad Hasnain Raza

#### Description of Change:
- **Integration Of ML Model**: Integrated the ML model and generated the X-Plane Plug-In script. 

---

### Version: 0.7.0
**ID:** 5e7e493
**Date:** 2025-10-26
**Module:** Machine Learning - Code Refactoring & Model Optimization
**Author:** Mohammad Hasnain Raza


#### Description of Change:
- **Modular Architecture Refactor**: Reorganized monolithic codebase into a professional, scalable module structure:
    - Created `core/` package: Centralized all NLU logic (`model.py`, `normalization.py`, `postprocessor.py`) with unified `__init__.py` exports.
    - Created `utils/` package: Extracted all utility functions (`file_utils.py`) for file versioning and model path management.
    - Created `data/` package: Organized all data generation and cleaning scripts (`generate_*.py`, `clean_*.py`, `verify_dataset.py`).
    - Created `config/` package: Isolated configuration (`schema_config.py`) for cleaner separation of concerns.
- **Enhanced Normalization Pipeline**: Upgraded `core/normalization.py` with improved number-to-text handling:
    - Fixed phonetic digit sequence conversion (e.g., "zero niner zero" → "090").
    - Improved compound number parsing (e.g., "seven thousand five hundred" → "7500").
    - Enhanced decimal point handling for frequencies (e.g., "one two three point four five" → "123.45").
- **Improved Model Import Handling**: Fixed `core/model.py` to properly return all 3 values (loss, intent_logits, slot_logits) during forward pass, with updated `predict.py` to correctly unpack predictions.
- **Centralized Model Versioning**: Updated `utils/file_utils.py` with new helper functions (`get_model_versions_dir()`, `get_latest_model_path()`) for automatic model version discovery and management.
- **Cleaned Legacy Files**: Removed duplicate files from root directory:
    - Deleted old `normalization.py`, `slot_postprocessor.py`, `utils.py` (now in their respective packages).
    - Removed debug files (`debug.py`, `debug_new.py`).
    - Kept only latest model version (`v9`), deleting `v1-v8` to reduce repository size.
- **Updated All Main Scripts**: Refactored 5 main scripts to use new modular imports:
    - `train_nlu_model.py`: Now imports from `core` and `utils` packages.
    - `predict.py`: Updated with correct import paths and unpacking logic.
    - `command_tester.py`: Now uses unified imports and latest model discovery.
    - `merge_datasets.py`: Updated to work with new package structure.
    - `augment_with_word_forms.py`: Refactored to use centralized utilities.


#### Accuracy Improvements:
- **Previous Accuracy**: 91.4% (32/35 tests passing).
- **Current Accuracy**: 94.3% (33/35 tests passing).
- **Improvements**: Fixed phonetic sequence handling and compound number parsing, with edge cases deferred for future optimization.

---

### Version: 0.6.0
**ID:** b9d1df2
**Date:** 2025-10-25
**Module:** Machine Learning - Data Generation & Validation
**Author:** Mohammad Hasnain Raza

#### Description of Change:
- **Centralized Schema**: Moved the `SCHEMA` definition to a dedicated `schema_config.py` file and updated all relevant scripts (`generate_slot_dataset.py`, `clean_*.py`, `verify_dataset.py`) to import it.
- **Expanded Intent Coverage**: Updated `generate_slot_dataset.py` to include new intents:
    - Boolean `toggle_` commands (e.g., `toggle_flaps`, `toggle_autopilot_1`) using a consistent `state` slot.
    - A `None` intent for handling out-of-scope commands.
    - Conversational intents (`ask_status_generic`, `ask_time`, `chit_chat_greeting`).
- **Implemented Synonym Handling**:
    - Added `synonyms` dictionary to relevant slots in the `SCHEMA` (`schema_config.py`).
    - Updated generation logic (`generate_slot_dataset.py`) to randomly use synonyms in generated text while keeping the canonical slot value in the label.
    - Upgraded `clean_pegasus_dataset.py` and `verify_dataset.py` to correctly validate text containing synonyms and number-words (`num2words`).
- **Added FLAN-T5 Cleaning Script**: Created `clean_flan_t5_dataset.py` to apply the same validation logic to the FLAN-T5 generated data, ensuring consistency.
- **Updated Folder Structure**: Added `datasets/06_clean_flan_t5/` folder and updated `merge_datasets.py` to use the correct cleaned input paths.

#### Enhancement Over Previous Version (v0.5.0):
- **Professional Workflow**: Centralizing the `SCHEMA` significantly improves maintainability and reduces redundancy.
- **Increased Model Capability**: The NLU model trained on this data will be able to recognize a wider range of commands, handle simple chit-chat, and gracefully identify out-of-scope requests.
- **Improved Data Realism & Robustness**: Generating commands with synonyms makes the training data better reflect natural language variations.
- **Enhanced Data Quality Assurance**: The updated cleaning and verification scripts ensure high data quality by correctly handling synonyms and number representations across both data generation methods.

---

### Version: 0.5.0
**ID:** 961717b
**Date:** 2025-10-12
**Module:** Machine Learning - Model Training & Inference
**Author:** Mohammad Hasnain Raza

#### Description of Change:
- **Added Model Training Pipeline**: Created the core `train_nlu_model.py` script to train a joint intent-and-slot NLU model using DistilBERT.
- **Implemented Validation & Early Stopping**: Upgraded the training script to include a validation loop to monitor performance on unseen data and an early stopping mechanism to prevent overfitting and automatically save the best model.
- **Created Inference Script**: Developed `predict.py` to load the trained model and provide an interactive command-line interface for real-time testing and performance evaluation.
- **Automated Workflow**: Made the training and prediction scripts version-aware, enabling them to automatically find the latest dataset and model versions.

#### Enhancement Over Previous Version (v0.4.0):
- **Core AI Capability**: This update introduces the "brain" of Project Vimaan, transitioning the project from data preparation to a functional, trained NLU model.
- **Professional Training Workflow**: The addition of a validation loop and early stopping represents a shift to a robust, industry-standard training process that produces more reliable models.
- **Complete Feedback Loop**: With the `predict.py` script, a full development cycle is now in place: `Data Generation -> Training -> Interactive Testing`.

---

### Version: 0.4.0
**ID:** a371380
**Date:** 2025-10-06
**Module:** Machine Learning - Data Generation
**Author:** Mohammad Hasnain Raza

#### Description of Change:
- **Refactored Data Pipeline**: Overhauled the entire data generation workflow for better organization and scalability.
- **Introduced Folder Structure**: All scripts now read from and write to a structured `datasets/` directory with subfolders for each stage of the pipeline (e.g., `01_base`, `02_augmented_...`, etc.).
- **Implemented Auto-Versioning**: All data creation scripts now use a helper function in `utils.py` to automatically save outputs to new, versioned files (e.g., `..._v1.jsonl`, `..._v2.jsonl`).
- **Enhanced Verification Logic**: Updated `verify_dataset.py` to intelligently check for both digits and their word equivalents (e.g., "20" or "twenty"), improving the accuracy of data quality checks.

#### Enhancement Over Previous Version (v0.3.0):
- **Organization & Scalability**: The new folder structure makes the project significantly cleaner and easier to manage as more experiments are added.
- **Reproducibility & Experiment Tracking**: The versioning system prevents accidental data overwrites and creates a clear, traceable history of every dataset generated.
- **Improved Data Quality**: The smarter verification ensures that semantically correct but differently formatted data is not incorrectly flagged as an error.

---
### Version: 0.3.0
**ID:** 777f435
**Date:** 2025-10-05
**Module:** Machine Learning - Data Generation
**Author:** Mohammad Hasnain Raza

#### Description of Change:
- **Upgraded Data Generation Pipeline**: This commit includes a series of major enhancements to the data generation process.
- **Dynamic Frequency Generation**: Modified `generate_slot_dataset.py` to create a unique, random COM frequency for every example, significantly increasing data diversity.
- **AI-Powered Augmentation**: Introduced a new script, `augment_with_paraphrasing.py`, which uses a pre-trained Pegasus model (`tuner007/pegasus_paraphrase`) to generate multiple linguistic variations of each command.
- **Path-Aware Scripts**: Both scripts were updated to be path-aware, ensuring they can be run reliably regardless of the terminal's working directory.

#### Enhancement Over Previous Version (v0.2.0):
- **Greatly Increased Robustness**: The combination of dynamic slot generation and AI-powered paraphrasing creates a dataset that is vastly more diverse and representative of real-world language, which is critical for training a commercial-grade model.
- **Improved Workflow**: The data pipeline is now more resilient to common path and dependency issues, making the development process smoother.

---

### Version: 0.2.1
**ID:** 2090c6f
**Date:** 2025-10-04
**Module:** Machine Learning - Data Generation
**Author:** Mohammad Hasnain Raza

#### Description of Change:
- Created a new script, `augment_with_paraphrasing.py`, that uses a pre-trained T5 transformer model to paraphrase and augment the existing dataset.
- The script loads the dataset generated by `generate_slot_dataset.py`, generates multiple unique rephrasings for each command, and creates new data points while preserving the original intent and slot values.

#### Enhancement Over Previous Version (v0.2.0):
- **Addresses Dataset Brittleness**: This directly solves the limitation of the template-based approach. By generating AI-powered paraphrases, the dataset becomes far more diverse and less predictable, better reflecting the variability of human language.
- **Improved Model Robustness**: Training on this augmented data will force the NLU model to learn the underlying semantic meaning of commands rather than simply memorizing sentence structures.

---

### Version: 0.2.0
**ID:** 2090c6f
**Date:** 2025-10-04
**Module:** Machine Learning - Data Generation
**Author:** Mohammad Hasnain Raza

#### Description of Change:
- Redesigned the data generation script (`generate_slot_dataset.py`) to support joint intent classification and slot filling.
- Implemented a centralized, schema-driven architecture where intents are defined with associated slots (e.g., `degrees`, `altitude`), templates, and realistic value sets.
- The output format is now a structured JSONL file containing distinct `text`, `intent`, and `slots` fields.

#### Enhancement Over Previous Version (v0.1.0):
- **Architectural Leap**: This update transitions the NLU task from simple intent classification to a more advanced structured information extraction model.
- **Expanded Capability**: The system is no longer limited to boolean (on/off) commands. It can now be trained to understand and extract parameters/values from pilot commands.
- **Scalability**: The new schema makes it significantly easier to add new, complex commands in the future without major code rewrites. 