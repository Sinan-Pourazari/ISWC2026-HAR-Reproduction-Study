# Reproducing HAR Models in the Wild: Code Implementation and Evaluation Hurdles

This repository contains the source code, patches, and execution scripts for our ISWC reproduction study evaluating five Human Activity Recognition (HAR) frameworks.

Due to conflicting dependencies and varying architectural requirements across the evaluated models, this repository is structured using isolated Git branches. The `main` branch serves only as a directory; no execution code is stored here.

## Repository Structure

Each reproduced paper has its own dedicated branch containing the original codebase (via submodules), our applied patches, and specific execution instructions.

To view the code and reproduction steps for a specific model, switch to its corresponding branch:

* **`TinierHAR`**: Reproduction of *TinierHAR: Towards ultra-lightweight deep learning models for efficient human activity recognition on edge devices*. Includes fixes for dataloader VRAM memory leaks and missing ablation study components.
* **`MobHAR`**: Reproduction of *MobHAR: Source-free knowledge transfer for human activity recognition on mobile devices*. Details the structural incompatibilities and missing DIG model selection logic.
* **`ContrastSense`**: Reproduction of *ContrastSense: Domain-invariant contrastive learning for in-the-wild wearable sensing*. Includes spatial padding fixes and tensor alignments for the NinaPro DB4 and DB5 EMG datasets.
* **`STMAE`**: Reproduction of *Spatial-Temporal Masked Autoencoder for Multi-Device Wearable Human Activity Recognition*. Details environment reconstruction and target user subset evaluations.
* **`GILE`**: Reproduction of *Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition*. Includes hyperparameter alignment and undocumented KL-divergence weighting parameters.

## Getting Started

To explore a specific framework, clone this repository and checkout the desired branch. For example, to view the TinierHAR reproduction:

```bash
# 1. Clone the repository
git clone [https://github.com/Sinan-Pourazari/ISWC2026-HAR-Reproduction-Study.git](https://github.com/Sinan-Pourazari/ISWC2026-HAR-Reproduction-Study.git)
cd iswc-har-reproduction

# 2. Checkout the specific paper's branch
git checkout tinierhar

# 3. Initialize the original authors' code submodule
git submodule update --init --recursive

# 4. Navigate to the original codebase
cd original_code

# 5. Apply the patch
patch -p1 < ../changes.patch