# Geo-Trajectories Repository

This repository contains code for processing and analyzing single-cell streaming data, as described in the associated [preprint "Single-cell FM streaming using genetically encoded oscillators"](https://www.biorxiv.org/content/10.1101/2025.02.28.640587v2). The pipeline enables the generation of GEO frequency landscapes, developmental state-space trajectories (MDS), and pixel-level pseudotime mapping and projection onto cell contours and masks.

## Repository Structure

- `src/geo_trajectories/`: Core library containing modular logic for image processing, landscape analysis, and pseudotime calculation.
- `scripts/`: Standardized entry-point scripts for executing the analysis pipeline.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for FFT processing)

### Dependencies
Install the required packages using pip:
```bash
pip install numpy pandas torch tifffile scikit-image scipy seaborn scikit-learn progressbar2
```

The core FFT processing logic depends on the `cellstream` package, which can be installed from:
[https://github.com/CoyleLab-UW-Madison/cellstream](https://github.com/CoyleLab-UW-Madison/cellstream)

## Workflow Guide

The analysis is divided into six sequential steps. Each script supports command-line arguments for custom configuration (run with `--help` for details).

### 1. Pixel Extraction
Process raw `.nd2` or `.tif` files to extract pixel-level GEO features (D levels, E levels, associated metrics (F_bin, amplitude, etc.)).
```bash
python scripts/01_process_pixels.py --base_dir ./data --output all_pixels.parquet
```

### 2. Frequency Landscape Generation
Generate frequency landscapes (2D histograms) for each condition and replicate.
```bash
python scripts/02_generate_landscapes.py --input all_pixels.parquet --output_prefix landscapes
```

### 3. Trajectory Analysis
Compute pairwise distances between landscapes and project them into a lower dimensional space using MDS.
```bash
python scripts/03_analyze_trajectories.py --input landscapes_collapsed.pbz2 --method welch
```

### 4. Z-Score Computation
Compute pixel-level Z-scores for every pixel against a collection of temporally evolving developmental landscapes.
```bash
python scripts/04_compute_zscores.py --pixel_input all_pixels.parquet --landscape_input landscapes_collapsed.pbz2
```

### 5. Pseudotime Calculation
Calculate a pseudotime estimate for every pixels based on similarity to frequency landscapes from different timepoints.
```bash
python scripts/05_compute_pseudotime.py --input all_pixels_with_zscores.parquet --output all_pixels_with_pseudotime.parquet
```

### 6. Visualization & Cell Stats
Project pseudotime back onto images and aggregate results for individual cell contours or masks.
```bash
python scripts/06_visualize_results.py --input all_pixels_with_pseudotime.parquet --label_dir ./masks
```

## Core Modules (`src/geo_trajectories/`)

- `core.py`: FFT-based feature extraction from time-series images.
- `landscapes.py`: 2D histogram pooling and Welch's T-test comparisons.
- `trajectories.py`: MDS fitting and 3D Waddington trajectory visualization.
- `pseudotime.py`: Vectorized pseudotime estimation and image mapping.
- `utils.py`: Compression-enabled data I/O and helper functions.
"# geo-trajectories" 
