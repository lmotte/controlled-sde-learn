# controlled-sde-learn
![License](https://img.shields.io/badge/License-MIT-black.svg)
[![codecov](https://codecov.io/gh/lmotte/controlled-sde-learn/graph/badge.svg?token=V4O1J7FGOM)](https://codecov.io/gh/lmotte/controlled-sde-learn)
## Overview

This repository provides a Python implementation for estimating the coefficients of controlled stochastic differential
equations (SDEs). The approach leverages kernel methods and Fokker-Planck equation matching to estimate the drift and
diffusion coefficients from a data set of controlled sample paths with random controls.

## Examples

The `examples` folder contains several scripts demonstrating different applications of the `controlled-sde-learn`
library.

1. **example_ornstein_uhlenbeck_paths_plot.py**. Illustrates the generation and plotting of sample paths from a
   controlled Ornstein-Uhlenbeck process.
2. **example_dubins_paths_plot.py**. Illustrates the generation and plotting of sample paths from a
   controlled Dubins process.
3. **example_kde_plot.py**. Demonstrates the use of the `ProbaDensityEstimator` for estimating and visualizing the
   probability density of sample paths from a controlled SDE under different controls.
4. **example_sde_identification_1d.py**. Provides a complete example for simulating a one-dimensional controlled SDE and
   estimating its coefficients using Fokker-Planck matching.
5. **example_sde_identification_2d.py**. Presents a complete example for estimating the coefficients of a
   two-dimensional nonlinear controlled SDE.

## Installation

To install:

1. Clone the repository.
   ```bash
   git clone https://github.com/lmotte/controlled-sde-learn.git
   ```
2. Install the required dependencies (Python 3.x required).
   ```bash
   pip install -r requirements.txt
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.