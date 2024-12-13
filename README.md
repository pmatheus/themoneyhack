# README

## Overview

This project provides a training and testing framework for a trading reinforcement learning (RL) agent using a Transformer-based Double DQN approach with LoRA (Low-Rank Adaptation) modules and noisy linear layers. The code fetches financial data, prepares features (including technical indicators via TA-Lib), trains an RL agent, and then tests and selects episodes for a "Reference Class" (RC) based on performance metrics. Finally, results are packaged into a neat experimental directory.

## Prerequisites

- **Operating System:** The code runs on macOS, Linux, and Windows.
- **Python Version:** Python 3.8+ is recommended.
- **Conda (recommended):** Using a conda environment ensures easier management of dependencies.
- **Pip:** Some packages will be installed via `pip` after creating the conda environment.

## Setting Up the Environment

### 1. Create and Activate a Conda Environment

```bash
conda create -n trading-env python=3.9
conda activate trading-env
```

*(Feel free to adjust the Python version as needed.)*

### 2. Installing Dependencies

Most Python packages can be installed via `pip` inside the conda environment. However, **TA-Lib** often needs special handling due to its C dependencies.

**Required Packages:**
- `numpy`
- `pandas`
- `matplotlib`
- `torch` (PyTorch)
- `ccxt`
- `psutil`
- `tensorboard`
- `json`, `hashlib`, `datetime`, `cProfile` (part of the Python standard library)

**Special note for TA-Lib:**
- TA-Lib (the Python binding) depends on the [TA-Lib C library](https://github.com/TA-Lib/ta-lib). You must install the TA-Lib C library separately before installing the Python wrapper.

Below are platform-specific instructions for TA-Lib:

#### macOS

On macOS, you can use `brew` to install the TA-Lib C library:

```bash
brew install ta-lib
```

Once the library is installed, you can install the Python wrapper:

```bash
pip install TA-Lib
```

#### Linux (Ubuntu/Debian)

For Linux, install TA-Lib from source:

```bash
sudo apt-get update
sudo apt-get install -y gcc make wget

# Download and install TA-Lib C library
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..

# Now install the Python wrapper
pip install TA-Lib
```

*(You may need to adjust the prefix and install directories depending on your environment. Some distributions may also have TA-Lib in their package repositories.)*

#### Windows

On Windows, you can either:

1. Use [conda-forge] channel for a pre-compiled TA-Lib library:
   ```bash
   conda install -c conda-forge ta-lib
   ```

2. Or manually download TA-Lib binary files from:
   [https://github.com/mrjbq7/ta-lib#dependencies](https://github.com/mrjbq7/ta-lib#dependencies)

   After placing the TA-Lib `.dll` files appropriately (e.g., in a location on your `PATH` or using the recommended instructions), run:

   ```bash
   pip install TA-Lib
   ```

If you have trouble on Windows, the conda-forge approach is often the simplest.

### 3. Installing the Remaining Python Packages

After TA-Lib is installed, you can install the rest via `pip`:

```bash
pip install numpy pandas matplotlib psutil ccxt tensorboard
```

For PyTorch, please refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your operating system and hardware (CPU/GPU). For example:

```bash
pip install torch torchvision torchaudio
```

*(Adjust as needed for CUDA/cuDNN if using GPU.)*

### 4. Verifying the Setup

Once all packages are installed, verify that TA-Lib and PyTorch are working:

```python
import talib
import torch

print("TA-Lib version:", talib.__version__)
print("PyTorch version:", torch.__version__)
```

If no errors occur, you are good to proceed.

## Running the Code

1. **Data Fetching:** The code automatically fetches OHLCV data from Binance via `ccxt`. Make sure you have a stable internet connection. Data will be saved in a `data/` directory.
   
2. **Training:** Run the main Python script:
   ```bash
   python main.py
   ```

   The code will:
   - Fetch data (or load existing data if available).
   - Prepare data by adding technical indicators (using TA-Lib).
   - Train the agent for the specified number of episodes.
   - Save model checkpoints, plots, and logs in corresponding directories.

3. **Testing and RC Selection:** After training, the code tests episodes and selects certain runs based on performance criteria, packaging them into a neat experiment folder.

4. **Results:** Results (plots, metrics, weights) are saved in `experiments/`, `reports/`, `rc/`, and `weights/` directories. After the run completes, these directories are organized into an `experiments/` folder for archiving.

## Additional Notes

- **Reproducibility:** A fixed random seed is set in the code to ensure reproducibility.
- **GPU Support:** If CUDA or Apple MPS is available, the code will use GPU acceleration. Otherwise, it falls back to CPU.
- **LoRA and NoisyNets:** The code includes LoRA adapters and noisy linear layers for model experimentation and test-time training.
- **Plots and Logging:** Live plotting uses `matplotlib` and logging uses `tensorboard`. Check `runs/` directory for TensorBoard logs:
  
  ```bash
  tensorboard --logdir runs
  ```

## Troubleshooting

- **TA-Lib Installation Errors:**  
  Ensure the TA-Lib C library is installed first. Check environment variables and library paths if encountering errors.
  
- **Missing Data:**  
  If Binance fails to fetch data, ensure you have an internet connection and correct symbol/timeframe. If issues persist, try other exchanges supported by ccxt.

- **Memory/Performance Issues:**  
  For large datasets or many episodes, consider more RAM or filtering data. Adjust hyperparameters (like `NUM_EPISODES` or `PLOT_UPDATE_INTERVAL`) to manage performance.

## Conclusion

This README should guide you through setting up the environment and running the code on macOS, Linux, or Windows. By carefully following the steps, especially for TA-Lib installation, you can successfully train and test the RL trading agent.