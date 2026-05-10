# Empirical Deep Hedging

Code used in the article Empirical Deep Hedging (Mikkilä & Kanniainen, 2021)

Prepared settings files:

Constant volatility:
- GBM_kappa1 (risk factor = 1)
- GBM_kappa2 (risk factor = 2)
- GBM_kappa3 (risk factor = 3)

Constant volatility:
- Heston_kappa1 (risk factor = 1)
- Heston_kappa2 (risk factor = 2)
- Heston_kappa3 (risk factor = 3)

Empirical data:
- Empirical_kappa1 (risk factor = 1)
- Empirical_kappa2 (risk factor = 2)
- Empirical_kappa3 (risk factor = 3)

These files can be used to replicate the results in the article. The codebase has been tested on Windows with an environment created from the requirements.txt file. Python 3.8.

### Commands

Training: `main.py --settings Heston_kappa1` (parameter "settings" is optional. If not provided, uses the settings in settings.json)

Validation: `testing.py --validate --model Heston_kappa1` (reads validation result files and returns the best state of the model)

Testing: `testing.py --test --model Heston_kappa1_2000` (use the model name given by running the validation script)






Markdown
# README: How to Run the Code

## Installation
Ensure you have the required libraries installed before running any version:
```bash
pip install -r requirements.txt
VERSION 1: Crypto Deep Hedging (Base Model)
Objective: Hedge Bitcoin options using base TD3.

1. Train the model

Bash
python main_crypto.py --settings Crypto
2. Validate the models

Bash
python testing_crypto.py --validate --model Crypto
3. Test the best model (e.g., Crypto_18000)

Bash
python testing_crypto.py --test --model Crypto_18000
4. Plot the results

Bash
python plot_crypto.py Crypto_18000
VERSION 2: Crypto V2 (Quant Upgraded)
Objective: Improved hedging using Welford Variance, Sortino Penalty, and Heston Randomization.

1. Train the model

Bash
python main_crypto_v2.py --settings CryptoV2
2. Validate the models

Bash
python testing_crypto_v2.py --validate --model CryptoV2
3. Test the best model (e.g., CryptoV2_12000 or CryptoV2_16000)

Bash
python testing_crypto_v2.py --test --model CryptoV2_12000
4. Plot the results

Bash
python plot_crypto_v2.py CryptoV2_12000
VERSION 3: The Transformer Experiment
Objective: Testing Self-Attention mechanisms on NSE data (Batch Size 128).

1. Train the model

Bash
python main_transformer.py
2. Evaluate the results

Bash
python -c "import test_transformer; test_transformer.result_eval('NSE_7000_results')"
VERSION 4: NSE Base Model (Indian Equities)
Objective: Isolated pipeline for NSE without advanced risk logic (Kappa Stress Test).

1. Train the model

Bash
python main_nse.py --settings Nse
2. Validate the models

Bash
python testing_nse.py --validate --model NSE
3. Test the best model (e.g., NSE_13000)

Bash
python testing_nse.py --test --model NSE_13000
4. Plot the results

Bash
python plot_nse.py NSE_13000
VERSION 5: NSE V2 (The Final "Triple Crown" Architecture)
Objective: The ultimate model with PnL Normalization, Log-Euler Maruyama, and Full Welford Variance (Kappa = 225).

1. Train the model

Bash
python main_nse_v2.py --settings Nse_V2
2. Validate the models

Bash
python testing_nse_v2.py --validate --model Nse_V2
3. Test the best model (e.g., Nse_V2_3000)

Bash
python testing_nse_v2.py --test --model Nse_V2_3000
4. Plot the results

Bash
python plot_nse_v2.py Nse_V2_3000






Empirical Deep Hedging using Reinforcement Learning (TD3)

This repository contains the complete, reproducible code for training and testing a Deep Reinforcement Learning agent to hedge European Call Options in the Indian Equity (NSE) and Cryptocurrency markets.

Required Libraries & Packages

This project requires Python 3.8+ and the following specific package versions to ensure full reproducibility. These are listed in the requirements.txt file:

joblib==1.4.2
matplotlib==3.10.9
numpy==2.4.4
pandas==3.0.2
QuantLib==1.42.1
Requests==2.33.1
scikit_learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
torch==2.9.1
yfinance==1.3.0


Setup Instructions & Dependencies

Extract the ZIP file and navigate into the Code folder using your terminal/command prompt.

Set up a virtual environment (recommended):

python -m venv deep_hedging_env
source deep_hedging_env/bin/activate  # On Windows use: deep_hedging_env\Scripts\activate


Install the dependencies:

pip install -r requirements.txt


Ensure Data is Present: Verify that your historical data CSV files (if any) are located in the data/ folder and your JSON configuration files are in the settings/ folder before running the scripts.

Steps to Run the Code

The project is divided into 5 distinct architectural versions. Run the commands below in your terminal depending on which phase of the research you wish to reproduce.

VERSION 1: Crypto Deep Hedging (Base Model)

Objective: Hedge Bitcoin options using base TD3.

1. Train the model

python main_crypto.py --settings Crypto


2. Validate the models

python testing_crypto.py --validate --model Crypto


3. Test the best model (e.g., Crypto_18000)

python testing_crypto.py --test --model Crypto_18000


4. Plot the results

python plot_crypto.py Crypto_18000


VERSION 2: Crypto V2 (Quant Upgraded)

Objective: Improved hedging using Welford Variance, Sortino Penalty, and Heston Randomization.

1. Train the model

python main_crypto_v2.py --settings CryptoV2


2. Validate the models

python testing_crypto_v2.py --validate --model CryptoV2


3. Test the best model (e.g., CryptoV2_12000 or CryptoV2_16000)

python testing_crypto_v2.py --test --model CryptoV2_12000


4. Plot the results

python plot_crypto_v2.py CryptoV2_12000


VERSION 3: The Transformer Experiment

Objective: Testing Self-Attention mechanisms on NSE data (Batch Size 128).

1. Train the model

python main_transformer.py


2. Evaluate the results

python -c "import test_transformer; test_transformer.result_eval('NSE_7000_results')"


VERSION 4: NSE Base Model (Indian Equities)

Objective: Isolated pipeline for NSE without advanced risk logic (Kappa Stress Test).

1. Train the model

python main_nse.py --settings Nse


2. Validate the models

python testing_nse.py --validate --model NSE


3. Test the best model (e.g., NSE_13000)

python testing_nse.py --test --model NSE_13000


4. Plot the results

python plot_nse.py NSE_13000


VERSION 5: NSE V2 (The Final "Triple Crown" Architecture)

Objective: The ultimate model with PnL Normalization, Log-Euler Maruyama, and Full Welford Variance (Kappa = 225).

1. Train the model

python main_nse_v2.py --settings Nse_V2


2. Validate the models

python testing_nse_v2.py --validate --model Nse_V2


3. Test the best model (e.g., Nse_V2_3000)

python testing_nse_v2.py --test --model Nse_V2_3000


4. Plot the results

python plot_nse_v2.py Nse_V2_3000
