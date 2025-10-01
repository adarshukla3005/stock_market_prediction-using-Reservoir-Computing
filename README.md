# Stock Market Price Prediction Using Different Neural Network Architectures

This project implements and compares different neural network architectures (LSTM, RNN, and Echo State Network/Reservoir Computing) for predicting stock market prices. The implementation focuses on NASDAQ (^IXIC) and Dow Jones (^DJI) indices.

<img width="680" height="283" alt="image" src="https://github.com/user-attachments/assets/9439d115-4a26-4b02-b4e3-29ebfb5231d1" />

<img width="680" height="283" alt="image" src="https://github.com/user-attachments/assets/49af40d3-2300-471d-a695-034ccd58e969" />

## Features

- Data acquisition using yfinance API
- Comprehensive data preprocessing and visualization
- Implementation of three different neural network architectures:
  - Long Short-Term Memory (LSTM)
  - Simple Recurrent Neural Network (RNN)
  - Echo State Network (ESN) / Reservoir Computing
- Performance comparison using multiple metrics
- Visualization of predictions vs actual prices

## Requirements

- Python 3.x
- Required packages:
  - yfinance
  - tensorflow
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - mplfinance
  - pyESN (included as submodule)

## Project Structure

- `stock_reservoir.ipynb`: Implementation for Dow Jones (^DJI) index
- `stock_reservoir_2.ipynb`: Implementation for NASDAQ (^IXIC) index
- `pyESN/`: Echo State Network implementation submodule

## Data Processing

- Historical data from 2009-12-31 to 2025-12-28
- Features: Open, High, Low, Close, Adjusted Close, Volume
- Data normalization using MinMaxScaler
- 80-20 train-test split
- Time step of 60 days for sequence creation

## Model Architectures

### 1. LSTM Network
- Two LSTM layers (50 units each)
- Dense layers for dimensionality reduction
- Adam optimizer with MSE loss

### 2. Simple RNN
- Single RNN layer (50 units)
- Dense layers for output
- Tanh activation
- Adam optimizer with MSE loss

### 3. Echo State Network (Reservoir Computing)
- Input dimension: 60
- Reservoir size: 110-190 neurons
- Sparsity: 0.4
- Spectral radius: 0.9
- Small noise addition (1e-6)

## Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Visualizations

- Historical close price trends
- OHLC (Open-High-Low-Close) plots
- Moving averages (50 and 200 days)
- Candlestick charts with volume
- Actual vs predicted prices for each model

## Analysis Tools

- Moving average calculations (50-day and 200-day)
- Candlestick pattern visualization
- Volume analysis
- Price trend visualization

## Usage

1. Clone the repository:
```bash
git clone [https://github.com/adarshukla3005/stock_market_prediction-using-Reservoir-Computing.git]
git submodule update --init --recursive
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebooks:
```bash
jupyter notebook
```

## Results

The project implements and compares three different neural network architectures for stock price prediction. Each architecture has its own strengths:

- LSTM: Good at capturing long-term dependencies
- RNN: Efficient for shorter sequences
- ESN: Fast training with competitive performance

Detailed performance metrics and visualizations are available in the respective notebooks.

## References

- pyESN implementation: [https://github.com/sunone5/pyESN/](https://github.com/sunone5/pyESN/)

- Yahoo Finance API documentation
