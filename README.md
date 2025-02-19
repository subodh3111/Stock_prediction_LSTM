# Stock Forecasting Using LSTM

## Project Overview
This project utilizes Long Short-Term Memory (LSTM) neural networks to predict Apple stock prices using historical data from 2020 to febuary 2025. LSTMs are well-suited for time-series forecasting due to their ability to capture long-term dependencies and trends in sequential data.

## Dataset
- **Source**: Apple stock data
- **Time Range**: 2020 - 2025
- **Features**: Open, High, Low, Close, Volume
- **Target**: Future stock price prediction

## Technologies Used
- Python
- Pandas, NumPy (Data Processing)
- Matplotlib, Seaborn (Data Visualization)
- TensorFlow, Keras (LSTM Model Implementation)
- Scikit-learn (Data Scaling and Evaluation)

## Project Workflow
1. **Data Collection**: Apple stock price data collected from 2020 to 2025.
2. **Data Preprocessing**:
   - Handling missing values
   - Normalization using MinMaxScaler
   - Creating time-series sequences for LSTM
3. **Model Training**:
   - Building and training an LSTM-based neural network
   - Tuning hyperparameters for optimal performance
4. **Evaluation**:
   - Measuring performance using RMSE and MAPE
   - Visualizing predictions vs actual stock prices
5. **Prediction**:
   - Forecasting future Apple stock prices

## How to Run the Notebook
1. Install dependencies using:
   ```sh
   pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
   ```
2. Open the Jupyter Notebook and run each cell sequentially.
3. View visualizations and predicted stock price trends.

## Results
The trained LSTM model provides insights into future Apple stock prices by learning patterns from historical data. The accuracy and reliability of the forecast depend on the quality and amount of training data.

## Future Enhancements
- Use additional technical indicators for better predictions.
- Implement bidirectional LSTMs or Transformer models.
- Deploy the model as a web application.

## Author
Subodh Kumar

