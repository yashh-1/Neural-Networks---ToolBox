# Neural Network Toolbox

An interactive, educational Streamlit application that visualizes and trains various neural network architectures from scratch.

## Features
- **Backpropagation**: Multilayer Perceptron for simple binary classification.
- **Perceptron**: Single-layer perceptron for linearly separable logic.
- **RNN & LSTM**: Recurrent networks for binary sentiment analysis.
- **Linear Regression**: Mean Squared Error (MSE) loss visualizations in 2D and 3D.
- **CNN**: Convolutional Neural Network for face detection and classification.
- **Hopfield Network**: Recurrent associative memory for alphabet recognition.

## Project Structure
```
NN_ToolBox/
├── src/                    # Source code
│   ├── app.py              # Main Streamlit application entry point
│   ├── models/             # Neural network implementations from scratch
│   └── utils/              # Visualization and helper scripts
├── data/                   # Sample datasets for training
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. **Ensure you have Python 3.9+ installed.**
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run src/app.py
   ```
4. Open the provided local URL (usually `http://localhost:8501`) in your browser to interact with the application.

## Sample Data
The `data/` directory contains sample datasets that you can use directly from the application interface:
- `sample_data.csv`: Simple 2-feature dataset for Backpropagation and Perceptron.
- `sample_sentiment.csv`: Text data for RNN and LSTM models.
- `sample_music_genre.csv`: Alternate text data for sequence models.
