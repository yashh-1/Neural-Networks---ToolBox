import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os

# Set up absolute path for data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

from models.backprop import train_network, predict, DEFAULT_LR, DEFAULT_EPOCHS
from models.perceptron import train_perceptron, predict_perceptron, DEFAULT_LR as P_DEFAULT_LR, DEFAULT_EPOCHS as P_DEFAULT_EPOCHS
from models.rnn import train_rnn, DEFAULT_HIDDEN as R_DEFAULT_HIDDEN, DEFAULT_LR as R_DEFAULT_LR, DEFAULT_EPOCHS as R_DEFAULT_EPOCHS
from models.lstm import train_lstm, DEFAULT_HIDDEN as L_DEFAULT_HIDDEN, DEFAULT_LR as L_DEFAULT_LR, DEFAULT_EPOCHS as L_DEFAULT_EPOCHS
from models.mse import train_mse_single, predict_single, train_mse_dual, predict_dual, DEFAULT_LR as M_DEFAULT_LR, DEFAULT_EPOCHS as M_DEFAULT_EPOCHS
from models.cnn import train_cnn, preprocess_image, detect_face_pil, augment_image, FaceCNN, DEFAULT_LR as C_DEFAULT_LR, DEFAULT_EPOCHS as C_DEFAULT_EPOCHS, IMG_SIZE
from models.hopfield import HopfieldNetwork, ALPHABET_PATTERNS, GRID_SIZE, PATTERN_SIZE, get_pattern_vector
from utils.visualizations import (
    plot_decision_boundary, plot_confidence_heatmap, plot_weight_heatmap_mlp,
    plot_confusion_matrix, plot_activation_distribution, plot_loss_curve,
    plot_regression_line, plot_regression_3d, plot_perceptron_boundary,
    plot_residual, plot_sentiment_distribution, plot_loss_accuracy, plot_word_frequency
)


def backprop_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1, label: 'x1\\n(Input 1)', color: '#4CAF50', font: {color: 'white'}, group: 'input', x: -300, y: -100},
                {id: 2, label: 'x2\\n(Input 2)', color: '#4CAF50', font: {color: 'white'}, group: 'input', x: -300, y: 100},
                {id: 3, label: 'h1\\n(Hidden)', color: '#2196F3', font: {color: 'white'}, group: 'hidden', x: 0, y: -100},
                {id: 4, label: 'h2\\n(Hidden)', color: '#2196F3', font: {color: 'white'}, group: 'hidden', x: 0, y: 100},
                {id: 5, label: 'o\\n(Output)', color: '#FF5722', font: {color: 'white'}, group: 'output', x: 300, y: 0},
                {id: 6, label: 'bh1', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: 0, y: -250},
                {id: 7, label: 'bh2', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: 0, y: 250},
                {id: 8, label: 'bo', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: 300, y: -150}
            ]);

            var edges = new vis.DataSet([
                {from: 1, to: 3, label: 'w1', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 1, to: 4, label: 'w2', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 2, to: 3, label: 'w3', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 2, to: 4, label: 'w4', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 3, to: 5, label: 'w5', color: '#FF9800', font: {color: '#fff', strokeWidth: 0}, width: 2},
                {from: 4, to: 5, label: 'w6', color: '#FF9800', font: {color: '#fff', strokeWidth: 0}, width: 2},
                {from: 6, to: 3, color: '#666', dashes: true},
                {from: 7, to: 4, color: '#666', dashes: true},
                {from: 8, to: 5, color: '#666', dashes: true}
            ]);

            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 35, font: {size: 12} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


def perceptron_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1, label: 'x1\\n(Input 1)', color: '#4CAF50', font: {color: 'white'}, x: -300, y: -100},
                {id: 2, label: 'x2\\n(Input 2)', color: '#4CAF50', font: {color: 'white'}, x: -300, y: 100},
                {id: 3, label: 'bias\\n(1)', color: '#9E9E9E', font: {color: 'white'}, x: -300, y: 0},
                {id: 4, label: 'Σ\\n(Sum)', color: '#2196F3', font: {color: 'white'}, size: 40, x: 0, y: 0},
                {id: 5, label: 'Step\\nFunction', color: '#FF9800', font: {color: 'white'}, shape: 'box', x: 200, y: 0},
                {id: 6, label: 'y\\n(Output)', color: '#E91E63', font: {color: 'white'}, x: 400, y: 0}
            ]);

            var edges = new vis.DataSet([
                {from: 1, to: 4, label: 'w1', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 2, to: 4, label: 'w2', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 3, to: 4, label: 'b', color: '#666', font: {color: '#fff', strokeWidth: 0}, dashes: true},
                {from: 4, to: 5, color: '#FF9800', width: 2},
                {from: 5, to: 6, label: '0 or 1', color: '#E91E63', font: {color: '#fff', strokeWidth: 0}, width: 2}
            ]);

            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 35, font: {size: 12} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """

def rnn_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1,  label: 'x₁\\n(word 1)', color: '#4CAF50', font: {color: 'white'}, x: -400, y: 0},
                {id: 2,  label: 'x₂\\n(word 2)', color: '#4CAF50', font: {color: 'white'}, x: -200, y: 0},
                {id: 3,  label: 'x₃\\n(word 3)', color: '#4CAF50', font: {color: 'white'}, x: 0, y: 0},
                {id: 4,  label: '...', color: '#666', font: {color: 'white'}, shape: 'text', x: 150, y: 0},
                {id: 5,  label: 'xₜ\\n(last)', color: '#4CAF50', font: {color: 'white'}, x: 300, y: 0},
                {id: 11, label: 'h₁', color: '#2196F3', font: {color: 'white'}, x: -400, y: -150},
                {id: 12, label: 'h₂', color: '#2196F3', font: {color: 'white'}, x: -200, y: -150},
                {id: 13, label: 'h₃', color: '#2196F3', font: {color: 'white'}, x: 0, y: -150},
                {id: 15, label: 'hₜ', color: '#2196F3', font: {color: 'white'}, x: 300, y: -150},
                {id: 20, label: 'σ(y)\\nOutput', color: '#FF5722', font: {color: 'white'}, size: 40, x: 300, y: -300},
                {id: 30, label: 'Embed', color: '#9C27B0', font: {color: 'white'}, shape: 'box', x: -500, y: 0}
            ]);
            var edges = new vis.DataSet([
                {from: 1, to: 11, label: 'Wₓₕ', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 2, to: 12, label: 'Wₓₕ', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 3, to: 13, label: 'Wₓₕ', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 5, to: 15, label: 'Wₓₕ', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 11, to: 12, label: 'Wₕₕ', color: '#FF9800', font: {color:'#fff',strokeWidth:0}, width: 2},
                {from: 12, to: 13, label: 'Wₕₕ', color: '#FF9800', font: {color:'#fff',strokeWidth:0}, width: 2},
                {from: 13, to: 15, label: 'Wₕₕ', color: '#FF9800', font: {color:'#fff',strokeWidth:0}, width: 2, dashes: true},
                {from: 15, to: 20, label: 'Wₕᵧ', color: '#E91E63', font: {color:'#fff',strokeWidth:0}, width: 2},
                {from: 30, to: 1, color: '#9C27B0', dashes: true}
            ]);
            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 35, font: {size: 12} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


def lstm_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1, label: 'x_t\\n(Input)', color: '#4CAF50', font: {color: 'white'}, x: -350, y: 0},
                {id: 2, label: 'Embed', color: '#66BB6A', font: {color: 'white'}, shape: 'box', x: -200, y: 0},
                {id: 3, label: 'f_t\\n(Forget)', color: '#F44336', font: {color: 'white'}, x: 0, y: -150},
                {id: 4, label: 'i_t\\n(Input)', color: '#2196F3', font: {color: 'white'}, x: 0, y: -50},
                {id: 5, label: 'g_t\\n(Candidate)', color: '#9C27B0', font: {color: 'white'}, x: 0, y: 50},
                {id: 6, label: 'o_t\\n(Output)', color: '#FF9800', font: {color: 'white'}, x: 0, y: 150},
                {id: 7, label: 'c_t\\n(Cell)', color: '#607D8B', font: {color: 'white'}, size: 40, x: 200, y: 0},
                {id: 8, label: 'h_t\\n(Hidden)', color: '#E91E63', font: {color: 'white'}, size: 40, x: 350, y: 0},
                {id: 9, label: 'y\\n(Output)', color: '#FF5722', font: {color: 'white'}, x: 500, y: 0},
                {id: 10, label: 'h_{t-1}', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: -200, y: 200},
                {id: 11, label: 'c_{t-1}', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: -200, y: -200}
            ]);

            var edges = new vis.DataSet([
                {from: 1, to: 2, color: '#aaa'},
                {from: 2, to: 3, color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 2, to: 4, color: '#aaa'},
                {from: 2, to: 5, color: '#aaa'},
                {from: 2, to: 6, color: '#aaa'},
                {from: 10, to: 3, color: '#666', dashes: true},
                {from: 10, to: 4, color: '#666', dashes: true},
                {from: 10, to: 5, color: '#666', dashes: true},
                {from: 10, to: 6, color: '#666', dashes: true},
                {from: 11, to: 7, label: 'f*c', color: '#F44336', font: {color: '#fff', strokeWidth: 0}},
                {from: 3, to: 7, label: 'forget', color: '#F44336', font: {color: '#fff', strokeWidth: 0}},
                {from: 4, to: 7, label: 'i*g', color: '#2196F3', font: {color: '#fff', strokeWidth: 0}},
                {from: 5, to: 7, color: '#9C27B0'},
                {from: 6, to: 8, label: 'gate', color: '#FF9800', font: {color: '#fff', strokeWidth: 0}},
                {from: 7, to: 8, label: 'tanh', color: '#607D8B', font: {color: '#fff', strokeWidth: 0}, width: 2},
                {from: 8, to: 9, label: 'W_hy', color: '#FF5722', font: {color: '#fff', strokeWidth: 0}, width: 2}
            ]);

            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 30, font: {size: 11} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


def mse_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1, label: 'x₁\\n(Feature 1)', color: '#4CAF50', font: {color: 'white'}, x: -300, y: -100},
                {id: 2, label: 'x₂\\n(Feature 2)', color: '#4CAF50', font: {color: 'white'}, x: -300, y: 100},
                {id: 3, label: 'Σ\\n(Weighted Sum)', color: '#2196F3', font: {color: 'white'}, size: 45, x: 50, y: 0},
                {id: 4, label: 'ŷ\\n(Prediction)', color: '#FF9800', font: {color: 'white'}, x: 300, y: 0},
                {id: 5, label: 'MSE\\nLoss', color: '#F44336', font: {color: 'white'}, shape: 'box', x: 500, y: 0},
                {id: 6, label: 'y\\n(True)', color: '#9C27B0', font: {color: 'white'}, x: 500, y: 150},
                {id: 7, label: 'bias', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: 50, y: 200}
            ]);
            var edges = new vis.DataSet([
                {from: 1, to: 3, label: 'w₁', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 2, to: 3, label: 'w₂', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 7, to: 3, label: 'b', color: '#666', font: {color:'#fff',strokeWidth:0}, dashes: true},
                {from: 3, to: 4, color: '#FF9800', width: 2},
                {from: 4, to: 5, label: '(ŷ-y)²', color: '#F44336', font: {color:'#fff',strokeWidth:0}, width: 2},
                {from: 6, to: 5, color: '#9C27B0', dashes: true}
            ]);
            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 35, font: {size: 12} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


def cnn_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1, label: 'Input\\n32x32x1', color: '#4CAF50', font: {color: 'white'}, shape: 'box', x: -400, y: 0},
                {id: 2, label: 'Conv1\\n3x3, 8 filters', color: '#2196F3', font: {color: 'white'}, shape: 'box', x: -250, y: 0},
                {id: 3, label: 'ReLU', color: '#FF9800', font: {color: 'white'}, shape: 'box', x: -130, y: 0},
                {id: 4, label: 'MaxPool\\n2x2', color: '#9C27B0', font: {color: 'white'}, shape: 'box', x: -10, y: 0},
                {id: 5, label: 'Conv2\\n3x3, 16 filters', color: '#2196F3', font: {color: 'white'}, shape: 'box', x: 130, y: 0},
                {id: 6, label: 'ReLU', color: '#FF9800', font: {color: 'white'}, shape: 'box', x: 240, y: 0},
                {id: 7, label: 'MaxPool\\n2x2', color: '#9C27B0', font: {color: 'white'}, shape: 'box', x: 350, y: 0},
                {id: 8, label: 'Flatten', color: '#607D8B', font: {color: 'white'}, shape: 'box', x: 460, y: 0},
                {id: 9, label: 'FC1\\n(64)', color: '#E91E63', font: {color: 'white'}, shape: 'box', x: 570, y: 0},
                {id: 10, label: 'ReLU', color: '#FF9800', font: {color: 'white'}, shape: 'box', x: 660, y: 0},
                {id: 11, label: 'FC2\\n(classes)', color: '#F44336', font: {color: 'white'}, shape: 'box', x: 770, y: 0},
                {id: 12, label: 'Softmax', color: '#00BCD4', font: {color: 'white'}, shape: 'box', x: 880, y: 0}
            ]);
            var edges = new vis.DataSet([
                {from: 1, to: 2, color: '#aaa'},
                {from: 2, to: 3, color: '#aaa'},
                {from: 3, to: 4, color: '#aaa'},
                {from: 4, to: 5, color: '#aaa'},
                {from: 5, to: 6, color: '#aaa'},
                {from: 6, to: 7, color: '#aaa'},
                {from: 7, to: 8, color: '#aaa'},
                {from: 8, to: 9, color: '#aaa'},
                {from: 9, to: 10, color: '#aaa'},
                {from: 10, to: 11, color: '#aaa'},
                {from: 11, to: 12, color: '#aaa'}
            ]);
            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'box', size: 30, font: {size: 11} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.05} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


st.set_page_config(page_title="Neural Network Toolbox", layout="wide")

# ── Theme Toggle ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme = st.sidebar.toggle("Light Mode", value=(st.session_state.theme == "Light"), key="theme_toggle")
st.session_state.theme = "Light" if theme else "Dark"

if st.session_state.theme == "Light":
    st.markdown("""<style>
        :root {
            --bg: #ffffff; --text: #111111; --card: #f8f9fa; --border: #dee2e6;
        }
        .stApp { background-color: var(--bg); color: var(--text); }
        .stMarkdown, .stText, h1, h2, h3, p, span, label { color: var(--text) !important; }
        .stDataFrame { background: var(--card); }
        div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
        div[data-testid="stMetricValue"] { color: var(--text) !important; }
        section[data-testid="stSidebar"] { background: #f0f2f6; }
        section[data-testid="stSidebar"] * { color: #111 !important; }
        .stTabs [data-baseweb="tab"] { color: var(--text) !important; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        :root {
            --bg: #0e1117; --text: #fafafa; --card: #1a1d24; --border: #333;
        }
        .stApp { background-color: var(--bg); color: var(--text); }
        div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
    </style>""", unsafe_allow_html=True)

st.title("Neural Network Toolbox")

# Session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
    st.session_state.weights = None
    st.session_state.loss_history = None
    st.session_state.model_type = None

_t = st.session_state.theme  # shorthand for passing theme to visualizations

# Sidebar - Model Selection
st.sidebar.divider()
st.sidebar.header("Select Model")
st.markdown("""<style>div[data-baseweb="select"] input {caret-color: transparent !important;}</style>""", unsafe_allow_html=True)
model_type = st.sidebar.selectbox(
    "Choose Neural Network Type",
    ["Backpropagation", "Perceptron", "RNN (Sentiment Analysis)", "LSTM (Sentiment Analysis)", "MSE Loss (Linear Regression)", "CNN (Face Classification)", "Hopfield Network (Alphabet)"]
)

st.sidebar.divider()

# ============================================
# BACKPROPAGATION — Training Algorithm
# ============================================
if model_type == "Backpropagation":
    st.sidebar.header("Backprop Parameters")
    st.sidebar.subheader("Initial Weights")
    st.sidebar.info("Random (generated on each training run)")
    b_lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=DEFAULT_LR, step=0.0001, format="%.5f", key="bp_lr")
    b_epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=5000, value=DEFAULT_EPOCHS, step=50, key="bp_epochs")

    st.header("Backpropagation — Training Algorithm")
    st.caption("The algorithm that teaches neural networks by propagating errors backward")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Train", "Predict", "Step-by-Step Trace", "Visualizations", "Algorithm"])

    with tab1:
        st.subheader("Step 1: Upload CSV File")
        st.info("CSV format: First 2 columns = features, Last column = label (0 or 1)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="bp_upload")
        use_sample = st.checkbox("Use sample data instead", key="bp_sample")

        if use_sample:
            df = pd.read_csv(os.path.join(DATA_DIR, "sample_data.csv"))
            st.write("Sample Data (Student Performance):")
            st.dataframe(df, use_container_width=True)
            st.session_state.bp_df = df
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df, use_container_width=True)
            st.session_state.bp_df = df

        if "bp_df" in st.session_state:
            df = st.session_state.bp_df
            raw_X = df.iloc[:, :2].values
            x_min = raw_X.min(axis=0)
            x_max = raw_X.max(axis=0)
            X = ((raw_X - x_min) / (x_max - x_min + 1e-8)).tolist()
            Y = df.iloc[:, -1].values.tolist()
            st.session_state.col_names = [df.columns[0], df.columns[1]]
            st.session_state.bp_x_min = x_min
            st.session_state.bp_x_max = x_max
        else:
            X = None
            Y = None

        st.subheader("Step 2: Train with Backpropagation")

        if X is not None and Y is not None:
            if st.button("Train", type="primary", key="bp_train"):
                with st.spinner("Training with backpropagation..."):
                    weights, loss_history, init_weights = train_network(X, Y, l_rate=b_lr, n_epochs=b_epochs)
                    st.session_state.trained = True
                    st.session_state.weights = weights
                    st.session_state.loss_history = loss_history
                    st.session_state.model_type = "backprop"
                    st.session_state.bp_init_weights = init_weights
                    st.session_state.bp_X = X
                    st.session_state.bp_Y = Y

                st.success("Training Complete!")

                iw1, iw2, iw3, iw4, iw5, iw6, _, _, _ = init_weights
                st.subheader("Initial → Trained Weights")
                w1, w2, w3, w4, w5, w6, bh1, bh2, bo = weights
                weight_df = pd.DataFrame({
                    "Weight": ["w1", "w2", "w3", "w4", "w5", "w6", "bh1", "bh2", "bo"],
                    "Initial": [iw1, iw2, iw3, iw4, iw5, iw6, 0.0, 0.0, 0.0],
                    "Trained": [w1, w2, w3, w4, w5, w6, bh1, bh2, bo]
                })
                st.dataframe(weight_df, use_container_width=True)

                st.subheader("Loss Curve (Error Reduction Over Time)")
                loss_df = pd.DataFrame({'Epoch': range(1, len(loss_history) + 1), 'Total Error': loss_history})
                st.line_chart(loss_df.set_index('Epoch'))
                st.caption("Backpropagation minimizes this error by adjusting weights after each sample.")
        else:
            st.warning("Please upload a CSV file or use sample data to train.")

    with tab2:
        st.subheader("Predict Output")
        
        if st.session_state.trained and st.session_state.model_type == "backprop":
            st.success("Model is trained and ready!")
            col_names = st.session_state.get('col_names', ['Feature 1', 'Feature 2'])

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                pred_x1 = st.number_input(col_names[0], value=0.5, key="bp_x1")
            with pcol2:
                pred_x2 = st.number_input(col_names[1], value=0.5, key="bp_x2")

            if st.button("Predict", type="primary", key="bp_predict"):
                # Scale input if needed based on training data ranges
                if "bp_x_min" in st.session_state and "bp_x_max" in st.session_state:
                    xmin = st.session_state.bp_x_min
                    xmax = st.session_state.bp_x_max
                    sc_x1 = (pred_x1 - xmin[0]) / (xmax[0] - xmin[0] + 1e-8)
                    sc_x2 = (pred_x2 - xmin[1]) / (xmax[1] - xmin[1] + 1e-8)
                else:
                    sc_x1, sc_x2 = pred_x1, pred_x2

                score, pred = predict(sc_x1, sc_x2, st.session_state.weights)

                st.subheader("Prediction Result")
                st.metric("Model Output (Activation)", f"{score:.6f}")
                st.metric("Predicted Class", "Class 1" if pred == 1 else "Class 0")
        else:
            st.warning("Please train the backpropagation model first.")

    with tab3:
        st.subheader("Step-by-Step Trace (1 Sample)")
        st.markdown("See exactly how backpropagation processes **one data point**:")

        if st.session_state.get("model_type") == "backprop" and st.session_state.get("trained"):
            weights = st.session_state.weights
            w1, w2, w3, w4, w5, w6, bh1, bh2, bo = weights
            X = st.session_state.bp_X
            Y = st.session_state.bp_Y

            sample_idx = st.slider("Pick a sample", 0, len(X) - 1, 0, key="bp_trace_idx")
            x1, x2 = X[sample_idx]
            target = Y[sample_idx]
            st.write(f"**Input:** x1 = {x1}, x2 = {x2} | **Target:** {target}")

            import math
            st.divider()
            st.markdown("**1. Forward Pass — Hidden Layer:**")
            zh1 = x1 * w1 + x2 * w3 + bh1
            zh2 = x1 * w2 + x2 * w4 + bh2
            h1 = 1 / (1 + math.exp(-zh1))
            h2 = 1 / (1 + math.exp(-zh2))
            st.code(f"zh1 = {x1}*{w1:.4f} + {x2}*{w3:.4f} + {bh1:.4f} = {zh1:.4f}\nzh2 = {x1}*{w2:.4f} + {x2}*{w4:.4f} + {bh2:.4f} = {zh2:.4f}\nh1 = sigmoid({zh1:.4f}) = {h1:.4f}\nh2 = sigmoid({zh2:.4f}) = {h2:.4f}")

            st.markdown("**2. Forward Pass — Output:**")
            zo = h1 * w5 + h2 * w6 + bo
            o = 1 / (1 + math.exp(-zo))
            st.code(f"zo = {h1:.4f}*{w5:.4f} + {h2:.4f}*{w6:.4f} + {bo:.4f} = {zo:.4f}\no  = sigmoid({zo:.4f}) = {o:.4f}")

            st.markdown("**3. Error:**")
            error = target - o
            st.code(f"error = {target} - {o:.4f} = {error:.4f}")

            st.markdown("**4. Backpropagation — Compute Deltas:**")
            delta_o = error * o * (1 - o)
            delta_h1 = delta_o * w5 * h1 * (1 - h1)
            delta_h2 = delta_o * w6 * h2 * (1 - h2)
            st.code(f"delta_o  = {error:.4f} * {o:.4f} * {1-o:.4f} = {delta_o:.6f}\ndelta_h1 = {delta_o:.6f} * {w5:.4f} * {h1:.4f} * {1-h1:.4f} = {delta_h1:.6f}\ndelta_h2 = {delta_o:.6f} * {w6:.4f} * {h2:.4f} * {1-h2:.4f} = {delta_h2:.6f}")

            st.markdown("**5. Weight Updates:**")
            lr = b_lr
            st.code(f"w5_new = {w5:.4f} + {lr} * {delta_o:.6f} * {h1:.4f} = {w5 + lr*delta_o*h1:.6f}\nw6_new = {w6:.4f} + {lr} * {delta_o:.6f} * {h2:.4f} = {w6 + lr*delta_o*h2:.6f}\nw1_new = {w1:.4f} + {lr} * {delta_h1:.6f} * {x1} = {w1 + lr*delta_h1*x1:.6f}")
        else:
            st.warning("Train the model first (go to Upload & Train tab).")

    with tab4:
        st.subheader("Visualizations")
        if st.session_state.get("model_type") == "backprop" and st.session_state.get("trained"):
            viz_type = st.selectbox("Select Visualization", [
                "Decision Boundary", "Confidence Heatmap", "Weight Heatmap",
                "Confusion Matrix", "Activation Distribution", "Loss Curve"
            ], key="bp_viz")

            weights = st.session_state.weights
            X_data = st.session_state.get("bp_X")
            Y_data = st.session_state.get("bp_Y")

            if viz_type == "Decision Boundary" and X_data:
                fig = plot_decision_boundary(X_data, Y_data, lambda x1, x2: predict(x1, x2, weights), "Backprop Decision Boundary", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Shows how backpropagation shaped the decision boundary over training.")

            elif viz_type == "Confidence Heatmap" and X_data:
                fig = plot_confidence_heatmap(X_data, Y_data, lambda x1, x2: predict(x1, x2, weights), "Backprop Confidence Map", theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Weight Heatmap":
                fig = plot_weight_heatmap_mlp(weights, "Trained Weight Heatmap", theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Confusion Matrix" and X_data:
                y_pred = [predict(x[0], x[1], weights)[1] for x in X_data]
                fig, metrics = plot_confusion_matrix(Y_data, y_pred, theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                mcol2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                mcol3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                mcol4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

            elif viz_type == "Activation Distribution" and X_data:
                fig = plot_activation_distribution(X_data, weights, theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Loss Curve" and st.session_state.get("loss_history"):
                fig = plot_loss_curve(st.session_state.loss_history, "Backprop Training Loss", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train with backpropagation first to see visualizations.")

    with tab5:
        st.subheader("Backpropagation Algorithm")
        st.caption("How neural networks learn by propagating errors backward through layers")
        components.html(backprop_diagram(), height=520)

        st.divider()
        st.subheader("What is Backpropagation?")
        st.markdown("""
**Backpropagation** is the training algorithm used to update weights in a neural network. It works by:
1. Running a **forward pass** to get a prediction
2. Computing the **error** (how wrong the prediction is)
3. Propagating the error **backward** through each layer
4. **Updating weights** using the chain rule of calculus

It is NOT a network architecture — it is the **learning algorithm** that can train any feedforward network (MLP, etc.).
""")

        st.divider()
        st.subheader("Step 1: Compute Error")
        st.latex(r"\text{error} = t - o")
        st.latex(r"\text{total\_error} = \sum_{i=1}^{n} (t_i - o_i)^2")
        st.code("""error = target - output
total_error += error ** 2""", language="python")

        st.divider()
        st.subheader("Step 2: Output Layer Gradient (Delta)")
        st.markdown("How much the output neuron contributed to the error:")
        st.latex(r"\delta_o = \text{error} \times o \times (1 - o)")
        st.markdown("This uses the **derivative of sigmoid**: $\\sigma'(x) = \\sigma(x)(1-\\sigma(x))$")
        st.code("delta_o = error * o * (1 - o)", language="python")

        st.divider()
        st.subheader("Step 3: Hidden Layer Gradients (Chain Rule)")
        st.markdown("Error is **propagated backward** through weights to hidden neurons:")
        st.latex(r"\delta_{h1} = \delta_o \times w_5 \times h_1 \times (1 - h_1)")
        st.latex(r"\delta_{h2} = \delta_o \times w_6 \times h_2 \times (1 - h_2)")
        st.markdown("This is the **chain rule** in action — each layer's error depends on the next layer's error.")
        st.code("""delta_h1 = delta_o * w5 * h1 * (1 - h1)
delta_h2 = delta_o * w6 * h2 * (1 - h2)""", language="python")

        st.divider()
        st.subheader("Step 4: Update All Weights")
        st.latex(r"w_{new} = w_{old} + \eta \times \delta \times \text{input}")
        st.markdown("Where $\\eta$ is the learning rate.")
        st.code("""# Output layer weights
w5 = w5 + lr * delta_o * h1
w6 = w6 + lr * delta_o * h2
bo = bo + lr * delta_o

# Hidden layer weights (error flows backward)
w1 = w1 + lr * delta_h1 * x1
w3 = w3 + lr * delta_h1 * x2
w2 = w2 + lr * delta_h2 * x1
w4 = w4 + lr * delta_h2 * x2""", language="python")

        st.divider()
        st.subheader("Why Backpropagation Works")
        st.markdown("""
| Concept | Role |
|---|---|
| **Chain Rule** | Breaks complex derivatives into layer-by-layer products |
| **Gradient Descent** | Uses gradients to move weights toward lower error |
| **Error Signal** | Flows **backward** from output → hidden → input layers |
| **Learning Rate (η)** | Controls step size — too high = overshoot, too low = slow |
""")

# ============================================
# PERCEPTRON
# ============================================
elif model_type == "Perceptron":
    st.header("Perceptron (Single Layer)")
    st.caption("Simple binary classifier with no hidden layers")

    st.sidebar.header("Perceptron Parameters")
    p_lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=P_DEFAULT_LR, step=0.01, format="%.5f", key="p_lr")
    p_epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=5000, value=P_DEFAULT_EPOCHS, step=10, key="p_epochs")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture"])

    with tab1:
        st.subheader("Step 1: Upload CSV File")
        st.info("CSV format: First 2 columns = features, Last column = label (0 or 1)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="perceptron_upload")
        use_sample = st.checkbox("Use sample data instead", key="perceptron_sample")

        if use_sample:
            df = pd.read_csv(os.path.join(DATA_DIR, "sample_data.csv"))
            st.write("Sample Data (Student Performance):")
            st.dataframe(df, use_container_width=True)
            st.session_state.perceptron_df = df
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df, use_container_width=True)
            st.session_state.perceptron_df = df

        if "perceptron_df" in st.session_state:
            df = st.session_state.perceptron_df
            raw_X = df.iloc[:, :2].values
            x_min = raw_X.min(axis=0)
            x_max = raw_X.max(axis=0)
            X = ((raw_X - x_min) / (x_max - x_min + 1e-8)).tolist()
            Y = df.iloc[:, -1].values.tolist()
            st.session_state.col_names = [df.columns[0], df.columns[1]]
            st.session_state.perceptron_x_min = x_min
            st.session_state.perceptron_x_max = x_max
        else:
            X = None
            Y = None

        st.subheader("Step 2: Train Perceptron")

        if X is not None and Y is not None:
            if st.button("Train Perceptron", type="primary"):
                with st.spinner("Training..."):
                    weights, loss_history = train_perceptron(X, Y, lr=p_lr, epochs=p_epochs)

                st.session_state.trained = True
                st.session_state.weights = weights
                st.session_state.loss_history = loss_history
                st.session_state.model_type = "perceptron"
                st.session_state.perceptron_X = X
                st.session_state.perceptron_Y = Y

                st.success("Training Complete!")

                w1, w2, b = weights
                st.subheader("Trained Weights")
                st.write(f"w1: {w1:.6f}")
                st.write(f"w2: {w2:.6f}")
                st.write(f"bias: {b:.6f}")

                st.subheader("Training Loss Over Epochs")
                loss_df = pd.DataFrame({'Epoch': range(1, len(loss_history) + 1), 'Errors': loss_history})
                st.line_chart(loss_df.set_index('Epoch'))
        else:
            st.warning("Please upload a CSV file or use sample data.")

    with tab2:
        st.subheader("Predict Output")

        if st.session_state.trained and st.session_state.model_type == "perceptron":
            st.success("Model is trained and ready!")
            col_names = st.session_state.get('col_names', ['Feature 1', 'Feature 2'])

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                pred_x1 = st.number_input(col_names[0], value=5.0, key="p_x1")
            with pcol2:
                pred_x2 = st.number_input(col_names[1], value=75.0, key="p_x2")

            if st.button("Predict", type="primary", key="p_predict"):
                z, pred = predict_perceptron(pred_x1, pred_x2, st.session_state.weights)

                st.subheader("Prediction Result")
                st.metric("Raw Output (z)", f"{z:.6f}")
                st.metric("Predicted Class", "Pass" if pred == 1 else "Fail")
        else:
            st.warning("Please train the perceptron first.")

    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("model_type") == "perceptron" and st.session_state.get("trained"):
            viz_type = st.selectbox("Select Visualization", [
                "Decision Boundary", "Confusion Matrix", "Loss Curve"
            ], key="p_viz")

            weights = st.session_state.weights
            X_data = st.session_state.get("perceptron_X")
            Y_data = st.session_state.get("perceptron_Y")

            if viz_type == "Decision Boundary" and X_data:
                fig = plot_perceptron_boundary(X_data, Y_data, weights, "Perceptron Decision Boundary", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The perceptron finds a single straight line to separate the two classes.")

            elif viz_type == "Confusion Matrix" and X_data:
                y_pred = [predict_perceptron(x[0], x[1], weights)[1] for x in X_data]
                fig, metrics = plot_confusion_matrix(Y_data, y_pred, "Perceptron Confusion Matrix", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                mcol2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                mcol3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                mcol4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

            elif viz_type == "Loss Curve" and st.session_state.get("loss_history"):
                fig = plot_loss_curve(st.session_state.loss_history, "Perceptron Training Errors", ylabel="Misclassifications", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the perceptron first to see visualizations.")

    with tab4:
        st.subheader("Network Diagram")
        st.caption("Drag nodes to move, scroll to zoom, drag background to pan")
        components.html(perceptron_diagram(), height=520)

        st.divider()
        st.subheader("Step 1: Input Preparation")
        st.markdown("Add bias column (always `1`) to inputs. Convert binary targets to **bipolar**:")
        st.latex(r"\text{target} = \begin{cases} +1 & \text{if } y = 1 \\ -1 & \text{if } y = 0 \end{cases}")
        st.code("""inputs = np.array([[x[0], x[1], 1] for x in X])
targets = np.array([1 if y == 1 else -1 for y in Y])
weights = np.array([0.0, 0.0, 0.0])""", language="python")

        st.divider()
        st.subheader("Step 2: Compute Weighted Sum")
        st.latex(r"Y_{in} = x_1 \cdot w_1 + x_2 \cdot w_2 + 1 \cdot w_b = \mathbf{x} \cdot \mathbf{w}")
        st.code("""Y_in = np.dot(x, weights)""", language="python")

        st.divider()
        st.subheader("Step 3: Step Activation Function")
        st.latex(r"Y_{out} = \begin{cases} 1 & \text{if } Y_{in} > 0 \\ 0 & \text{otherwise} \end{cases}")
        st.code("""if Y_in > 0:
    Y_out = 1
else:
    Y_out = 0""", language="python")

        st.divider()
        st.subheader("Step 4: Check Match (Bipolar Logic)")
        st.markdown("Compare prediction with bipolar target:")
        st.latex(r"\text{match} = (t = +1 \text{ and } Y_{out} = 1) \text{ or } (t = -1 \text{ and } Y_{out} = 0)")
        st.code("""is_match = (t == 1 and Y_out == 1) or (t == -1 and Y_out == 0)""", language="python")

        st.divider()
        st.subheader("Step 5: Update Weights (Only on Mismatch)")
        st.latex(r"\mathbf{w}_{new} = \mathbf{w}_{old} + \eta \cdot t \cdot \mathbf{x}")
        st.markdown("Where $\\eta$ is the learning rate and $t$ is the bipolar target.")
        st.code("""if not is_match:
    weights = weights + learning_rate * (t * x)""", language="python")

        st.divider()
        st.subheader("Final Decision Boundary")
        st.latex(r"w_1 \cdot x_1 + w_2 \cdot x_2 + w_b = 0")

# ============================================
# RNN (Sentiment Analysis)
# ============================================
elif model_type == "RNN (Sentiment Analysis)":
    st.header("RNN — Sentiment Analysis")
    st.caption("Vanilla RNN with word embeddings for binary text classification")

    st.sidebar.header("RNN Parameters")
    r_hidden = st.sidebar.number_input("Hidden Size", min_value=8, max_value=256, value=R_DEFAULT_HIDDEN, step=8, key="r_hidden")
    r_lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=R_DEFAULT_LR, step=0.001, format="%.4f", key="r_lr")
    r_epochs = st.sidebar.number_input("Epochs", min_value=5, max_value=500, value=R_DEFAULT_EPOCHS, step=5, key="r_epochs")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture"])

    # ── Tab 1: Upload & Train ────────────────────────────────
    with tab1:
        st.subheader("Step 1: Upload Sentiment CSV")
        st.info("CSV format: A **text** column and a **sentiment** column (0 = Negative, 1 = Positive)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="rnn_upload")
        use_sample = st.checkbox("Use sample sentiment data instead", key="rnn_sample")

        if use_sample:
            df = pd.read_csv(os.path.join(DATA_DIR, "sample_sentiment.csv"))
            st.write("Sample Sentiment Data:")
            st.dataframe(df, use_container_width=True)
            st.session_state.rnn_df = df
            st.session_state.rnn_text_col_name = "text"
            st.session_state.rnn_label_col_name = "sentiment"
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df, use_container_width=True)
            st.session_state.rnn_df = df
            text_col = st.selectbox("Select text column", df.columns, index=0, key="rnn_text_col")
            label_col = st.selectbox("Select label column (0/1)", df.columns, index=min(1, len(df.columns) - 1), key="rnn_label_col")
            st.session_state.rnn_text_col_name = text_col
            st.session_state.rnn_label_col_name = label_col

        if "rnn_df" in st.session_state:
            df = st.session_state.rnn_df
            texts = df[st.session_state.rnn_text_col_name].astype(str).tolist()
            labels = df[st.session_state.rnn_label_col_name].astype(int).tolist()
        else:
            texts = None
            labels = None

        st.subheader("Step 2: Train RNN")

        if texts is not None and labels is not None:
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            st.write(f"**Dataset:** {len(labels)} samples — {pos_count} Positive, {neg_count} Negative")

            if st.button("Train RNN", type="primary", key="rnn_train"):
                with st.spinner("Training RNN (this may take a moment)..."):
                    model, vocab, loss_hist, acc_hist = train_rnn(
                        texts, labels,
                        hidden_size=r_hidden, lr=r_lr, epochs=r_epochs
                    )
                    st.session_state.trained = True
                    st.session_state.rnn_model = model
                    st.session_state.rnn_vocab = vocab
                    st.session_state.loss_history = loss_hist
                    st.session_state.rnn_acc_history = acc_hist
                    st.session_state.model_type = "rnn"
                    st.session_state.rnn_texts = texts
                    st.session_state.rnn_labels = labels

                st.success("Training Complete!")

                st.write(f"**Vocab size:** {len(vocab)} words  |  **Hidden size:** {r_hidden}")

                col_l, col_r = st.columns(2)
                with col_l:
                    st.subheader("Training Loss")
                    loss_df = pd.DataFrame({"Epoch": range(1, len(loss_hist) + 1), "Loss": loss_hist})
                    st.line_chart(loss_df.set_index("Epoch"))
                with col_r:
                    st.subheader("Training Accuracy")
                    acc_df = pd.DataFrame({"Epoch": range(1, len(acc_hist) + 1), "Accuracy": acc_hist})
                    st.line_chart(acc_df.set_index("Epoch"))

                # Show final accuracy
                final_acc = acc_hist[-1] * 100
                st.metric("Final Training Accuracy", f"{final_acc:.1f}%")
        else:
            st.warning("Please upload a CSV file or use sample data.")

    # ── Tab 2: Predict ───────────────────────────────────────
    with tab2:
        st.subheader("Predict Sentiment")

        if st.session_state.get("model_type") == "rnn" and st.session_state.get("trained"):
            st.success("RNN model is trained and ready!")

            input_text = st.text_area("Enter text to analyse:", height=100, key="rnn_input",
                                       placeholder="e.g. This product is amazing and I love it!")

            if st.button("Analyse Sentiment", type="primary", key="rnn_predict"):
                if input_text.strip():
                    model = st.session_state.rnn_model
                    vocab = st.session_state.rnn_vocab
                    score, label = model.predict_text(input_text, vocab)

                    st.subheader("Result")
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        if label == "Positive":
                            st.success(f"**{label}**")
                        else:
                            st.error(f"**{label}**")
                    with rcol2:
                        st.metric("Confidence Score", f"{score:.4f}")

                    st.progress(score)
                    st.caption(f"Score: {score:.4f}  (>= 0.5 → Positive, < 0.5 → Negative)")
                else:
                    st.warning("Please enter some text.")

            st.divider()
            st.subheader("Batch Prediction")
            st.info("Upload a CSV with a text column to predict sentiment for many rows at once.")
            batch_file = st.file_uploader("Upload CSV for batch prediction", type="csv", key="rnn_batch")
            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                batch_col = st.selectbox("Select text column", batch_df.columns, key="rnn_batch_col")
                if st.button("Run Batch Prediction", key="rnn_batch_run"):
                    model = st.session_state.rnn_model
                    vocab = st.session_state.rnn_vocab
                    scores = []
                    preds = []
                    for t in batch_df[batch_col].astype(str):
                        s, l = model.predict_text(t, vocab)
                        scores.append(round(s, 4))
                        preds.append(l)
                    batch_df["Score"] = scores
                    batch_df["Prediction"] = preds
                    st.dataframe(batch_df, use_container_width=True)

                    pos = preds.count("Positive")
                    neg = preds.count("Negative")
                    st.write(f"**Summary:** {pos} Positive, {neg} Negative out of {len(preds)} texts")
        else:
            st.warning("Please train the RNN first.")

    # ── Tab 3: Visualizations ────────────────────────────────
    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("model_type") == "rnn" and st.session_state.get("trained"):
            viz_type = st.selectbox("Select Visualization", [
                "Sentiment Distribution", "Confusion Matrix", "Loss & Accuracy Curves", "Word Cloud Preview"
            ], key="rnn_viz")

            if viz_type == "Sentiment Distribution":
                labels_data = st.session_state.get("rnn_labels", [])
                fig = plot_sentiment_distribution(labels_data, theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Confusion Matrix":
                model = st.session_state.rnn_model
                vocab = st.session_state.rnn_vocab
                texts_data = st.session_state.get("rnn_texts", [])
                labels_data = st.session_state.get("rnn_labels", [])
                y_pred = []
                for t in texts_data:
                    _, lbl = model.predict_text(t, vocab)
                    y_pred.append(1 if lbl == "Positive" else 0)
                fig, metrics = plot_confusion_matrix(labels_data, y_pred, "RNN Confusion Matrix", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                mcol2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                mcol3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                mcol4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

            elif viz_type == "Loss & Accuracy Curves":
                loss_hist = st.session_state.get("loss_history", [])
                acc_hist = st.session_state.get("rnn_acc_history", [])
                fig = plot_loss_accuracy(loss_hist, acc_hist, theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Word Cloud Preview":
                texts_data = st.session_state.get("rnn_texts", [])
                labels_data = st.session_state.get("rnn_labels", [])
                from rnn import tokenize
                fig = plot_word_frequency(texts_data, labels_data, tokenize, theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the RNN first to see visualizations.")

    # ── Tab 4: Architecture ──────────────────────────────────
    with tab4:
        st.subheader("Network Diagram")
        st.caption("Drag nodes to move, scroll to zoom")
        components.html(rnn_diagram(), height=520)

        st.divider()
        st.subheader("Step 1: Word Embedding")
        st.markdown("Each word is mapped to a dense vector via an embedding matrix:")
        st.latex(r"\mathbf{x}_t = \mathbf{W}_{embed}[\text{word}_t]")
        st.code('x_t = W_embed[word_index]  # shape: (1, embed_size)', language="python")

        st.divider()
        st.subheader("Step 2: Recurrent Hidden State")
        st.markdown("At each time step, the hidden state is updated:")
        st.latex(r"\mathbf{h}_t = \tanh(\mathbf{x}_t \cdot \mathbf{W}_{xh} + \mathbf{h}_{t-1} \cdot \mathbf{W}_{hh} + \mathbf{b}_h)")
        st.code("""z_t = x_t @ W_xh + h_{t-1} @ W_hh + b_h
h_t = tanh(z_t)""", language="python")

        st.divider()
        st.subheader("Step 3: Output (Many-to-One)")
        st.markdown("After processing all words, the final hidden state produces the output:")
        st.latex(r"y = \sigma(\mathbf{h}_T \cdot \mathbf{W}_{hy} + \mathbf{b}_y)")
        st.code("""logit = h_T @ W_hy + b_y
y = sigmoid(logit)  # 0..1 probability""", language="python")

        st.divider()
        st.subheader("Step 4: Loss (Binary Cross-Entropy)")
        st.latex(r"\mathcal{L} = -\left[ y_{true} \cdot \log(y) + (1 - y_{true}) \cdot \log(1 - y) \right]")
        st.code('loss = -(y_true * log(y) + (1 - y_true) * log(1 - y))', language="python")

        st.divider()
        st.subheader("Step 5: Backpropagation Through Time (BPTT)")
        st.markdown("Gradients flow backward through each time step:")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \sum_{t=1}^{T} \mathbf{x}_t^T \cdot \delta_t")
        st.latex(r"\delta_t = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \odot (1 - \tanh^2(\mathbf{z}_t))")
        st.code("""for t in reversed(range(T)):
    dtanh = dh * (1 - tanh(z_t)^2)
    dW_xh += x_t.T @ dtanh
    dW_hh += h_{t-1}.T @ dtanh
    dh = dtanh @ W_hh.T  # propagate to previous step""", language="python")

# ============================================
# LSTM (Sentiment Analysis)
# ============================================
elif model_type == "LSTM (Sentiment Analysis)":
    st.header("LSTM — Long Short-Term Memory")
    st.caption("An advanced recurrent network with gates that control memory flow")

    st.sidebar.header("LSTM Parameters")
    l_hidden = st.sidebar.number_input("Hidden Size", min_value=8, max_value=256, value=L_DEFAULT_HIDDEN, step=8, key="l_hidden")
    l_lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=L_DEFAULT_LR, step=0.001, format="%.4f", key="l_lr")
    l_epochs = st.sidebar.number_input("Epochs", min_value=5, max_value=500, value=L_DEFAULT_EPOCHS, step=5, key="l_epochs")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture"])

    # ── Tab 1: Upload & Train ────────────────────────────────
    with tab1:
        st.subheader("Step 1: Upload CSV File")
        st.info("CSV format: A **text** column and a **label** column (0 or 1)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="lstm_upload")
        sample_choice = st.radio("Or use sample data:", ["None", "Sentiment Analysis", "Music Genre Classification"], key="lstm_sample_choice", horizontal=True)

        if sample_choice == "Sentiment Analysis":
            df = pd.read_csv(os.path.join(DATA_DIR, "sample_sentiment.csv"))
            st.write("Sample Sentiment Data:")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Sentiment CSV", df.to_csv(index=False), "sample_sentiment.csv", "text/csv", key="dl_sentiment")
            st.session_state.lstm_df = df
            st.session_state.lstm_text_col_name = "text"
            st.session_state.lstm_label_col_name = "sentiment"
        elif sample_choice == "Music Genre Classification":
            df = pd.read_csv(os.path.join(DATA_DIR, "sample_music_genre.csv"))
            st.write("Sample Music Genre Data (0 = Calm/Soft, 1 = Heavy/Rock):")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Music Genre CSV", df.to_csv(index=False), "sample_music_genre.csv", "text/csv", key="dl_music")
            st.session_state.lstm_df = df
            st.session_state.lstm_text_col_name = "text"
            st.session_state.lstm_label_col_name = "genre"
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df, use_container_width=True)
            st.session_state.lstm_df = df
            text_col = st.selectbox("Select text column", df.columns, index=0, key="lstm_text_col")
            label_col = st.selectbox("Select label column (0/1)", df.columns, index=min(1, len(df.columns) - 1), key="lstm_label_col")
            st.session_state.lstm_text_col_name = text_col
            st.session_state.lstm_label_col_name = label_col

        if "lstm_df" in st.session_state:
            df = st.session_state.lstm_df
            texts = df[st.session_state.lstm_text_col_name].astype(str).tolist()
            labels = df[st.session_state.lstm_label_col_name].astype(int).tolist()
        else:
            texts = None
            labels = None

        st.subheader("Step 2: Train LSTM")

        if texts is not None and labels is not None:
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            st.write(f"**Dataset:** {len(labels)} samples — {pos_count} Positive, {neg_count} Negative")

            if st.button("Train LSTM", type="primary", key="lstm_train"):
                with st.spinner("Training LSTM (this may take a moment)..."):
                    model, vocab, loss_hist, acc_hist = train_lstm(
                        texts, labels,
                        hidden_size=l_hidden, lr=l_lr, epochs=l_epochs
                    )
                    st.session_state.trained = True
                    st.session_state.lstm_model = model
                    st.session_state.lstm_vocab = vocab
                    st.session_state.loss_history = loss_hist
                    st.session_state.lstm_acc_history = acc_hist
                    st.session_state.model_type = "lstm"
                    st.session_state.lstm_texts = texts
                    st.session_state.lstm_labels = labels

                st.success("Training Complete!")

                st.write(f"**Vocab size:** {len(vocab)} words  |  **Hidden size:** {l_hidden}")

                col_l, col_r = st.columns(2)
                with col_l:
                    st.subheader("Training Loss")
                    loss_df = pd.DataFrame({"Epoch": range(1, len(loss_hist) + 1), "Loss": loss_hist})
                    st.line_chart(loss_df.set_index("Epoch"))
                with col_r:
                    st.subheader("Training Accuracy")
                    acc_df = pd.DataFrame({"Epoch": range(1, len(acc_hist) + 1), "Accuracy": acc_hist})
                    st.line_chart(acc_df.set_index("Epoch"))

                final_acc = acc_hist[-1] * 100
                st.metric("Final Training Accuracy", f"{final_acc:.1f}%")
        else:
            st.warning("Please upload a CSV file or use sample data.")

    # ── Tab 2: Predict ───────────────────────────────────────
    with tab2:
        st.subheader("Predict Sentiment")

        if st.session_state.get("model_type") == "lstm" and st.session_state.get("trained"):
            st.success("LSTM model is trained and ready!")

            input_text = st.text_area("Enter text to analyse:", height=100, key="lstm_input",
                                       placeholder="e.g. This product is amazing and I love it!")

            if st.button("Analyse Sentiment", type="primary", key="lstm_predict"):
                if input_text.strip():
                    model = st.session_state.lstm_model
                    vocab = st.session_state.lstm_vocab
                    score, label = model.predict_text(input_text, vocab)

                    st.subheader("Result")
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        if label == "Positive":
                            st.success(f"**{label}**")
                        else:
                            st.error(f"**{label}**")
                    with rcol2:
                        st.metric("Confidence Score", f"{score:.4f}")

                    st.progress(score)
                    st.caption(f"Score: {score:.4f}  (>= 0.5 → Positive, < 0.5 → Negative)")
                else:
                    st.warning("Please enter some text.")

            st.divider()
            st.subheader("Batch Prediction")
            st.info("Upload a CSV with a text column to predict sentiment for many rows at once.")
            batch_file = st.file_uploader("Upload CSV for batch prediction", type="csv", key="lstm_batch")
            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                batch_col = st.selectbox("Select text column", batch_df.columns, key="lstm_batch_col")
                if st.button("Run Batch Prediction", key="lstm_batch_run"):
                    model = st.session_state.lstm_model
                    vocab = st.session_state.lstm_vocab
                    scores = []
                    preds = []
                    for t in batch_df[batch_col].astype(str):
                        s, l = model.predict_text(t, vocab)
                        scores.append(round(s, 4))
                        preds.append(l)
                    batch_df["Score"] = scores
                    batch_df["Prediction"] = preds
                    st.dataframe(batch_df, use_container_width=True)

                    pos = preds.count("Positive")
                    neg = preds.count("Negative")
                    st.write(f"**Summary:** {pos} Positive, {neg} Negative out of {len(preds)} texts")
        else:
            st.warning("Please train the LSTM first.")

    # ── Tab 3: Visualizations ────────────────────────────────
    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("model_type") == "lstm" and st.session_state.get("trained"):
            viz_type = st.selectbox("Select Visualization", [
                "Sentiment Distribution", "Confusion Matrix", "Loss & Accuracy Curves", "Word Cloud Preview"
            ], key="lstm_viz")

            if viz_type == "Sentiment Distribution":
                labels_data = st.session_state.get("lstm_labels", [])
                fig = plot_sentiment_distribution(labels_data, theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Confusion Matrix":
                model = st.session_state.lstm_model
                vocab = st.session_state.lstm_vocab
                texts_data = st.session_state.get("lstm_texts", [])
                labels_data = st.session_state.get("lstm_labels", [])
                y_pred = []
                for t in texts_data:
                    _, lbl = model.predict_text(t, vocab)
                    y_pred.append(1 if lbl == "Positive" else 0)
                fig, metrics = plot_confusion_matrix(labels_data, y_pred, "LSTM Confusion Matrix", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                mcol2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                mcol3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                mcol4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

            elif viz_type == "Loss & Accuracy Curves":
                loss_hist = st.session_state.get("loss_history", [])
                acc_hist = st.session_state.get("lstm_acc_history", [])
                fig = plot_loss_accuracy(loss_hist, acc_hist, theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Word Cloud Preview":
                texts_data = st.session_state.get("lstm_texts", [])
                labels_data = st.session_state.get("lstm_labels", [])
                from lstm import tokenize
                fig = plot_word_frequency(texts_data, labels_data, tokenize, theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the LSTM first to see visualizations.")

    # ── Tab 4: Architecture ──────────────────────────────────
    with tab4:
        st.subheader("Network Diagram")
        st.caption("Drag nodes to move, scroll to zoom")
        components.html(lstm_diagram(), height=520)

        st.divider()
        st.subheader("What is LSTM?")
        st.markdown("""
An **LSTM (Long Short-Term Memory)** is an advanced recurrent neural network that solves the **vanishing gradient problem** of vanilla RNNs using a **cell state** and **three gates**:
- **Forget Gate** — decides what to discard from memory
- **Input Gate** — decides what new information to store
- **Output Gate** — decides what to output from memory

This allows LSTMs to learn **long-range dependencies** in sequences (e.g. sentiment across a long sentence).
""")

        st.divider()
        st.subheader("Step 1: Forget Gate")
        st.markdown("Decides how much of the previous cell state to **keep**:")
        st.latex(r"f_t = \sigma(\mathbf{x}_t \cdot \mathbf{W}_{xf} + \mathbf{h}_{t-1} \cdot \mathbf{W}_{hf} + \mathbf{b}_f)")
        st.code("f_t = sigmoid(x_t @ W_xf + h_prev @ W_hf + b_f)", language="python")

        st.divider()
        st.subheader("Step 2: Input Gate + Candidate")
        st.markdown("Decides what **new information** to add to the cell state:")
        st.latex(r"i_t = \sigma(\mathbf{x}_t \cdot \mathbf{W}_{xi} + \mathbf{h}_{t-1} \cdot \mathbf{W}_{hi} + \mathbf{b}_i)")
        st.latex(r"\tilde{c}_t = \tanh(\mathbf{x}_t \cdot \mathbf{W}_{xg} + \mathbf{h}_{t-1} \cdot \mathbf{W}_{hg} + \mathbf{b}_g)")
        st.code("""i_t = sigmoid(x_t @ W_xi + h_prev @ W_hi + b_i)
g_t = tanh(x_t @ W_xg + h_prev @ W_hg + b_g)""", language="python")

        st.divider()
        st.subheader("Step 3: Cell State Update")
        st.markdown("Combines forget and input gates to update memory:")
        st.latex(r"c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t")
        st.code("c_t = f_t * c_prev + i_t * g_t", language="python")

        st.divider()
        st.subheader("Step 4: Output Gate")
        st.markdown("Decides what part of the cell state to **output**:")
        st.latex(r"o_t = \sigma(\mathbf{x}_t \cdot \mathbf{W}_{xo} + \mathbf{h}_{t-1} \cdot \mathbf{W}_{ho} + \mathbf{b}_o)")
        st.latex(r"h_t = o_t \odot \tanh(c_t)")
        st.code("""o_t = sigmoid(x_t @ W_xo + h_prev @ W_ho + b_o)
h_t = o_t * tanh(c_t)""", language="python")

        st.divider()
        st.subheader("Step 5: Final Output")
        st.markdown("After processing all words, the final hidden state produces the prediction:")
        st.latex(r"y = \sigma(\mathbf{h}_T \cdot \mathbf{W}_{hy} + \mathbf{b}_y)")
        st.code("""y = sigmoid(h_T @ W_hy + b_y)  # 0..1 probability""", language="python")

        st.divider()
        st.subheader("LSTM vs RNN")
        st.markdown("""
| | RNN | LSTM |
|---|---|---|
| **Memory** | Short-term only | Short-term + long-term (cell state) |
| **Gates** | None | Forget, Input, Output |
| **Vanishing Gradient** | Severe problem | Solved by cell state |
| **Long Sequences** | Struggles | Handles well |
| **Parameters** | Fewer | ~4x more (4 gate weight matrices) |
""")

# ============================================
# MSE LOSS (Linear Regression)
# ============================================
elif model_type == "MSE Loss (Linear Regression)":
    st.header("MSE Loss — Linear Regression")
    st.caption("Gradient descent with Mean Squared Error loss")

    st.sidebar.header("MSE Parameters")
    mse_mode = st.sidebar.radio("Input Mode", ["Single Variable (1 feature)", "Two Variables (2 features)"], key="mse_mode")
    m_lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=M_DEFAULT_LR, step=0.001, format="%.5f", key="m_lr")
    m_epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=5000, value=M_DEFAULT_EPOCHS, step=10, key="m_epochs")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture"])

    # ── Tab 1: Upload & Train ────────────────────────────────
    with tab1:
        st.subheader("Step 1: Upload CSV File")

        if mse_mode == "Single Variable (1 feature)":
            st.info("CSV format: First column = feature (X), Last column = target (y)")
        else:
            st.info("CSV format: First 2 columns = features (X1, X2), Last column = target (y)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="mse_upload")
        use_sample = st.checkbox("Use sample data instead", key="mse_sample")

        if use_sample:
            if mse_mode == "Single Variable (1 feature)":
                X_single = [0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0]
                y_vals =   [150, 180, 200, 230, 245, 270, 290, 320, 350, 400, 440, 470, 510, 550, 590, 620, 700, 780]
                df = pd.DataFrame({"House_Size_1000sqft": X_single, "Price_1000USD": y_vals})
                st.write("Sample Data — House Price vs Size:")
                st.dataframe(df, use_container_width=True)
                st.session_state.mse_col_names = ["House_Size_1000sqft"]
            else:
                X1_vals = [0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0, 5.5]
                X2_vals = [1,   1,   2,   2,   2,   2,   3,   3,   3,   3,   4,   4,   4,   4,   5,   5,   5,   6]
                y_vals  = [150, 180, 210, 240, 250, 280, 340, 370, 410, 450, 490, 520, 560, 600, 640, 720, 800, 880]
                df = pd.DataFrame({"House_Size_1000sqft": X1_vals, "Bedrooms": X2_vals, "Price_1000USD": y_vals})
                st.write("Sample Data — House Price vs Size & Bedrooms:")
                st.dataframe(df, use_container_width=True)
                st.session_state.mse_col_names = ["House_Size_1000sqft", "Bedrooms"]
            st.session_state.mse_df = df
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df)
            st.session_state.mse_df = df
            if mse_mode == "Single Variable (1 feature)":
                st.session_state.mse_col_names = [df.columns[0]]
            else:
                st.session_state.mse_col_names = [df.columns[0], df.columns[1]]

        if "mse_df" in st.session_state:
            df = st.session_state.mse_df
            y_vals = df.iloc[:, -1].values.tolist()
            if mse_mode == "Single Variable (1 feature)":
                X_single = df.iloc[:, 0].values.tolist()
            else:
                X1_vals = df.iloc[:, 0].values.tolist()
                X2_vals = df.iloc[:, 1].values.tolist()
        else:
            df = None
            y_vals = None

        st.subheader("Step 2: Train Model")

        if df is not None and y_vals is not None:
            if st.button("Train", type="primary", key="mse_train"):
                with st.spinner("Training..."):
                    if mse_mode == "Single Variable (1 feature)":
                        w, b, init_w, init_b, loss_hist = train_mse_single(
                            X_single, y_vals, learning_rate=m_lr, epochs=m_epochs
                        )
                        st.session_state.trained = True
                        st.session_state.mse_weights = (w, b)
                        st.session_state.loss_history = loss_hist
                        st.session_state.model_type = "mse_single"
                        st.session_state.mse_X_single = X_single
                        st.session_state.mse_y = y_vals

                        st.success("Training Complete!")

                        st.subheader("Initial Random Weights")
                        st.write(f"w: {init_w:.6f},  b: {init_b:.6f}")

                        st.subheader("Trained Weights")
                        st.write(f"w: {w:.6f},  b: {b:.6f}")

                    else:
                        w1, w2, b, iw1, iw2, ib, loss_hist = train_mse_dual(
                            X1_vals, X2_vals, y_vals, learning_rate=m_lr, epochs=m_epochs
                        )
                        st.session_state.trained = True
                        st.session_state.mse_weights = (w1, w2, b)
                        st.session_state.loss_history = loss_hist
                        st.session_state.model_type = "mse_dual"
                        st.session_state.mse_X1 = X1_vals
                        st.session_state.mse_X2 = X2_vals
                        st.session_state.mse_y = y_vals

                        st.success("Training Complete!")

                        st.subheader("Initial Random Weights")
                        st.write(f"w1: {iw1:.6f},  w2: {iw2:.6f},  b: {ib:.6f}")

                        st.subheader("Trained Weights")
                        wcol1, wcol2, wcol3 = st.columns(3)
                        with wcol1:
                            st.write(f"w1: {w1:.6f}")
                        with wcol2:
                            st.write(f"w2: {w2:.6f}")
                        with wcol3:
                            st.write(f"b: {b:.6f}")

                    st.subheader("Training Loss (MSE) Over Epochs")
                    loss_df = pd.DataFrame({"Epoch": range(1, len(loss_hist) + 1), "MSE Loss": loss_hist})
                    st.line_chart(loss_df.set_index("Epoch"))

                    st.metric("Final MSE Loss", f"{loss_hist[-1]:.6f}")
        else:
            st.warning("Please upload a CSV file or use sample data.")

    # ── Tab 2: Predict ───────────────────────────────────────
    with tab2:
        st.subheader("Predict Output")

        if st.session_state.get("trained") and st.session_state.get("model_type", "").startswith("mse"):
            st.success("Model is trained and ready!")
            col_names = st.session_state.get("mse_col_names", ["Feature 1", "Feature 2"])

            if st.session_state.model_type == "mse_single":
                pred_x = st.number_input(col_names[0], value=6.0, key="mse_pred_x")
                if st.button("Predict", type="primary", key="mse_predict"):
                    w, b = st.session_state.mse_weights
                    result = predict_single(pred_x, w, b)
                    st.subheader("Prediction Result")
                    st.metric("Predicted Value (ŷ)", f"{result:.4f}")
                    st.caption(f"ŷ = {w:.4f} × {pred_x} + {b:.4f} = {result:.4f}")
            else:
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    pred_x1 = st.number_input(col_names[0], value=6.0, key="mse_pred_x1")
                with pcol2:
                    pred_x2 = st.number_input(col_names[1], value=80.0, key="mse_pred_x2")
                if st.button("Predict", type="primary", key="mse_predict"):
                    w1, w2, b = st.session_state.mse_weights
                    result = predict_dual(pred_x1, pred_x2, w1, w2, b)
                    st.subheader("Prediction Result")
                    st.metric("Predicted Value (ŷ)", f"{result:.4f}")
                    st.caption(f"ŷ = {w1:.4f}×{pred_x1} + {w2:.4f}×{pred_x2} + {b:.4f} = {result:.4f}")
        else:
            st.warning("Please train the model first.")

    # ── Tab 3: Visualizations ────────────────────────────────
    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("trained") and st.session_state.get("model_type", "").startswith("mse"):
            viz_type = st.selectbox("Select Visualization", [
                "Regression Fit", "Residual Plot", "Loss Curve"
            ], key="mse_viz")

            if viz_type == "Regression Fit":
                if st.session_state.model_type == "mse_single":
                    w, b = st.session_state.mse_weights
                    X_data = st.session_state.get("mse_X_single", [])
                    y_data = st.session_state.get("mse_y", [])
                    fig = plot_regression_line(X_data, y_data, w, b, "Linear Regression Fit (Single Variable)", theme=_t)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Blue dots = actual data. Red line = fitted model. Gray dashed = residuals (errors).")
                else:
                    w1, w2, b = st.session_state.mse_weights
                    X1_data = st.session_state.get("mse_X1", [])
                    X2_data = st.session_state.get("mse_X2", [])
                    y_data = st.session_state.get("mse_y", [])
                    fig = plot_regression_3d(X1_data, X2_data, y_data, w1, w2, b, "3D Regression Surface", theme=_t)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Blue dots = actual data. Surface = fitted plane. Drag to rotate in 3D.")

            elif viz_type == "Residual Plot":
                y_data = st.session_state.get("mse_y", [])
                if st.session_state.model_type == "mse_single":
                    w, b = st.session_state.mse_weights
                    X_data = st.session_state.get("mse_X_single", [])
                    y_pred = [predict_single(x, w, b) for x in X_data]
                else:
                    w1, w2, b = st.session_state.mse_weights
                    X1_data = st.session_state.get("mse_X1", [])
                    X2_data = st.session_state.get("mse_X2", [])
                    y_pred = [predict_dual(X1_data[i], X2_data[i], w1, w2, b) for i in range(len(X1_data))]
                fig = plot_residual(y_data, y_pred, theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points near zero = good predictions. Patterns suggest model is missing something.")

            elif viz_type == "Loss Curve" and st.session_state.get("loss_history"):
                fig = plot_loss_curve(st.session_state.loss_history, "MSE Training Loss", ylabel="MSE", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the model first to see visualizations.")

    # ── Tab 4: Architecture ──────────────────────────────────
    with tab4:
        st.subheader("Network Diagram")
        st.caption("Drag nodes to move, scroll to zoom")
        components.html(mse_diagram(), height=520)

        st.divider()
        st.subheader("Step 1: Prediction (Forward Pass)")
        st.markdown("**Single variable:**")
        st.latex(r"\hat{y} = w \cdot x + b")
        st.markdown("**Two variables:**")
        st.latex(r"\hat{y} = w_1 \cdot x_1 + w_2 \cdot x_2 + b")
        st.code("""y_pred = w1 * x1 + w2 * x2 + b""", language="python")

        st.divider()
        st.subheader("Step 2: MSE Loss")
        st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2")
        st.code("""loss = sum((y_pred - y_true)^2) / n""", language="python")

        st.divider()
        st.subheader("Step 3: Compute Gradients")
        st.latex(r"\frac{\partial \text{MSE}}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \cdot x_i")
        st.latex(r"\frac{\partial \text{MSE}}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)")
        st.code("""for i in range(n):
    error = y_pred - y[i]
    dw += error * x[i]
    db += error
dw = (2 / n) * dw
db = (2 / n) * db""", language="python")

        st.divider()
        st.subheader("Step 4: Update Weights (Gradient Descent)")
        st.latex(r"w = w - \eta \cdot \frac{\partial \text{MSE}}{\partial w}")
        st.latex(r"b = b - \eta \cdot \frac{\partial \text{MSE}}{\partial b}")
        st.code("""w = w - learning_rate * dw
b = b - learning_rate * db""", language="python")

# ============================================
# CNN (Face Classification)
# ============================================
elif model_type == "CNN (Face Classification)":
    import numpy as np
    import matplotlib.pyplot as plt

    st.title("CNN — Face Classification")

    st.sidebar.header("CNN Parameters")
    c_lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=C_DEFAULT_LR, step=0.001, format="%.4f", key="c_lr")
    c_epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=C_DEFAULT_EPOCHS, step=1, key="c_epochs")
    c_augment = st.sidebar.checkbox("Data Augmentation", value=True, key="c_augment")

    _t = st.session_state.theme

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture", "Live Attendance"])

    # ── Tab 1: Upload & Train ─────────────────────────────────
    with tab1:
        st.subheader("Upload Face Images")
        st.markdown("Upload images for **each person** you want to classify. Provide at least 2-3 images per person for best results.")

        num_classes = st.number_input("Number of people/classes", min_value=2, max_value=10, value=2, step=1, key="cnn_num_classes")

        all_images = []
        all_labels = []
        class_names = []

        for c in range(num_classes):
            st.markdown(f"**Person {c + 1}**")
            name = st.text_input(f"Name for person {c + 1}", value=f"Person {c + 1}", key=f"cnn_name_{c}")
            class_names.append(name)
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                uploaded = st.file_uploader(f"Upload images for {name}", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key=f"cnn_upload_{c}")
            with c_col2:
                camera_img = st.camera_input(f"Or take a picture for {name}", key=f"cnn_camera_{c}")
                
            all_files_for_class = []
            if uploaded:
                all_files_for_class.extend(uploaded)
            if camera_img:
                all_files_for_class.append(camera_img)

            if all_files_for_class:
                for f in all_files_for_class:
                    img_bytes = f.read()
                    face_crop, bbox, full_img = detect_face_pil(img_bytes)
                    processed = preprocess_image(face_crop, IMG_SIZE)
                    if c_augment:
                        augmented = augment_image(processed)
                        for aug in augmented:
                            all_images.append(aug)
                            all_labels.append(c)
                    else:
                        all_images.append(processed)
                        all_labels.append(c)
                st.success(f"Loaded {len(all_files_for_class)} image(s) for {name}" + (f" (augmented to {len([l for l in all_labels if l == c])} samples)" if c_augment else ""))

        st.divider()
        if st.button("Train CNN", key="cnn_train_btn"):
            if len(all_images) < 2:
                st.error("Upload at least 1 image per class to train.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart = st.empty()

                loss_history_live = []
                acc_history_live = []

                def progress_cb(epoch, total, loss, acc):
                    progress_bar.progress(epoch / total)
                    status_text.markdown(f"**Epoch {epoch}/{total}** — Loss: `{loss:.4f}` — Accuracy: `{acc:.2%}`")
                    loss_history_live.append(loss)
                    acc_history_live.append(acc)

                model, loss_hist, acc_hist = train_cnn(
                    all_images, all_labels, num_classes,
                    lr=c_lr, epochs=c_epochs, img_size=IMG_SIZE,
                    progress_callback=progress_cb
                )

                progress_bar.progress(1.0)
                st.success(f"Training complete! Final accuracy: {acc_hist[-1]:.2%}")

                st.session_state.cnn_model = model
                st.session_state.cnn_class_names = class_names
                st.session_state.cnn_loss_history = loss_hist
                st.session_state.cnn_acc_history = acc_hist
                st.session_state.cnn_train_images = all_images
                st.session_state.cnn_train_labels = all_labels
                st.session_state.model_type = "cnn"
                st.session_state.trained = True

    # ── Tab 2: Predict ──────────────────────────────────────
    with tab2:
        if st.session_state.get("model_type") == "cnn" and st.session_state.get("trained"):
            st.subheader("Upload or capture a face image to classify")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                pred_file = st.file_uploader("Upload test image", type=["jpg", "jpeg", "png"], key="cnn_pred_upload")
            with p_col2:
                pred_cam = st.camera_input("Or take a picture", key="cnn_pred_camera")
                
            img_to_predict = pred_file or pred_cam

            if img_to_predict:
                img_bytes = img_to_predict.read()
                face_crop, bbox, full_img = detect_face_pil(img_bytes)
                processed = preprocess_image(face_crop, IMG_SIZE)

                model = st.session_state.cnn_model
                class_names = st.session_state.cnn_class_names

                pred_class, confidence, probs = model.predict(processed)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(full_img, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(face_crop, caption="Detected Face", use_container_width=True)

                st.markdown(f"### Prediction: **{class_names[pred_class]}**")
                st.markdown(f"Confidence: **{confidence:.2%}**")

                st.divider()
                st.subheader("Class Probabilities")
                prob_data = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
                st.bar_chart(prob_data)

            st.divider()
            st.subheader("Batch Prediction")
            batch_files = st.file_uploader("Upload multiple test images", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key="cnn_batch_upload")
            if batch_files:
                model = st.session_state.cnn_model
                class_names = st.session_state.cnn_class_names
                results = []
                for f in batch_files:
                    img_bytes = f.read()
                    face_crop, bbox, full_img = detect_face_pil(img_bytes)
                    processed = preprocess_image(face_crop, IMG_SIZE)
                    pred_class, confidence, probs = model.predict(processed)
                    results.append({"File": f.name, "Prediction": class_names[pred_class], "Confidence": f"{confidence:.2%}"})
                st.dataframe(results, use_container_width=True)
        else:
            st.warning("Train the CNN model first to make predictions.")

    # ── Tab 3: Visualizations ───────────────────────────────
    with tab3:
        if st.session_state.get("model_type") == "cnn" and st.session_state.get("trained"):
            viz_type = st.selectbox("Select Visualization", [
                "Loss & Accuracy Curves",
                "Feature Maps",
                "Conv Filters",
                "Confusion Matrix"
            ], key="cnn_viz_type")

            model = st.session_state.cnn_model

            if viz_type == "Loss & Accuracy Curves":
                fig = plot_loss_accuracy(
                    st.session_state.cnn_loss_history,
                    st.session_state.cnn_acc_history,
                    theme=_t
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Feature Maps":
                st.markdown("Upload an image to visualize intermediate feature maps.")
                fm_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="cnn_fm_upload")
                if fm_file:
                    img_bytes = fm_file.read()
                    face_crop, _, _ = detect_face_pil(img_bytes)
                    processed = preprocess_image(face_crop, IMG_SIZE)
                    maps = model.get_feature_maps(processed)

                    for layer_name, fmap in maps.items():
                        st.markdown(f"**{layer_name}** — shape: {fmap.shape}")
                        n_filters = fmap.shape[2]
                        cols = st.columns(min(n_filters, 8))
                        for i in range(min(n_filters, 8)):
                            with cols[i]:
                                fig, ax = plt.subplots(figsize=(2, 2))
                                ax.imshow(fmap[:, :, i], cmap='viridis')
                                ax.axis('off')
                                ax.set_title(f"F{i}", fontsize=8)
                                st.pyplot(fig)
                                plt.close(fig)

            elif viz_type == "Conv Filters":
                filters = model.get_filters()
                for layer_name, f in filters.items():
                    st.markdown(f"**{layer_name}** — {f.shape[0]} filters, shape {f.shape[1]}x{f.shape[2]}")
                    n = f.shape[0]
                    cols = st.columns(min(n, 8))
                    for i in range(min(n, 8)):
                        with cols[i]:
                            fig, ax = plt.subplots(figsize=(2, 2))
                            ax.imshow(f[i, :, :, 0], cmap='gray')
                            ax.axis('off')
                            ax.set_title(f"F{i}", fontsize=8)
                            st.pyplot(fig)
                            plt.close(fig)

            elif viz_type == "Confusion Matrix":
                train_images = st.session_state.cnn_train_images
                train_labels = st.session_state.cnn_train_labels
                class_names = st.session_state.cnn_class_names
                y_true = train_labels
                y_pred = []
                for img in train_images:
                    pred_class, _, _ = model.predict(img)
                    y_pred.append(pred_class)
                fig = plot_confusion_matrix(y_true, y_pred, labels=class_names, theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the model first to see visualizations.")

    # ── Tab 4: Architecture ─────────────────────────────────
    with tab4:
        st.subheader("CNN Architecture")
        st.caption("Drag nodes to move, scroll to zoom")
        components.html(cnn_diagram(), height=520)

        st.divider()
        st.subheader("Step 1: Convolution (Forward Pass)")
        st.latex(r"\text{output}(i,j,f) = \sum_{m}\sum_{n}\sum_{c} \text{input}(i{+}m, j{+}n, c) \cdot \text{filter}(m,n,c,f) + b_f")
        st.code("""# im2col: extract all patches as matrix rows
cols = im2col(padded_input, kernel_size, stride)
# Matrix multiply: output = cols @ filters + biases
output = cols @ W + biases""", language="python")

        st.divider()
        st.subheader("Step 2: ReLU Activation")
        st.latex(r"\text{ReLU}(x) = \max(0, x)")

        st.divider()
        st.subheader("Step 3: Max Pooling")
        st.latex(r"\text{MaxPool}(x) = \max_{(m,n) \in \text{window}} x(m,n)")
        st.code("""# 2x2 max pooling reduces spatial dimensions by half
output[i,j] = max(input[2i:2i+2, 2j:2j+2])""", language="python")

        st.divider()
        st.subheader("Step 4: Softmax + Cross-Entropy Loss")
        st.latex(r"\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}")
        st.latex(r"\mathcal{L} = -\log(\text{softmax}(z)_{\text{true class}})")

        st.divider()
        st.subheader("Step 5: Backpropagation")
        st.markdown("Gradients flow backwards through each layer using the chain rule, updating filters and weights via gradient descent.")
        st.code("""# Gradient of cross-entropy + softmax
d = probs.copy()
d[true_label] -= 1

# Backprop through each layer in reverse
for layer in reversed(layers):
    d = layer.backward(d, learning_rate)""", language="python")

    # ── Tab 5: Live Attendance ─────────────────────────────────
    with tab5:
        import datetime, cv2, io
        from PIL import Image as PILImage

        st.subheader("Live Attendance System")

        if st.session_state.get("model_type") != "cnn" or not st.session_state.get("trained"):
            st.warning("Train the CNN model first (Tab 1) before using live attendance.")
        else:
            model = st.session_state.cnn_model
            class_names = st.session_state.cnn_class_names

            # Initialize attendance log
            if "attendance_log" not in st.session_state:
                st.session_state.attendance_log = []

            conf_threshold = st.slider(
                "Confidence Threshold", 0.50, 0.99, 0.70, 0.05,
                help="Only mark attendance if confidence exceeds this threshold",
                key="att_conf_threshold"
            )

            st.divider()
            st.markdown("**Capture a photo from your webcam to mark attendance:**")

            camera_img = st.camera_input("Take a photo", key="att_camera")

            if camera_img is not None:
                img_bytes = camera_img.getvalue()

                # Detect face and preprocess
                face_crop, bbox, full_img = detect_face_pil(img_bytes)
                processed = preprocess_image(face_crop, IMG_SIZE)

                # Predict
                pred_class, confidence, probs = model.predict(processed)
                pred_name = class_names[pred_class]

                # Show detection results
                col_a, col_b = st.columns(2)
                with col_a:
                    # Draw bounding box on original image
                    annotated = full_img.copy()
                    x, y, w, h = bbox
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_text = f"{pred_name} ({confidence:.0%})"
                    cv2.putText(annotated, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    st.image(annotated, caption="Detected Face", use_container_width=True)
                with col_b:
                    st.image(face_crop, caption="Cropped Face", use_container_width=True)
                    st.markdown(f"**Identified:** {pred_name}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")

                    # Class probabilities
                    prob_data = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
                    st.bar_chart(prob_data)

                # Mark attendance
                if confidence >= conf_threshold:
                    now = datetime.datetime.now()
                    # Check if this person was already marked in the last 5 minutes
                    already_marked = False
                    for entry in st.session_state.attendance_log:
                        if entry["Name"] == pred_name:
                            prev_time = datetime.datetime.strptime(entry["Time"], "%Y-%m-%d %H:%M:%S")
                            if (now - prev_time).total_seconds() < 300:
                                already_marked = True
                                break

                    if already_marked:
                        st.info(f"{pred_name} was already marked present within the last 5 minutes.")
                    else:
                        st.session_state.attendance_log.append({
                            "Name": pred_name,
                            "Time": now.strftime("%Y-%m-%d %H:%M:%S"),
                            "Confidence": f"{confidence:.2%}"
                        })
                        st.success(f"Attendance marked for **{pred_name}** at {now.strftime('%H:%M:%S')}")
                else:
                    st.warning(f"Confidence ({confidence:.2%}) is below threshold ({conf_threshold:.0%}). Attendance NOT marked.")

            # ── Attendance Log ──
            st.divider()
            st.subheader("Attendance Log")

            if st.session_state.attendance_log:
                log_df = pd.DataFrame(st.session_state.attendance_log)
                st.dataframe(log_df, use_container_width=True)

                # Summary
                st.markdown(f"**Total entries:** {len(log_df)} | **Unique people:** {log_df['Name'].nunique()}")

                # Download as CSV
                csv = log_df.to_csv(index=False)
                st.download_button(
                    "Download Attendance CSV",
                    csv,
                    file_name=f"attendance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="att_download"
                )

                # Clear log
                if st.button("Clear Attendance Log", key="att_clear"):
                    st.session_state.attendance_log = []
                    st.rerun()
            else:
                st.info("No attendance recorded yet. Capture a photo above to start.")

# ============================================
# HOPFIELD NETWORK — Alphabet Recognition
# ============================================
elif model_type == "Hopfield Network (Alphabet)":
    import numpy as np
    import matplotlib.pyplot as plt
    from streamlit_drawable_canvas import st_canvas
    from PIL import Image as PILImage

    st.title("Hopfield Network — Alphabet Recognition")

    st.sidebar.header("Hopfield Parameters")
    hop_letters = st.sidebar.multiselect(
        "Letters to store",
        list(ALPHABET_PATTERNS.keys()),
        default=["A", "B", "C", "D", "E"],
        key="hop_letters"
    )
    hop_noise = st.sidebar.slider("Noise Level (for testing)", 0.0, 0.5, 0.15, 0.05, key="hop_noise")
    hop_max_iter = st.sidebar.number_input("Max Recall Iterations", 1, 100, 20, key="hop_max_iter")
    hop_rule = st.sidebar.selectbox(
        "Training Rule",
        ["pseudo-inverse", "hebbian"],
        index=0,
        help="Pseudo-inverse has higher capacity and handles correlated patterns better than Hebbian.",
        key="hop_rule"
    )

    _t = st.session_state.theme

    tab1, tab2, tab3, tab4 = st.tabs(["Draw & Recognize", "Stored Patterns", "Noise Test", "How It Works"])

    # Train the network with selected letters
    if len(hop_letters) >= 1:
        patterns = {letter: get_pattern_vector(letter) for letter in hop_letters}
        hopfield_net = HopfieldNetwork(PATTERN_SIZE)
        hopfield_net.train(patterns, rule=hop_rule)
        st.session_state.hopfield_net = hopfield_net
        st.session_state.hop_patterns = patterns

    # ── Tab 1: Draw & Recognize ────────────────────────────────
    with tab1:
        st.subheader("Draw an Alphabet Letter")
        st.markdown(f"The network has **{len(hop_letters)} letters** stored: {', '.join(hop_letters)}")
        st.markdown("Draw a letter on the canvas below. The grid is 10x10 — fill in the cells to form a letter shape.")

        CELL_SIZE = 40
        CANVAS_SIZE = GRID_SIZE * CELL_SIZE  # 400px

        col_draw, col_result = st.columns([1, 1])

        with col_draw:
            st.markdown("**Draw here** (use the brush to fill cells):")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=CELL_SIZE - 2,
                stroke_color="#FFFFFF",
                background_color="#1a1a2e",
                width=CANVAS_SIZE,
                height=CANVAS_SIZE,
                drawing_mode="freedraw",
                key="hopfield_canvas",
            )

            if st.button("Recognize", key="hop_recognize"):
                if canvas_result.image_data is not None:
                    # Convert canvas to 10x10 grid
                    img_array = canvas_result.image_data[:, :, :3]  # drop alpha
                    brightness = np.mean(img_array, axis=2)  # grayscale

                    # Downsample to 10x10 using fill ratio per cell
                    grid = np.zeros((GRID_SIZE, GRID_SIZE))
                    for r in range(GRID_SIZE):
                        for c in range(GRID_SIZE):
                            block = brightness[
                                r * CELL_SIZE:(r + 1) * CELL_SIZE,
                                c * CELL_SIZE:(c + 1) * CELL_SIZE
                            ]
                            filled = np.mean(block > 127)
                            grid[r, c] = 1 if filled >= 0.15 else -1

                    bipolar = grid.flatten()

                    st.session_state.hop_input = bipolar
                    st.session_state.hop_input_grid = (grid == 1).astype(int)

                    # Recall
                    net = st.session_state.hopfield_net
                    recalled, energy_hist, snapshots = net.recall(bipolar, max_iterations=hop_max_iter, asynchronous=True)
                    label, similarity = net.identify(recalled)

                    st.session_state.hop_recalled = recalled
                    st.session_state.hop_energy = energy_hist
                    st.session_state.hop_snapshots = snapshots
                    st.session_state.hop_label = label
                    st.session_state.hop_similarity = similarity
                else:
                    st.warning("Please draw something on the canvas first.")

        with col_result:
            if "hop_label" in st.session_state:
                label = st.session_state.hop_label
                similarity = st.session_state.hop_similarity
                recalled = st.session_state.hop_recalled
                input_grid = st.session_state.hop_input_grid
                energy_hist = st.session_state.hop_energy

                st.markdown(f"### Recognized: **{label}**")
                st.markdown(f"**Similarity:** {similarity:.2%}")
                st.markdown(f"**Converged in:** {len(energy_hist) - 1} iterations")

                # Show input vs recalled vs stored
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                bg = '#1a1a2e' if _t == 'dark' else '#ffffff'
                fig.patch.set_facecolor(bg)

                # Input
                axes[0].imshow(input_grid.reshape(GRID_SIZE, GRID_SIZE),
                               cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
                axes[0].set_title("Your Drawing", color='white' if _t == 'dark' else 'black', fontsize=11)
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                # Add grid lines
                for ax in axes:
                    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
                    ax.grid(which='minor', color='gray', linewidth=0.5)
                    ax.tick_params(which='minor', size=0)
                    ax.set_facecolor(bg)

                # Recalled
                recalled_grid = ((recalled + 1) / 2).reshape(GRID_SIZE, GRID_SIZE)
                axes[1].imshow(recalled_grid, cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
                axes[1].set_title("Recalled", color='white' if _t == 'dark' else 'black', fontsize=11)
                axes[1].set_xticks([])
                axes[1].set_yticks([])

                # Stored
                stored = st.session_state.hop_patterns[label]
                stored_grid = ((stored + 1) / 2).reshape(GRID_SIZE, GRID_SIZE)
                axes[2].imshow(stored_grid, cmap='Oranges', interpolation='nearest', vmin=0, vmax=1)
                axes[2].set_title(f"Stored '{label}'", color='white' if _t == 'dark' else 'black', fontsize=11)
                axes[2].set_xticks([])
                axes[2].set_yticks([])

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Energy curve
                st.markdown("**Energy Convergence:**")
                fig2, ax2 = plt.subplots(figsize=(6, 2.5))
                fig2.patch.set_facecolor(bg)
                ax2.set_facecolor(bg)
                ax2.plot(energy_hist, marker='o', color='#4CAF50', linewidth=2, markersize=5)
                ax2.set_xlabel("Iteration", color='white' if _t == 'dark' else 'black')
                ax2.set_ylabel("Energy", color='white' if _t == 'dark' else 'black')
                ax2.tick_params(colors='white' if _t == 'dark' else 'black')
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

    # ── Tab 2: Stored Patterns ─────────────────────────────────
    with tab2:
        st.subheader("Stored Alphabet Patterns")
        st.markdown(f"Currently storing **{len(hop_letters)}** letters. Select letters from the sidebar.")

        if len(hop_letters) > 0:
            # Display patterns in a grid
            cols_per_row = 5
            for row_start in range(0, len(hop_letters), cols_per_row):
                row_letters = hop_letters[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for i, letter in enumerate(row_letters):
                    with cols[i]:
                        pattern = get_pattern_vector(letter)
                        grid = ((pattern + 1) / 2).reshape(GRID_SIZE, GRID_SIZE)
                        fig, ax = plt.subplots(figsize=(2.5, 2.5))
                        bg = '#1a1a2e' if _t == 'dark' else '#ffffff'
                        fig.patch.set_facecolor(bg)
                        ax.set_facecolor(bg)
                        ax.imshow(grid, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
                        ax.set_title(letter, fontsize=18, fontweight='bold',
                                     color='white' if _t == 'dark' else 'black')
                        ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
                        ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
                        ax.grid(which='minor', color='gray', linewidth=0.5)
                        ax.tick_params(which='minor', size=0)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            # Weight matrix visualization
            st.divider()
            st.subheader("Weight Matrix")
            st.markdown("The Hopfield weight matrix shows connection strengths between all 100 neurons (10x10 grid).")
            fig, ax = plt.subplots(figsize=(6, 5))
            bg = '#1a1a2e' if _t == 'dark' else '#ffffff'
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            net = st.session_state.hopfield_net
            im = ax.imshow(net.weights, cmap='RdBu', interpolation='nearest', aspect='auto')
            ax.set_title("Weight Matrix (100x100)", color='white' if _t == 'dark' else 'black')
            ax.set_xlabel("Neuron j", color='white' if _t == 'dark' else 'black')
            ax.set_ylabel("Neuron i", color='white' if _t == 'dark' else 'black')
            ax.tick_params(colors='white' if _t == 'dark' else 'black')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Capacity info
            st.divider()
            st.info(f"**Hopfield capacity rule:** A network with {PATTERN_SIZE} neurons can reliably store ~{int(PATTERN_SIZE / (2 * np.log(PATTERN_SIZE)))} patterns. You have {len(hop_letters)} stored.")

    # ── Tab 3: Noise Test ──────────────────────────────────────
    with tab3:
        st.subheader("Noise Robustness Test")
        st.markdown("See how well the network recovers letters from noisy (corrupted) inputs.")

        if len(hop_letters) >= 1:
            test_letter = st.selectbox("Select letter to test", hop_letters, key="hop_test_letter")

            if st.button("Run Noise Test", key="hop_noise_test"):
                net = st.session_state.hopfield_net
                original = get_pattern_vector(test_letter)

                noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                fig, axes = plt.subplots(2, len(noise_levels), figsize=(15, 6))
                bg = '#1a1a2e' if _t == 'dark' else '#ffffff'
                fig.patch.set_facecolor(bg)

                for i, nl in enumerate(noise_levels):
                    if nl == 0:
                        noisy = original.copy()
                    else:
                        noisy = net.add_noise(original, nl)

                    recalled, energy, _ = net.recall(noisy, max_iterations=hop_max_iter, asynchronous=True)
                    label, sim = net.identify(recalled)

                    # Noisy input
                    noisy_grid = ((noisy + 1) / 2).reshape(GRID_SIZE, GRID_SIZE)
                    axes[0, i].imshow(noisy_grid, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
                    axes[0, i].set_title(f"Noise: {nl:.0%}",
                                         color='white' if _t == 'dark' else 'black', fontsize=10)
                    axes[0, i].set_xticks([])
                    axes[0, i].set_yticks([])
                    axes[0, i].set_facecolor(bg)

                    # Recalled output
                    recalled_grid = ((recalled + 1) / 2).reshape(GRID_SIZE, GRID_SIZE)
                    color = '#4CAF50' if label == test_letter else '#f44336'
                    axes[1, i].imshow(recalled_grid, cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
                    axes[1, i].set_title(f"→ {label} ({sim:.0%})",
                                         color=color, fontsize=10, fontweight='bold')
                    axes[1, i].set_xticks([])
                    axes[1, i].set_yticks([])
                    axes[1, i].set_facecolor(bg)

                axes[0, 0].set_ylabel("Input", color='white' if _t == 'dark' else 'black', fontsize=12)
                axes[1, 0].set_ylabel("Recalled", color='white' if _t == 'dark' else 'black', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Tab 4: How It Works ────────────────────────────────────
    with tab4:
        st.subheader("How Hopfield Networks Work")

        st.markdown("""
A **Hopfield Network** is a recurrent neural network that acts as **content-addressable memory**
(associative memory). Given a noisy or partial input, it recalls the closest stored pattern.
        """)

        st.divider()
        st.subheader("Step 1: Storing Patterns")
        st.markdown("Patterns are stored by computing the weight matrix. Two common rules are supported in this toolbox:")
        
        st.markdown("**1. Hebbian Learning (Classic):**")
        st.latex(r"W = \frac{1}{N} \sum_{p=1}^{P} \mathbf{x}^{(p)} {\mathbf{x}^{(p)}}^T, \quad W_{ii} = 0")
        st.markdown("Simple and biologically plausible, but has low capacity and struggles with correlated patterns (like alphabet letters).")
        
        st.markdown("**2. Pseudo-inverse (Projection) Rule:**")
        st.latex(r"W = P P^+, \quad W_{ii} = 0")
        st.markdown("Where $P$ is the matrix of patterns and $P^+$ is its pseudo-inverse. This rule can store up to $N$ patterns and handles correlated letters much better by orthogonalizing the weight space.")

        st.divider()
        st.subheader("Step 2: Recall (Pattern Completion)")
        st.markdown("Given a noisy input, the network iteratively updates until convergence:")
        st.latex(r"\mathbf{s}(t+1) = \text{sign}(W \cdot \mathbf{s}(t))")
        st.markdown("The network converges to the stored pattern that is closest to the input (a local energy minimum).")

        st.divider()
        st.subheader("Step 3: Energy Function")
        st.markdown("The network always minimizes its energy function:")
        st.latex(r"E = -\frac{1}{2} \mathbf{s}^T W \mathbf{s}")
        st.markdown("Each stored pattern is a **local minimum** in the energy landscape. The recall process is like a ball rolling downhill to the nearest valley.")

        st.divider()
        st.subheader("Storage Capacity")
        st.latex(r"P_{\max} \approx \frac{N}{2 \ln N}")
        st.markdown(f"For a {PATTERN_SIZE}-neuron network using Hebbian learning: ~**{int(PATTERN_SIZE / (2 * np.log(PATTERN_SIZE)))}** patterns can be stored reliably.")
        st.markdown("Storing too many patterns causes **spurious states** — the network recalls patterns that were never stored or gets stuck in random noise.")
