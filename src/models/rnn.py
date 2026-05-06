import numpy as np
import re

DEFAULT_HIDDEN = 64
DEFAULT_LR = 0.01  # Reduced from 0.05 for stability
DEFAULT_EPOCHS = 50

# ─── Text helpers ───────────────────────────────────────────
def tokenize(text):
    """Lowercase, strip punctuation, split into words."""
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return text.split()


def build_vocab(texts):
    """Build word→index mapping from list of strings."""
    vocab = {"<UNK>": 0}
    for text in texts:
        for w in tokenize(text):
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def text_to_indices(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(w, 0) for w in tokens]


# ─── Activation functions ───────────────────────────────────
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(s):
    return s * (1 - s)


# ─── RNN class (many-to-one for binary classification) ─────
class SentimentRNN:
    """
    Vanilla RNN:  h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
    Output:       y   = sigmoid(W_hy · h_T + b_y)
    Trained with BPTT (back-propagation through time).
    """

    def __init__(self, vocab_size, hidden_size=DEFAULT_HIDDEN, embed_size=32):
        scale_e = 0.01
        scale_h = np.sqrt(1.0 / hidden_size)
        scale_x = np.sqrt(1.0 / embed_size)

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Embedding matrix
        self.W_embed = np.random.randn(vocab_size, embed_size) * scale_e

        # RNN weights
        self.W_xh = np.random.randn(embed_size, hidden_size) * scale_x
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_h
        self.b_h  = np.zeros((1, hidden_size))

        # Output weights
        self.W_hy = np.random.randn(hidden_size, 1) * scale_h
        self.b_y  = np.zeros((1, 1))

    # ── Forward ──────────────────────────────────────────────
    def forward(self, indices):
        """Run forward pass; return (output, cache)."""
        T = len(indices)
        h_states = {-1: np.zeros((1, self.hidden_size))}
        x_embeds = {}
        z_states = {}

        for t in range(T):
            x_embeds[t] = self.W_embed[indices[t]].reshape(1, -1)
            z_states[t] = x_embeds[t] @ self.W_xh + h_states[t - 1] @ self.W_hh + self.b_h
            h_states[t] = tanh(z_states[t])

        logit = h_states[T - 1] @ self.W_hy + self.b_y
        y_pred = sigmoid(logit)

        cache = (indices, x_embeds, h_states, z_states, T)
        return y_pred, cache

    # ── Backward (BPTT) ─────────────────────────────────────
    def backward(self, y_pred, y_true, cache, lr, clip=5.0):
        indices, x_embeds, h_states, z_states, T = cache

        # Output gradient  (binary cross-entropy derivative)
        dL_dy = y_pred - y_true                       # (1,1)

        dW_hy = h_states[T - 1].T @ dL_dy
        db_y  = dL_dy
        dh    = dL_dy @ self.W_hy.T                   # (1, hidden)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)
        dW_embed_updates = {}

        for t in reversed(range(T)):
            dtanh = dh * tanh_deriv(z_states[t])      # (1, hidden)
            dW_xh += x_embeds[t].T @ dtanh
            dW_hh += h_states[t - 1].T @ dtanh
            db_h  += dtanh

            dx = dtanh @ self.W_xh.T                  # (1, embed_size)
            idx = indices[t]
            if idx in dW_embed_updates:
                dW_embed_updates[idx] += dx.flatten()
            else:
                dW_embed_updates[idx] = dx.flatten()

            dh = dtanh @ self.W_hh.T

        # Gradient clipping
        for grad in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(grad, -clip, clip, out=grad)

        # Update
        self.W_xh -= lr * dW_xh
        self.W_hh -= lr * dW_hh
        self.b_h  -= lr * db_h
        self.W_hy -= lr * dW_hy
        self.b_y  -= lr * db_y

        for idx, grad in dW_embed_updates.items():
            np.clip(grad, -clip, clip, out=grad)
            self.W_embed[idx] -= lr * grad

    # ── Predict single text ─────────────────────────────────
    def predict_text(self, text, vocab):
        indices = text_to_indices(text, vocab)
        if len(indices) == 0:
            return 0.5, "Neutral"
        y_pred, _ = self.forward(indices)
        score = float(y_pred[0, 0])
        label = "Positive" if score >= 0.5 else "Negative"
        return score, label


# ─── Training function (called from frontend) ──────────────
def train_rnn(texts, labels, hidden_size=DEFAULT_HIDDEN, lr=DEFAULT_LR, epochs=DEFAULT_EPOCHS):
    """
    texts:  list of strings
    labels: list of 0/1
    Returns: (model, vocab, loss_history, acc_history)
    """
    vocab = build_vocab(texts)
    model = SentimentRNN(len(vocab), hidden_size=hidden_size)

    indices_list = [text_to_indices(t, vocab) for t in texts]
    labels_arr   = [float(l) for l in labels]

    loss_history = []
    acc_history  = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        order = list(range(len(texts)))
        np.random.shuffle(order)

        for i in order:
            if len(indices_list[i]) == 0:
                continue
            y_pred, cache = model.forward(indices_list[i])
            y_true = labels_arr[i]

            # Binary cross-entropy loss
            eps = 1e-7
            p = float(y_pred[0, 0])
            loss = -(y_true * np.log(p + eps) + (1 - y_true) * np.log(1 - p + eps))
            total_loss += loss

            pred_class = 1 if p >= 0.5 else 0
            if pred_class == int(y_true):
                correct += 1

            model.backward(y_pred, y_true, cache, lr)

        avg_loss = total_loss / max(len(texts), 1)
        accuracy = correct / max(len(texts), 1)
        loss_history.append(avg_loss)
        acc_history.append(accuracy)

    return model, vocab, loss_history, acc_history
