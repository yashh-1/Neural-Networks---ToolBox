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
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(s):
    return s * (1 - s)


def tanh_fn(x):
    return np.tanh(x)


def tanh_deriv(t):
    return 1 - t ** 2


# ─── LSTM class (many-to-one for binary classification) ────
class SentimentLSTM:
    """
    LSTM cell at each time step:
        f_t = sigmoid(x_t · W_xf + h_{t-1} · W_hf + b_f)   (forget gate)
        i_t = sigmoid(x_t · W_xi + h_{t-1} · W_hi + b_i)   (input gate)
        g_t = tanh(x_t · W_xg + h_{t-1} · W_hg + b_g)      (candidate)
        o_t = sigmoid(x_t · W_xo + h_{t-1} · W_ho + b_o)   (output gate)
        c_t = f_t * c_{t-1} + i_t * g_t                      (cell state)
        h_t = o_t * tanh(c_t)                                 (hidden state)

    Output: y = sigmoid(h_T · W_hy + b_y)
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

        # Forget gate weights
        self.W_xf = np.random.randn(embed_size, hidden_size) * scale_x
        self.W_hf = np.random.randn(hidden_size, hidden_size) * scale_h
        self.b_f = np.ones((1, hidden_size))  # bias init to 1 (remember by default)

        # Input gate weights
        self.W_xi = np.random.randn(embed_size, hidden_size) * scale_x
        self.W_hi = np.random.randn(hidden_size, hidden_size) * scale_h
        self.b_i = np.zeros((1, hidden_size))

        # Candidate gate weights
        self.W_xg = np.random.randn(embed_size, hidden_size) * scale_x
        self.W_hg = np.random.randn(hidden_size, hidden_size) * scale_h
        self.b_g = np.zeros((1, hidden_size))

        # Output gate weights
        self.W_xo = np.random.randn(embed_size, hidden_size) * scale_x
        self.W_ho = np.random.randn(hidden_size, hidden_size) * scale_h
        self.b_o = np.zeros((1, hidden_size))

        # Final output weights
        self.W_hy = np.random.randn(hidden_size, 1) * scale_h
        self.b_y = np.zeros((1, 1))

    # ── Forward ──────────────────────────────────────────────
    def forward(self, indices):
        T = len(indices)
        h_states = {-1: np.zeros((1, self.hidden_size))}
        c_states = {-1: np.zeros((1, self.hidden_size))}
        x_embeds = {}
        f_gates = {}
        i_gates = {}
        g_gates = {}
        o_gates = {}
        c_tanh = {}

        for t in range(T):
            x_embeds[t] = self.W_embed[indices[t]].reshape(1, -1)
            h_prev = h_states[t - 1]
            c_prev = c_states[t - 1]
            xt = x_embeds[t]

            f_gates[t] = sigmoid(xt @ self.W_xf + h_prev @ self.W_hf + self.b_f)
            i_gates[t] = sigmoid(xt @ self.W_xi + h_prev @ self.W_hi + self.b_i)
            g_gates[t] = tanh_fn(xt @ self.W_xg + h_prev @ self.W_hg + self.b_g)
            o_gates[t] = sigmoid(xt @ self.W_xo + h_prev @ self.W_ho + self.b_o)

            c_states[t] = f_gates[t] * c_prev + i_gates[t] * g_gates[t]
            c_tanh[t] = tanh_fn(c_states[t])
            h_states[t] = o_gates[t] * c_tanh[t]

        logit = h_states[T - 1] @ self.W_hy + self.b_y
        y_pred = sigmoid(logit)

        cache = (indices, x_embeds, h_states, c_states, f_gates, i_gates, g_gates, o_gates, c_tanh, T)
        return y_pred, cache

    # ── Backward (BPTT) ─────────────────────────────────────
    def backward(self, y_pred, y_true, cache, lr, clip=5.0):
        indices, x_embeds, h_states, c_states, f_gates, i_gates, g_gates, o_gates, c_tanh, T = cache

        dL_dy = y_pred - y_true

        dW_hy = h_states[T - 1].T @ dL_dy
        db_y = dL_dy
        dh = dL_dy @ self.W_hy.T
        dc = np.zeros_like(c_states[0])

        # Accumulate gradients
        dW_xf = np.zeros_like(self.W_xf)
        dW_hf = np.zeros_like(self.W_hf)
        db_f = np.zeros_like(self.b_f)
        dW_xi = np.zeros_like(self.W_xi)
        dW_hi = np.zeros_like(self.W_hi)
        db_i = np.zeros_like(self.b_i)
        dW_xg = np.zeros_like(self.W_xg)
        dW_hg = np.zeros_like(self.W_hg)
        db_g = np.zeros_like(self.b_g)
        dW_xo = np.zeros_like(self.W_xo)
        dW_ho = np.zeros_like(self.W_ho)
        db_o = np.zeros_like(self.b_o)
        dW_embed_updates = {}

        for t in reversed(range(T)):
            # Gradients through h_t = o_t * tanh(c_t)
            do = dh * c_tanh[t]
            dc += dh * o_gates[t] * tanh_deriv(c_tanh[t])

            # Gradients through c_t = f_t * c_{t-1} + i_t * g_t
            df = dc * c_states[t - 1]
            di = dc * g_gates[t]
            dg = dc * i_gates[t]
            dc_prev = dc * f_gates[t]

            # Gate derivatives (through sigmoid/tanh)
            df_raw = df * sigmoid_deriv(f_gates[t])
            di_raw = di * sigmoid_deriv(i_gates[t])
            dg_raw = dg * tanh_deriv(g_gates[t])
            do_raw = do * sigmoid_deriv(o_gates[t])

            xt = x_embeds[t]
            h_prev = h_states[t - 1]

            # Forget gate gradients
            dW_xf += xt.T @ df_raw
            dW_hf += h_prev.T @ df_raw
            db_f += df_raw

            # Input gate gradients
            dW_xi += xt.T @ di_raw
            dW_hi += h_prev.T @ di_raw
            db_i += di_raw

            # Candidate gate gradients
            dW_xg += xt.T @ dg_raw
            dW_hg += h_prev.T @ dg_raw
            db_g += dg_raw

            # Output gate gradients
            dW_xo += xt.T @ do_raw
            dW_ho += h_prev.T @ do_raw
            db_o += do_raw

            # Gradients to previous hidden state
            dh = df_raw @ self.W_hf.T + di_raw @ self.W_hi.T + dg_raw @ self.W_hg.T + do_raw @ self.W_ho.T
            dc = dc_prev

            # Embedding gradient
            dx = df_raw @ self.W_xf.T + di_raw @ self.W_xi.T + dg_raw @ self.W_xg.T + do_raw @ self.W_xo.T
            idx = indices[t]
            if idx in dW_embed_updates:
                dW_embed_updates[idx] += dx.flatten()
            else:
                dW_embed_updates[idx] = dx.flatten()

        # Gradient clipping
        all_grads = [dW_xf, dW_hf, db_f, dW_xi, dW_hi, db_i,
                     dW_xg, dW_hg, db_g, dW_xo, dW_ho, db_o, dW_hy, db_y]
        for grad in all_grads:
            np.clip(grad, -clip, clip, out=grad)

        # Update weights
        self.W_xf -= lr * dW_xf
        self.W_hf -= lr * dW_hf
        self.b_f -= lr * db_f
        self.W_xi -= lr * dW_xi
        self.W_hi -= lr * dW_hi
        self.b_i -= lr * db_i
        self.W_xg -= lr * dW_xg
        self.W_hg -= lr * dW_hg
        self.b_g -= lr * db_g
        self.W_xo -= lr * dW_xo
        self.W_ho -= lr * dW_ho
        self.b_o -= lr * db_o
        self.W_hy -= lr * dW_hy
        self.b_y -= lr * db_y

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
def train_lstm(texts, labels, hidden_size=DEFAULT_HIDDEN, lr=DEFAULT_LR, epochs=DEFAULT_EPOCHS):
    vocab = build_vocab(texts)
    model = SentimentLSTM(len(vocab), hidden_size=hidden_size)

    indices_list = [text_to_indices(t, vocab) for t in texts]
    labels_arr = [float(l) for l in labels]

    loss_history = []
    acc_history = []

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
