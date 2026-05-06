import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Theme helper ────────────────────────────────────────────
def _layout(title="", theme="Dark"):
    is_dark = theme == "Dark"
    return dict(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a1d24" if is_dark else "#f8f9fa",
        margin=dict(l=50, r=30, t=50, b=50),
        font=dict(color="#fafafa" if is_dark else "#111"),
    )


# ─── Decision Boundary (2D classifiers) ─────────────────────
def plot_decision_boundary(X, Y, predict_fn, title="Decision Boundary", theme="Dark", resolution=150):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1 = np.linspace(x1_min, x1_max, resolution)
    xx2 = np.linspace(x2_min, x2_max, resolution)
    grid_x1, grid_x2 = np.meshgrid(xx1, xx2)

    Z = np.array([predict_fn(p[0], p[1])[1] for p in np.c_[grid_x1.ravel(), grid_x2.ravel()]])
    Z = Z.reshape(grid_x1.shape)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xx1, y=xx2, z=Z, colorscale=[[0, "#d73027"], [0.5, "#ffffbf"], [1, "#1a9850"]],
        opacity=0.4, showscale=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Contour(
        x=xx1, y=xx2, z=Z, contours=dict(start=0.5, end=0.5, size=0),
        line=dict(color="gray", width=2, dash="dash"), showscale=False, hoverinfo="skip"
    ))
    # Class 0
    mask0 = Y == 0
    fig.add_trace(go.Scatter(
        x=X[mask0, 0], y=X[mask0, 1], mode="markers", name="Class 0",
        marker=dict(color="#d73027", size=10, line=dict(width=1.5, color="black"))
    ))
    # Class 1
    mask1 = Y == 1
    fig.add_trace(go.Scatter(
        x=X[mask1, 0], y=X[mask1, 1], mode="markers", name="Class 1",
        marker=dict(color="#1a9850", size=10, line=dict(width=1.5, color="black"))
    ))
    fig.update_layout(**_layout(title, theme), xaxis_title="Feature 1", yaxis_title="Feature 2")
    return fig


# ─── Confidence Heatmap (2D) ────────────────────────────────
def plot_confidence_heatmap(X, Y, predict_fn, title="Prediction Confidence Map", theme="Dark", resolution=150):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1 = np.linspace(x1_min, x1_max, resolution)
    xx2 = np.linspace(x2_min, x2_max, resolution)
    grid_x1, grid_x2 = np.meshgrid(xx1, xx2)

    Z = np.array([predict_fn(p[0], p[1])[0] for p in np.c_[grid_x1.ravel(), grid_x2.ravel()]])
    Z = Z.reshape(grid_x1.shape)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xx1, y=xx2, z=Z, colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar=dict(title="Confidence"), hovertemplate="x1: %{x:.2f}<br>x2: %{y:.2f}<br>Score: %{z:.3f}<extra></extra>"
    ))
    fig.add_trace(go.Contour(
        x=xx1, y=xx2, z=Z, contours=dict(start=0.5, end=0.5, size=0),
        line=dict(color="black", width=2, dash="dash"), showscale=False, hoverinfo="skip"
    ))
    mask0, mask1 = Y == 0, Y == 1
    fig.add_trace(go.Scatter(x=X[mask0, 0], y=X[mask0, 1], mode="markers", name="Class 0",
        marker=dict(color="#d73027", size=10, line=dict(width=1.5, color="black"))))
    fig.add_trace(go.Scatter(x=X[mask1, 0], y=X[mask1, 1], mode="markers", name="Class 1",
        marker=dict(color="#1a9850", size=10, line=dict(width=1.5, color="black"))))
    fig.update_layout(**_layout(title, theme), xaxis_title="Feature 1", yaxis_title="Feature 2")
    return fig


# ─── Weight Heatmap ──────────────────────────────────────────
def plot_weight_heatmap_mlp(weights, title="MLP Weight Heatmap", theme="Dark"):
    w1, w2, w3, w4, w5, w6, bh1, bh2, bo = weights

    ih = np.array([[w1, w2], [w3, w4]])
    ho = np.array([[w5, bh1], [w6, bh2]])

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Input → Hidden", "Hidden → Output"],
                        horizontal_spacing=0.15)

    ih_text = [[f"w1={w1:.3f}", f"w2={w2:.3f}"], [f"w3={w3:.3f}", f"w4={w4:.3f}"]]
    fig.add_trace(go.Heatmap(
        z=ih, x=["h1", "h2"], y=["x1", "x2"], colorscale="RdBu_r", zmid=0,
        text=ih_text, texttemplate="%{text}", textfont=dict(size=14),
        showscale=True, colorbar=dict(x=0.42, len=0.9)
    ), row=1, col=1)

    ho_text = [[f"w5={w5:.3f}", f"bh1={bh1:.3f}"], [f"w6={w6:.3f}", f"bh2={bh2:.3f}"]]
    fig.add_trace(go.Heatmap(
        z=ho, x=["Weight", "Bias"], y=["h1→o", "h2→o"], colorscale="RdBu_r", zmid=0,
        text=ho_text, texttemplate="%{text}", textfont=dict(size=14),
        showscale=True, colorbar=dict(x=1.0, len=0.9)
    ), row=1, col=2)

    fig.update_layout(**_layout(title, theme), height=350)
    return fig


# ─── Confusion Matrix ───────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", theme="Dark"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    cm = np.array([[tn, fp], [fn, tp]])
    labels = [[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]]

    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: 0", "Pred: 1"], y=["True: 0", "True: 1"],
        colorscale="Blues", showscale=True,
        text=labels, texttemplate="%{text}", textfont=dict(size=18),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig.update_layout(**_layout(title, theme), height=400,
                      xaxis_title="Predicted", yaxis_title="Actual")

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
               "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return fig, metrics


# ─── Activation Distribution ────────────────────────────────
def plot_activation_distribution(X, weights, title="Neuron Activation Distribution", theme="Dark"):
    w1, w2, w3, w4, w5, w6, bh1, bh2, bo = weights
    X = np.array(X, dtype=float)

    h1_vals, h2_vals, o_vals = [], [], []
    for row in X:
        x1, x2 = row[0], row[1]
        zh1 = x1 * w1 + x2 * w3 + bh1
        zh2 = x1 * w2 + x2 * w4 + bh2
        h1 = 1 / (1 + np.exp(-np.clip(zh1, -500, 500)))
        h2 = 1 / (1 + np.exp(-np.clip(zh2, -500, 500)))
        zo = h1 * w5 + h2 * w6 + bo
        o = 1 / (1 + np.exp(-np.clip(zo, -500, 500)))
        h1_vals.append(h1); h2_vals.append(h2); o_vals.append(o)

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Hidden Neuron 1 (h1)", "Hidden Neuron 2 (h2)", "Output Neuron (o)"])

    fig.add_trace(go.Histogram(x=h1_vals, nbinsx=15, marker_color="#2196F3", name="h1", opacity=0.85), row=1, col=1)
    fig.add_trace(go.Histogram(x=h2_vals, nbinsx=15, marker_color="#FF9800", name="h2", opacity=0.85), row=1, col=2)
    fig.add_trace(go.Histogram(x=o_vals, nbinsx=15, marker_color="#4CAF50", name="output", opacity=0.85), row=1, col=3)

    for col in [1, 2, 3]:
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", row=1, col=col)

    fig.update_layout(**_layout(title, theme), height=380, showlegend=False)
    return fig


# ─── Loss Curve ──────────────────────────────────────────────
def plot_loss_curve(loss_history, title="Training Loss", ylabel="Loss", theme="Dark"):
    epochs = list(range(1, len(loss_history) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=loss_history, mode="lines", name="Loss",
        line=dict(color="#E91E63", width=2.5),
        fill="tozeroy", fillcolor="rgba(233,30,99,0.12)",
        hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>"
    ))
    fig.update_layout(**_layout(title, theme), xaxis_title="Epoch", yaxis_title=ylabel)
    return fig


# ─── Regression Plot (MSE single variable) ──────────────────
def plot_regression_line(X, y, w, b, title="Linear Regression Fit", theme="Dark"):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    x_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
    y_line = w * x_line + b

    fig = go.Figure()

    # Residual lines
    for xi, yi in zip(X, y):
        y_pred = w * xi + b
        fig.add_trace(go.Scatter(
            x=[xi, xi], y=[yi, y_pred], mode="lines",
            line=dict(color="gray", width=1, dash="dash"), showlegend=False, hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name=f"y = {w:.2f}x + {b:.2f}",
        line=dict(color="#E91E63", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=X, y=y, mode="markers", name="Data Points",
        marker=dict(color="#2196F3", size=10, line=dict(width=1.5, color="black")),
        hovertemplate="X: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(**_layout(title, theme), xaxis_title="X", yaxis_title="y")
    return fig


# ─── 3D Regression Surface (MSE dual) ───────────────────────
def plot_regression_3d(X1, X2, y, w1, w2, b, title="3D Regression Surface", theme="Dark"):
    X1 = np.array(X1, dtype=float)
    X2 = np.array(X2, dtype=float)
    y = np.array(y, dtype=float)

    x1_range = np.linspace(X1.min() - 0.5, X1.max() + 0.5, 30)
    x2_range = np.linspace(X2.min() - 0.5, X2.max() + 0.5, 30)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    yy = w1 * xx1 + w2 * xx2 + b

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xx1, y=xx2, z=yy, colorscale="RdYlGn", opacity=0.6,
        showscale=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=X1, y=X2, z=y, mode="markers", name="Data",
        marker=dict(color="#2196F3", size=6, line=dict(width=1, color="black")),
        hovertemplate="X1: %{x:.2f}<br>X2: %{y:.2f}<br>y: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(**_layout(title, theme), height=550,
        scene=dict(xaxis_title="X1", yaxis_title="X2", zaxis_title="y",
                   bgcolor="#1a1d24" if theme == "Dark" else "#f8f9fa"))
    return fig


# ─── Residual Plot ───────────────────────────────────────────
def plot_residual(y_true, y_pred, title="Residual Plot", theme="Dark"):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#E91E63", line_width=2)
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers", name="Residuals",
        marker=dict(color="#2196F3", size=10, line=dict(width=1.5, color="black")),
        hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"
    ))
    fig.update_layout(**_layout(title, theme),
                      xaxis_title="Predicted Value", yaxis_title="Residual (Actual - Predicted)")
    return fig


# ─── Perceptron Decision Line ───────────────────────────────
def plot_perceptron_boundary(X, Y, weights, title="Perceptron Decision Boundary", theme="Dark"):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    w1, w2, b = weights

    fig = go.Figure()

    mask0, mask1 = Y == 0, Y == 1
    fig.add_trace(go.Scatter(x=X[mask0, 0], y=X[mask0, 1], mode="markers", name="Class 0",
        marker=dict(color="#d73027", size=10, line=dict(width=1.5, color="black"))))
    fig.add_trace(go.Scatter(x=X[mask1, 0], y=X[mask1, 1], mode="markers", name="Class 1",
        marker=dict(color="#1a9850", size=10, line=dict(width=1.5, color="black"))))

    if abs(w2) > 1e-8:
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_line = np.linspace(x1_min, x1_max, 100)
        x2_line = -(w1 * x1_line + b) / w2
        fig.add_trace(go.Scatter(
            x=x1_line, y=x2_line, mode="lines",
            name=f"{w1:.2f}·x1 + {w2:.2f}·x2 + {b:.2f} = 0",
            line=dict(color="#E91E63", width=2.5)
        ))

    fig.update_layout(**_layout(title, theme), xaxis_title="Feature 1", yaxis_title="Feature 2")
    return fig


# ─── Sentiment Distribution Bar ─────────────────────────────
def plot_sentiment_distribution(labels, title="Dataset Sentiment Distribution", theme="Dark"):
    neg = labels.count(0)
    pos = labels.count(1)

    fig = go.Figure(go.Bar(
        x=["Negative (0)", "Positive (1)"], y=[neg, pos],
        marker_color=["#d73027", "#1a9850"], text=[neg, pos], textposition="outside",
        textfont=dict(size=16), hovertemplate="%{x}: %{y}<extra></extra>"
    ))
    fig.update_layout(**_layout(title, theme), yaxis_title="Count")
    return fig


# ─── Loss + Accuracy dual chart (RNN) ───────────────────────
def plot_loss_accuracy(loss_hist, acc_hist, title="Training Progress", theme="Dark"):
    epochs = list(range(1, len(loss_hist) + 1))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Training Loss", "Training Accuracy"])

    fig.add_trace(go.Scatter(
        x=epochs, y=loss_hist, mode="lines", name="Loss",
        line=dict(color="#E91E63", width=2.5),
        fill="tozeroy", fillcolor="rgba(233,30,99,0.12)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=[a * 100 for a in acc_hist], mode="lines", name="Accuracy",
        line=dict(color="#4CAF50", width=2.5),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.12)"
    ), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_layout(**_layout(title, theme), showlegend=False)
    return fig


# ─── Word Frequency Bar (RNN) ───────────────────────────────
def plot_word_frequency(texts, labels, tokenize_fn, title="Top Words by Sentiment", theme="Dark"):
    pos_words, neg_words = {}, {}
    for t, l in zip(texts, labels):
        for w in tokenize_fn(t):
            bucket = pos_words if l == 1 else neg_words
            bucket[w] = bucket.get(w, 0) + 1

    pos_top = sorted(pos_words.items(), key=lambda x: -x[1])[:15]
    neg_top = sorted(neg_words.items(), key=lambda x: -x[1])[:15]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Top Positive Words", "Top Negative Words"])

    if pos_top:
        words, counts = zip(*reversed(pos_top))
        fig.add_trace(go.Bar(y=list(words), x=list(counts), orientation="h",
            marker_color="#1a9850", name="Positive", hovertemplate="%{y}: %{x}<extra></extra>"), row=1, col=1)
    if neg_top:
        words, counts = zip(*reversed(neg_top))
        fig.add_trace(go.Bar(y=list(words), x=list(counts), orientation="h",
            marker_color="#d73027", name="Negative", hovertemplate="%{y}: %{x}<extra></extra>"), row=1, col=2)

    fig.update_layout(**_layout(title, theme), height=500, showlegend=False)
    return fig
