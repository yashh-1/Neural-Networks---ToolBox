import numpy as np

# ─── Defaults ───────────────────────────────────────────────
DEFAULT_LR = 0.005
DEFAULT_EPOCHS = 15
IMG_SIZE = 32


# ─── Activation functions ──────────────────────────────────
def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


# ─── im2col / col2im helpers (vectorized convolution) ─────
def _im2col(padded, kernel_size, stride, out_h, out_w):
    """Extract all patches as columns: (out_h*out_w, k*k*C)."""
    k = kernel_size
    C = padded.shape[2]
    cols = np.zeros((out_h * out_w, k * k * C))
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            hs = i * stride
            ws = j * stride
            patch = padded[hs:hs + k, ws:ws + k, :]
            cols[idx] = patch.flatten()
            idx += 1
    return cols


def _col2im(cols, padded_shape, kernel_size, stride, out_h, out_w):
    """Scatter column gradients back to padded input shape."""
    k = kernel_size
    d_padded = np.zeros(padded_shape)
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            hs = i * stride
            ws = j * stride
            patch = cols[idx].reshape(k, k, padded_shape[2])
            d_padded[hs:hs + k, ws:ws + k, :] += patch
            idx += 1
    return d_padded


# ─── Conv2D Layer (im2col-accelerated) ────────────────────
class Conv2D:
    """
    2D Convolution using im2col for fast matrix-multiply.
    Forward:  output = im2col(input) @ filters_matrix + biases
    Backward: gradient flows back via transposed multiply.
    """

    def __init__(self, num_filters, kernel_size, in_channels, stride=1, pad=1):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.in_channels = in_channels
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size, in_channels) * scale
        self.biases = np.zeros(num_filters)

    def forward(self, inp):
        self.input = inp  # (H, W, C)
        if self.pad > 0:
            padded = np.pad(inp, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
                            mode='constant')
        else:
            padded = inp
        self.padded = padded

        pH, pW, _ = padded.shape
        k = self.kernel_size
        self.out_h = (pH - k) // self.stride + 1
        self.out_w = (pW - k) // self.stride + 1

        # im2col: (out_h*out_w, k*k*C)
        self.cols = _im2col(padded, k, self.stride, self.out_h, self.out_w)

        # filters reshaped: (k*k*C, num_filters)
        W = self.filters.reshape(self.num_filters, -1).T  # (k*k*C, F)

        # Matrix multiply: (out_h*out_w, F) + biases
        out_flat = self.cols @ W + self.biases  # (N, F)

        output = out_flat.reshape(self.out_h, self.out_w, self.num_filters)
        return output

    def backward(self, d_output, lr):
        k = self.kernel_size
        # d_output: (out_h, out_w, F) → (out_h*out_w, F)
        d_flat = d_output.reshape(-1, self.num_filters)

        # Filter gradients: cols.T @ d_flat → (k*k*C, F)
        W_flat = self.filters.reshape(self.num_filters, -1).T  # (k*k*C, F)
        dW_flat = self.cols.T @ d_flat  # (k*k*C, F)
        d_biases = d_flat.sum(axis=0)

        # Input gradients: d_flat @ W.T → (out_h*out_w, k*k*C)
        d_cols = d_flat @ W_flat.T

        # col2im
        d_padded = _col2im(d_cols, self.padded.shape, k, self.stride,
                           self.out_h, self.out_w)

        if self.pad > 0:
            d_input = d_padded[self.pad:-self.pad, self.pad:-self.pad, :]
        else:
            d_input = d_padded

        # Reshape filter gradient and update
        d_filters = dW_flat.T.reshape(self.filters.shape)
        np.clip(d_filters, -5, 5, out=d_filters)
        np.clip(d_biases, -5, 5, out=d_biases)
        self.filters -= lr * d_filters
        self.biases -= lr * d_biases
        return d_input


# ─── MaxPool2D Layer (vectorized) ─────────────────────────
class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, inp):
        self.input = inp
        H, W, C = inp.shape
        p = self.pool_size
        out_h = H // p
        out_w = W // p

        # Reshape to (out_h, p, out_w, p, C) then take max over pool dims
        reshaped = inp[:out_h * p, :out_w * p, :].reshape(out_h, p, out_w, p, C)
        output = reshaped.max(axis=(1, 3))  # (out_h, out_w, C)

        # Build mask for backprop
        # Tile output back to input shape and compare
        out_tiled = np.repeat(np.repeat(output, p, axis=0), p, axis=1)
        self.mask = (inp[:out_h * p, :out_w * p, :] == out_tiled).astype(float)
        # Normalize mask so gradients distribute evenly if ties exist
        mask_sum = np.repeat(np.repeat(
            self.mask.reshape(out_h, p, out_w, p, C).sum(axis=(1, 3)),
            p, axis=0), p, axis=1)
        self.mask /= np.maximum(mask_sum, 1)

        self.out_h = out_h
        self.out_w = out_w
        return output

    def backward(self, d_output, lr=None):
        p = self.pool_size
        # Tile gradients back to input spatial size
        d_tiled = np.repeat(np.repeat(d_output, p, axis=0), p, axis=1)
        d_input = np.zeros_like(self.input)
        d_input[:self.out_h * p, :self.out_w * p, :] = self.mask * d_tiled
        return d_input


# ─── ReLU Layer ────────────────────────────────────────────
class ReLULayer:
    def forward(self, inp):
        self.input = inp
        return relu(inp)

    def backward(self, d_output, lr=None):
        return d_output * relu_deriv(self.input)


# ─── Flatten Layer ─────────────────────────────────────────
class FlattenLayer:
    def forward(self, inp):
        self.shape = inp.shape
        return inp.flatten()

    def backward(self, d_output, lr=None):
        return d_output.reshape(self.shape)


# ─── Dense (Fully Connected) Layer ─────────────────────────
class Dense:
    def __init__(self, in_size, out_size):
        scale = np.sqrt(2.0 / in_size)
        self.weights = np.random.randn(in_size, out_size) * scale
        self.biases = np.zeros(out_size)

    def forward(self, inp):
        self.input = inp
        return inp @ self.weights + self.biases

    def backward(self, d_output, lr):
        d_input = d_output @ self.weights.T
        d_weights = self.input.reshape(-1, 1) @ d_output.reshape(1, -1)
        d_biases = d_output.copy()

        np.clip(d_weights, -5, 5, out=d_weights)
        np.clip(d_biases, -5, 5, out=d_biases)
        self.weights -= lr * d_weights
        self.biases -= lr * d_biases
        return d_input


# ─── FaceCNN Model ─────────────────────────────────────────
class FaceCNN:
    """
    CNN for face classification.
        Conv1(8, 3x3) → ReLU → MaxPool(2x2)
        Conv2(16, 3x3) → ReLU → MaxPool(2x2)
        Flatten → FC1(→64) → ReLU → FC2(→num_classes)
        Softmax + Cross-Entropy
    """

    def __init__(self, num_classes, img_size=IMG_SIZE):
        self.num_classes = num_classes
        self.img_size = img_size

        self.conv1 = Conv2D(8, 3, 1, stride=1, pad=1)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPool2D(2)

        self.conv2 = Conv2D(16, 3, 8, stride=1, pad=1)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPool2D(2)

        self.flatten = FlattenLayer()

        fc1_in = (img_size // 4) * (img_size // 4) * 16
        self.fc1 = Dense(fc1_in, 64)
        self.relu3 = ReLULayer()
        self.fc2 = Dense(64, num_classes)

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
            self.fc1, self.relu3, self.fc2
        ]

    def forward(self, image):
        x = image
        for layer in self.layers:
            x = layer.forward(x)
        probs = softmax(x)
        return probs

    def get_embedding(self, image):
        x = image
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return x.copy()

    def backward(self, probs, label, lr):
        d = probs.copy()
        d[label] -= 1
        for layer in reversed(self.layers):
            d = layer.backward(d, lr)

    def predict(self, image):
        probs = self.forward(image)
        return int(np.argmax(probs)), float(np.max(probs)), probs

    def get_feature_maps(self, image):
        x = image
        maps = {}
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        maps['conv1_relu'] = x.copy()
        x = self.pool1.forward(x)
        maps['pool1'] = x.copy()
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        maps['conv2_relu'] = x.copy()
        x = self.pool2.forward(x)
        maps['pool2'] = x.copy()
        return maps

    def get_filters(self):
        return {
            'conv1': self.conv1.filters.copy(),
            'conv2': self.conv2.filters.copy()
        }


# ─── Image helpers ─────────────────────────────────────────
def preprocess_image(image, img_size=IMG_SIZE):
    """Convert to grayscale, resize, normalize to [0,1]. Returns (img_size, img_size, 1)."""
    from PIL import Image as PILImage
    import io

    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] in (3, 4):
            gray = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
        elif len(image.shape) == 2:
            gray = image
        else:
            gray = image[:, :, 0]
        pil_img = PILImage.fromarray(gray, mode='L')
    elif isinstance(image, bytes):
        pil_img = PILImage.open(io.BytesIO(image)).convert('L')
    else:
        pil_img = image.convert('L')

    pil_img = pil_img.resize((img_size, img_size), PILImage.BILINEAR)
    arr = np.array(pil_img, dtype=np.float64) / 255.0
    return arr.reshape(img_size, img_size, 1)


def detect_face_pil(image_bytes):
    """
    Simple face region extraction: centre-crop the largest square region.
    For proper face detection, OpenCV Haar cascade is used if available.
    """
    from PIL import Image as PILImage
    import io

    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(pil_img)

    try:
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = img_array[y:y + h, x:x + w]
            return face_crop, (x, y, w, h), img_array
    except ImportError:
        pass

    # Fallback: centre-crop
    h, w = img_array.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    face_crop = img_array[y0:y0 + s, x0:x0 + s]
    return face_crop, (x0, y0, s, s), img_array


# ─── Data Augmentation ─────────────────────────────────────
def augment_image(img):
    """Generate augmented copies: flip, brightness, noise."""
    augmented = [img]

    # Horizontal flip
    augmented.append(img[:, ::-1, :])

    # Brightness jitter
    for delta in [-0.1, 0.1]:
        aug = np.clip(img + delta, 0, 1)
        augmented.append(aug)

    # Gaussian noise
    noise = np.random.randn(*img.shape) * 0.03
    augmented.append(np.clip(img + noise, 0, 1))

    return augmented


# ─── Training function ─────────────────────────────────────
def train_cnn(images, labels, num_classes, lr=DEFAULT_LR,
              epochs=DEFAULT_EPOCHS, img_size=IMG_SIZE,
              progress_callback=None):
    """
    Train CNN on face images.
    images:  list of preprocessed (img_size, img_size, 1) arrays
    labels:  list of int class labels
    Returns: (model, loss_history, acc_history)
    """
    model = FaceCNN(num_classes, img_size=img_size)

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        order = list(range(len(images)))
        np.random.shuffle(order)

        for idx in order:
            probs = model.forward(images[idx])
            label = labels[idx]

            eps = 1e-7
            loss = -np.log(probs[label] + eps)
            total_loss += loss

            if np.argmax(probs) == label:
                correct += 1

            model.backward(probs, label, lr)

        avg_loss = total_loss / max(len(images), 1)
        accuracy = correct / max(len(images), 1)
        loss_history.append(float(avg_loss))
        acc_history.append(float(accuracy))

        if progress_callback:
            progress_callback(epoch + 1, epochs, float(avg_loss), float(accuracy))

    return model, loss_history, acc_history
