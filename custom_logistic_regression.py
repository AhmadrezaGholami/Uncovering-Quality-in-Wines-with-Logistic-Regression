import numpy as np
class LogisticRegression_Scratch:
    def __init__(self, learning_rate=0.01, epochs=1000, use_ova=False, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_ova = use_ova
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.classes = None
        self.classifiers = None  # Initialize classifiers for OvA

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def initialize_params(self, n_features, n_classes=1):
        if n_classes == 1:
            self.weights = np.zeros(n_features)
            self.bias = 0
        else:
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros(n_classes)

    def binary_cross_entropy_loss(self, X, y):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
        return loss

    def multi_class_cross_entropy_loss(self, X, y):
        n_classes = len(np.unique(y))
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(linear_model)
        y_one_hot = np.eye(n_classes)[y - 1]  # Adjust labels to be zero-indexed
        loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-9), axis=1))
        return loss

    def update_weights_binary(self, X, y):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        gradient = np.dot(X.T, (y_pred - y)) / y.size
        self.weights -= self.learning_rate * gradient
        self.bias -= self.learning_rate * np.mean(y_pred - y)

    def update_weights_multiclass(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(linear_model)
        n_classes = len(np.unique(y))
        y_one_hot = np.eye(n_classes)[y - 1]  # Adjust labels to be zero-indexed

        # Calculate the gradient
        gradient = np.dot(X.T, (y_pred - y_one_hot)) / y.shape[0]
        self.weights -= self.learning_rate * gradient
        self.bias -= self.learning_rate * np.mean(y_pred - y_one_hot, axis=0)

    def fit_binary(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_params(n_features, n_classes=1)

        for epoch in range(self.epochs):
            self.update_weights_binary(X, y)

    def fit_multiclass(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.initialize_params(n_features, n_classes=n_classes)

        for epoch in range(self.epochs):
            self.update_weights_multiclass(X, y)

    def predict_binary(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.where(y_pred >= self.threshold, 1, 0)

    def predict_multiclass(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(linear_model)
        return np.argmax(y_pred, axis=1) + 1  # Adjust back to one-indexing

    def fit(self, X, y):
        self.classes = np.unique(y)

        # Determine if it's binary or multi-class classification
        if len(self.classes) == 2:
            # Binary classification using BCE loss
            self.fit_binary(X, y)
        else:
            # Multi-class classification
            if self.use_ova:
                # One-vs-All (OvA) approach with binary classifiers for each class
                self.classifiers = {}
                for cls in self.classes:
                    print(f"Training classifier for class {cls}...")
                    y_binary = np.where(y == cls, 1, 0)
                    classifier = LogisticRegression_Scratch(
                        learning_rate=self.learning_rate,
                        epochs=self.epochs,
                        use_ova=False,
                        threshold=self.threshold
                    )
                    classifier.fit_binary(X, y_binary)
                    self.classifiers[cls] = classifier
            else:
                # Multi-class direct approach using MCE loss
                self.fit_multiclass(X, y)

    def predict(self, X):
        if len(self.classes) == 2:
            # Binary classification
            return self.predict_binary(X)
        else:
            # Multi-class classification
            if self.use_ova:
                # One-vs-All prediction by aggregating each classifier's result
                predictions = np.zeros((X.shape[0], len(self.classes)))
                for idx, cls in enumerate(self.classes):
                    predictions[:, idx] = self.classifiers[cls].predict_binary(X)
                return self.classes[np.argmax(predictions, axis=1)]
            else:
                # Multi-class direct prediction using softmax
                return self.predict_multiclass(X)
    # Method to save model
    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)