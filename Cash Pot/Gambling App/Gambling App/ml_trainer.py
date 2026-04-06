import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import joblib
import os

class LotteryMLTrainer:
    def __init__(self):
        self.sequence_length = 10 # Look at last 10 draws to predict next
        self.model = None
        self.model_path = 'trained_models/lstm_model.h5'
        self.model_meta_path = 'trained_models/lstm_meta.pkl'
        self.temperature = 1.0

        # Create models directory if it doesn't exist
        os.makedirs('trained_models', exist_ok=True)

    def prepare_sequences(self, numbers):
        """Create supervised sequences for next-draw classification."""
        if len(numbers) < self.sequence_length + 1:
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(len(numbers) - self.sequence_length):
            sequence = numbers[i:i + self.sequence_length]
            target = numbers[i + self.sequence_length]  # next draw number (1..36)
            X.append(sequence)
            y.append(target)

        return np.array(X), np.array(y)

    def create_lstm_model(self, input_shape):
        """Create LSTM model for 36-class probability prediction."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(48, activation='relu'),
            Dense(36, activation='softmax') # Probability for each number 1..36
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def train_models(self, df, force_retrain=False):
        """Train LSTM next-number classifier."""
        # DB rows are typically newest-first; sequences must be oldest → newest.
        numbers = df['numbers'].iloc[::-1].tolist()

        if len(numbers) < 20: # Need minimum data
            print("Insufficient data for training. Need at least 20 draws.")
            return False
        
        # Prepare sequences
        X, y = self.prepare_sequences(numbers)

        if len(X) == 0:
            print("Not enough sequences for training.")
            return False
        
        # Reshape for LSTM
        X_lstm = X.reshape((X.shape[0], X.shape[1], 1)).astype('float32') / 36.0
        y_idx = np.clip(y.astype(int) - 1, 0, 35)
        y_cat = to_categorical(y_idx, num_classes=36)

        # Train LSTM Model
        print("Training LSTM model...")
        self.model = self.create_lstm_model((X_lstm.shape[1], 1))

        history = self.model.fit(
            X_lstm, y_cat,
            epochs=40,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )

        # Calibrate probability sharpness on a holdout slice using temperature scaling.
        val_size = max(20, int(0.2 * len(X_lstm)))
        X_val = X_lstm[-val_size:]
        y_val = y_idx[-val_size:]
        val_probs = self.model.predict(X_val, verbose=0)
        self.temperature = self._fit_temperature(val_probs, y_val)

        # Save model + metadata for compatibility checks
        self.model.save(self.model_path)
        joblib.dump(
            {
                'task': 'next_number_classification',
                'classes': 36,
                'temperature': float(self.temperature),
            },
            self.model_meta_path
        )

        print(f"Model training completed. Final loss: {history.history['loss'][-1]:.4f}")
        return True
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                if os.path.exists(self.model_meta_path):
                    meta = joblib.load(self.model_meta_path)
                    self.temperature = float(meta.get('temperature', 1.0))
                else:
                    self.temperature = 1.0
                return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        return False

    def _fit_temperature(self, probs, y_idx):
        """Grid-search temperature minimizing NLL on validation predictions."""
        probs = np.clip(np.array(probs, dtype=float), 1e-8, 1.0)
        y_idx = np.array(y_idx, dtype=int)
        temps = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
        best_t = 1.0
        best_nll = float('inf')
        for t in temps:
            adj = np.power(probs, 1.0 / t)
            adj = adj / np.clip(adj.sum(axis=1, keepdims=True), 1e-8, None)
            p_true = np.clip(adj[np.arange(len(y_idx)), y_idx], 1e-8, 1.0)
            nll = float(-np.mean(np.log(p_true)))
            if nll < best_nll:
                best_nll = nll
                best_t = t
        return best_t

    def predict_next_probabilities(self, recent_numbers):
        """Predict probability distribution over numbers 1..36."""
        if self.model is None and not self.load_models():
            return None
        
        # Older regression models output shape (1, 5). Ignore them in this phase.
        out_shape = getattr(self.model, 'output_shape', None)
        if not out_shape or out_shape[-1] != 36:
            return None
        
        if len(recent_numbers) < self.sequence_length:
            padded = [0] * (self.sequence_length - len(recent_numbers)) + recent_numbers
        else:
            padded = recent_numbers[-self.sequence_length:]

        input_seq = (np.array(padded).reshape(1, self.sequence_length, 1).astype('float32')) / 36.0
        probs = self.model.predict(input_seq, verbose=0)[0]
        t = float(getattr(self, 'temperature', 1.0))
        if t != 1.0:
            probs = np.power(np.clip(probs, 1e-8, 1.0), 1.0 / t)
            probs = probs / np.clip(probs.sum(), 1e-8, None)
        return probs

    def predict_next_numbers(self, recent_numbers, count=5):
        """Predict top-k numbers (k=count) using 36-class probability output."""
        probs = self.predict_next_probabilities(recent_numbers)
        if probs is None:
            return None

        ranked_idx = np.argsort(probs)[::-1]  # descending
        predicted_numbers = []
        for idx in ranked_idx:
            num = int(idx) + 1
            if 1 <= num <= 36 and num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) >= count:
                break
        return sorted(predicted_numbers[:count])

class RandomForestPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()

    def prepare_features(self, numbers, window_size=10):
        """Create features from number sequences"""
        features, targets = [], []

        for i in range(len(numbers) - window_size - 4):
            # Statistical features from the window
            window = numbers[i:i + window_size]
            feature = [
                np.mean(window), np.std(window), np.min(window), np.max(window), len(set(window)), # unique count
                numbers[i + window_size - 1] # last number
            ]                        

            # Add lag features
            for lag in [1, 2, 3]:
                if i >= lag:
                    feature.append(numbers[i - lag])
                else:
                    feature.append(0)

            features.append(feature)
            targets.append(numbers[i + window_size:i + window_size + 5])

        return np.array(features), np.array(targets)

    def train(self, numbers):
        """Train Random Forest model"""
        if len(numbers) < 20:
            return False

        X, y = self.prepare_features(numbers)

        if len(X) == 0:
            return False

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train separate models for each of th 5 numbers
        self.models = []
        for i in range(5):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y[:, i])
            self.models.append(model)

        return True

    def predict(self, recent_numbers):
        """Predict next 5 numbers using Random Forest"""
        if not hasattr(self, 'models'):
            return None

        # Create features from recent numbers
        window = recent_numbers[-10:] if len(recent_numbers) >= 10 else recent_numbers
        if len(window) < 10:
            window = [0] * (10 - len(window)) + window

        feature = [
            np.mean(window), np.std(window), np.min(window), np.max(window), len(set(window)), window[-1]
        ]                    

        # Add lag features
        for lag in [1, 2, 3]:
            if len(recent_numbers) > lag:
                feature.append(recent_numbers[-lag - 1])
            else:
                feature.append(0)

        X_scaled = self.scaler.transform([feature])

        # Predict each number
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled)[0]
            rounded = max(1, min(36, int(round(pred))))
            if rounded not in predictions:
                predictions.append(rounded)

        # Ensure 5 unique numbers
        while len(predictions) < 5:
            new_num = np.random.randint(1, 37)
            if new_num not in predictions:
                predictions.append(new_num)

        return sorted(predictions[:5])