# ==========================================
# Bio-Sparse Vision AI Engine (V1.0)
# Developer: Mustafa Al-Tamimi
# ==========================================

import numpy as np
import os
import time
from datetime import datetime

class BioSparseAI:
    def __init__(self, memory_file="brain_memory.npy", log_file="activity_log.txt"):
        self.memory_file = memory_file
        self.log_file = log_file
        self.actions = ["STOP 🛑", "GO 🟢", "RIGHT ➡️", "LEFT ⬅️", "EMERGENCY ⚠️"]
        self.weights = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            return np.load(self.memory_file)
        return np.random.rand(20, len(self.actions))

    def predict(self, input_data):
        # Neural Sparsity Logic
        scores = np.dot(input_data, self.weights)
        limit = np.percentile(np.abs(scores), 90)
        output = np.where(np.abs(scores) >= limit, scores, 0)
        return np.argmax(output)

    def train(self, inputs, target_idx):
        target = np.zeros(len(self.actions))
        target[target_idx] = 1
        for i in range(len(self.actions)):
            pred = np.dot(inputs, self.weights[:, i])
            self.weights[:, i] += 0.1 * (target[i] - pred) * inputs

    def save(self):
        np.save(self.memory_file, self.weights)

if __name__ == "__main__":
    ai = BioSparseAI()
    print("🚀 Bio-Sparse AI is running...")
    # Simulated sensor loop
    try:
        while True:
            sensors = np.random.rand(20)
            idx = ai.predict(sensors)
            print(f"Decision: {ai.actions[idx]}")
            time.sleep(1)
    except KeyboardInterrupt:
        ai.save()
