from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
import numpy as np
import os
import pickle

class SmartChatbot:
    def __init__(self, dataset_path='dataset', models_path='models'):
        self.dataset_path = dataset_path
        self.models_path = models_path
        self.model = None
        self.vectorizer = None
        self.responses = {}
        self.has_responses = False
        self.initialize_model()

    def initialize_model(self):
        """Load model if exists, else train a new one."""
        model_file = os.path.join(self.models_path, 'model.pkl')
        vectorizer_file = os.path.join(self.models_path, 'vectorizer.pkl')
        if os.path.exists(model_file) and os.path.exists(vectorizer_file):
            self.load_model()
        else:
            self.train_model(force_train=True)

    def load_model(self):
        """Load trained model and vectorizer from disk."""
        try:
            with open(os.path.join(self.models_path, 'model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_path, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            _, self.responses, _ = self.load_dataset()
            self.has_responses = bool(self.responses)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model(force_train=True)

    def load_dataset(self):
        """Load training data, responses, and unlabeled data from dataset folder."""
        training_data = []
        responses = {}
        # Load training data
        training_path = os.path.join(self.dataset_path, 'training')
        for filename in os.listdir(training_path):
            if filename.endswith('.txt'):
                category = filename.split('.')[0]
                with open(os.path.join(training_path, filename), 'r', encoding='utf-8') as f:
                    phrases = f.read().splitlines()
                    training_data.extend([(p.strip(), category) for p in phrases if p.strip()])
        # Load responses
        responses_path = os.path.join(self.dataset_path, 'responses')
        for filename in os.listdir(responses_path):
            if filename.endswith('.txt'):
                category = filename.split('.')[0]
                with open(os.path.join(responses_path, filename), 'r', encoding='utf-8') as f:
                    responses[category] = [l.strip() for l in f.readlines() if l.strip()]
        # Load unlabeled data
        unlabeled_path = os.path.join(self.dataset_path, 'unlabeled.txt')
        unlabeled_data = []
        if os.path.exists(unlabeled_path):
            with open(unlabeled_path, 'r', encoding='utf-8') as f:
                unlabeled_data = [line.strip() for line in f if line.strip()]
        return training_data, responses, unlabeled_data

    def train_model(self, force_train=False):
        """Train and save the model using current dataset and unlabeled data if needed."""
        training_data, self.responses, unlabeled_data = self.load_dataset()
        self.has_responses = bool(self.responses)
        X_labeled = [x[0] for x in training_data]
        y_labeled = [x[1] for x in training_data]

        # If responses exist, train on labeled data only
        if self.has_responses:
            X_all = X_labeled
            y_all = np.array(y_labeled, dtype=object)
        else:
            # Use semi-supervised learning with unlabeled data
            X_all = X_labeled + unlabeled_data
            y_all = np.array(y_labeled + [-1] * len(unlabeled_data), dtype=object)

        self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2))
        X_vec = self.vectorizer.fit_transform(X_all)
        base_model = LogisticRegression(max_iter=1000)
        if self.has_responses:
            self.model = base_model.fit(X_vec, y_all)
        else:
            self.model = SelfTrainingClassifier(base_model, criterion='threshold', threshold=0.8)
            self.model.fit(X_vec, y_all)
        # Save model and vectorizer
        os.makedirs(self.models_path, exist_ok=True)
        with open(os.path.join(self.models_path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.models_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Model trained and saved successfully.")

    def get_response(self, user_input):
        """Generate a response for the given user input."""
        if not self.model or not self.vectorizer:
            return "Sorry, I'm not ready yet.", 0.0, "unknown"
        X_test = self.vectorizer.transform([user_input])
        prediction = self.model.predict(X_test)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_test)
            confidence = float(np.max(probabilities))
        else:
            confidence = 1.0
        category = prediction[0]
        if self.has_responses and confidence > 0.5 and category in self.responses:
            import random
            response = random.choice(self.responses[category])
        else:
            # Fallback for semi-supervised or unknown
            response = "I'm not sure how to respond to that yet."
            category = "unknown"
        return response, confidence, category

# --- Flask API ---
app = Flask(__name__)
CORS(app)
chatbot = SmartChatbot()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response, confidence, category = chatbot.get_response(user_input)
    return jsonify({
        "response": response,
        "confidence": confidence,
        "category": category
    })

def main():
    chatbot = SmartChatbot()
    print("\nChatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response, confidence, category = chatbot.get_response(user_input)
        print(f"Bot: {response} (confidence: {confidence:.2f}, category: {category})")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
    # main()  # Uncomment this line to run the CLI version instead