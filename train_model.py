
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent

training_data = {
    "ask_question": [
        "What is photosynthesis?",
        "Explain gravity",
        "What is algebra?",
        "Tell me about plants",
        "How does gravity work?",
        "Explain equations",
        "Define chlorophyll",
        "Teach me physics",
        "I have a science question",
        "Explain this concept",
        "What does this mean?",
        "How does this work?"
    ],
    "get_notes": [
        "Give me photosynthesis notes",
        "I need gravity notes",
        "Show algebra notes",
        "Give me revision material",
        "Summarize this topic",
        "Help me revise biology",
        "Give me study resources",
        "Show me important points",
        "I want study notes",
        "Provide a short summary",
        "Help with revision",
        "Give me learning materials"
    ],
    "start_quiz": [
        "Start a quiz",
        "Test me on gravity",
        "Give me an algebra quiz",
        "I want a practice test",
        "Quiz me on photosynthesis",
        "Start an assessment",
        "Give me multiple choice questions",
        "Test my knowledge",
        "I want to practise",
        "Start exam practice",
        "Check my learning",
        "Give me quiz questions"
    ],
    "greeting": [
        "Hi",
        "Hello",
        "Hey",
        "Good morning",
        "Hello EduBot",
        "Hi there",
        "Good afternoon",
        "Hey EduBot"
    ],
    "help": [
        "What can you do?",
        "Help me",
        "Show your features",
        "How does this bot work?",
        "What subjects do you support?",
        "How can you help me?",
        "Show available options",
        "How do I use this bot?"
    ]
}

# Save the training examples for inspection.
with open(
    BASE_DIR / "intent_training.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(training_data, file, indent=2)

texts = []
labels = []

for intent, examples in training_data.items():
    for example in examples:
        texts.append(example)
        labels.append(intent)

# Text processing and machine learning classifier.
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(texts, labels)

joblib.dump(model, BASE_DIR / "intent_model.joblib")

print("Training examples:", len(texts))
print("Intent categories:", list(training_data.keys()))
print("Model saved successfully!")
