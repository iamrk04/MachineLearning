# Build a simple sentiment analysis model

import math
from collections import defaultdict, Counter


class SentimentModel:
    def __init__(self):
        self.class_counts = Counter()
        self.word_freq_per_class = defaultdict(Counter)
        self.num_words_per_class = Counter()
        self.vocab = set()
        self.num_samples = 0

    # TC: O(n_samples * n_words_per_sample)
    # SC: O(n_classes * n_words_in_vocab)
    def fit(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            self.num_samples += 1

            words = text.lower().split()
            for word in words:
                self.word_freq_per_class[label][word] += 1
                self.num_words_per_class[label] += 1
                self.vocab.add(word)

    # TC: O(n_classes * n_words_in_input)
    # SC: O(n_words_in_input)
    def predict(self, text):
        words = text.lower().split()
        vocab_size = len(self.vocab)

        best_label = None
        best_score = float('-inf')

        for label in self.class_counts:
            # log prior
            log_prob = math.log(self.class_counts[label] / self.num_samples)

            for word in words:
                # Laplace smoothing
                word_freq = self.word_freq_per_class[label][word]
                prob = (word_freq + 1) / (self.num_words_per_class[label] + vocab_size)
                log_prob += math.log(prob)

            if log_prob > best_score:
                best_score = log_prob
                best_label = label

        return best_label


model = SentimentModel()
training_data = [
    ("good movie", "positive"),
    ("loved the film", "positive"),
    ("superb acting with great story", "positive"),
    ("waste of time", "negative"),
    ("one time watch", "negative"),
    ("bad movie", "negative")
]

reviews = []
labels = []
for review_text, sentiment_label in training_data:
    reviews.append(review_text)
    labels.append(sentiment_label)

model.fit(reviews, labels)

print(model.predict("impressive acting"))  # positive
print(model.predict("poor direction"))  # negative
print(model.predict("not watchable"))  # negative
print(model.predict("poor acting"))  # negative
