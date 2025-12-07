import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter

# -----------------------------
# CNN Model
# -----------------------------
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, embeddings, freeze=True):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(embeddings)
        if freeze:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, embedding_dim)) for k in [3,4,5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100 * len([3,4,5]), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        convs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]
        out = torch.cat(pools, 1)
        out = self.dropout(out)
        return self.fc(out)

# -----------------------------
# Dataset
# -----------------------------
class HPDataset(Dataset):
    def __init__(self, books, word_to_idx):
        self.samples = []
        self.labels = []
        self.word_to_idx = word_to_idx

        for label, book in enumerate(books):
            with open(book, "r", encoding="utf-8") as f:
                text = f.read().strip().split(".")
                for sent in text:
                    tokens = [re.sub(r'[^\w\s]', '', w.lower()) for w in sent.split()]
                    indices = [self.word_to_idx.get(w, self.word_to_idx["<UNK>"]) for w in tokens]
                    if indices:
                        self.samples.append(torch.tensor(indices, dtype=torch.long))
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels

# -----------------------------
# Build vocabulary
# -----------------------------
def build_vocab(books, min_freq=1):
    counter = Counter()
    for book in books:
        with open(book, "r", encoding="utf-8") as f:
            text = f.read().lower()
            words = re.findall(r'\b\w+\b', text)
            counter.update(words)
    vocab = [word for word, freq in counter.items() if freq >= min_freq]
    vocab.append("<UNK>")
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx

# -----------------------------
# Load GloVe embeddings
# -----------------------------
def load_glove_embeddings(glove_file, word_to_idx, embedding_dim=50):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_idx), embedding_dim))
    print("Loading GloVe embeddings...")

    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.strip().split()
            if len(split_line) != embedding_dim + 1:  # word + vector
                continue  # skip malformed lines
            word = split_line[0]
            try:
                vector = np.array(split_line[1:], dtype=np.float32)
            except ValueError:
                continue  # skip lines that cannot be converted to float
            if word in word_to_idx:
                embeddings[word_to_idx[word]] = vector

    return torch.tensor(embeddings, dtype=torch.float32)

# -----------------------------
# Use trained Word2Vec embeddings
# -----------------------------
def load_word2vec_embeddings(word2vec_model, word_to_idx):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_idx), word2vec_model.fc1.out_features))
    for word, idx in word_to_idx.items():
        if word in word2vec_model.idx_to_word.values():
            one_hot_idx = list(word2vec_model.word_to_idx.keys()).index(word)
            embeddings[idx] = word2vec_model.fc1.weight.data[one_hot_idx].numpy()
    return torch.tensor(embeddings, dtype=torch.float32)

# -----------------------------
# Training
# -----------------------------
def train_cnn(books, vocab_size, embedding_dim, num_classes, embeddings, word_to_idx, epochs=10, save_path="cnn_hp.pth"):
    model = CNN(vocab_size, embedding_dim, num_classes, embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = HPDataset(books, word_to_idx)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'num_classes': num_classes,
        'embeddings': embeddings,
        'word_to_idx': word_to_idx
    }, save_path)
    print(f"Model saved to {save_path}")

# -----------------------------
# Test
# -----------------------------
def test(model, test_loader, device="cpu"):
    """
    Evaluate a trained CNN on a test dataset.
    
    Args:
        model: trained CNN model (torch.nn.Module)
        test_loader: DataLoader for test dataset
        device: "cpu" or "cuda"
    Returns:
        accuracy (float)
    """
    model.eval()  # evaluation mode
    correct, total = 0, 0

    with torch.no_grad():  # no gradients needed
        for inputs, labels in test_loader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predicted class (highest logit)
            _, predicted = torch.max(outputs, 1)

            # Update counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc
# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    books_list = ["harry_potter/HP1.txt","harry_potter/HP2.txt","harry_potter/HP3.txt","harry_potter/HP4.txt","harry_potter/HP5.txt","harry_potter/HP6.txt","harry_potter/HP7.txt"]
    word_to_idx = build_vocab(books_list)
    vocab_size = len(word_to_idx)
    embedding_dim = 50
    num_classes = 7

    # Option 1: Use GloVe
    glove_file = "Lab 2/glove.6B.50d.txt"
    embeddings = load_glove_embeddings(glove_file, word_to_idx, embedding_dim)

    embeddings = load_word2vec_embeddings(word2vec_model, word_to_idx)

    train_cnn(books_list, vocab_size, embedding_dim, num_classes, embeddings, word_to_idx, epochs=100)
