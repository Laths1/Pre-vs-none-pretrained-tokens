# Lathitha Nongauza - 2615978
import re
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy.linalg import norm
import statistics as stats
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
# the effect of using pretrained or none-pretrained tokens

# -----------------------------
# word2vec
# -----------------------------
class word2vec_network(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(word2vec_network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
            
    def forward(self, x):
        return self.fc2(self.fc1(x))

class word2vec:
    def __init__(self, books, words, k, lr, epoch):
        self.books = books
        self.words = words
        self.k = k
        self.lr = lr
        self.epoch = epoch
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []

    def get_embeddings(self, model, word_vector_pairs):   
        model.eval()
        embeddings = []
        words = [w for w, _ in word_vector_pairs]

        for _, one_hot in word_vector_pairs:
            one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embedding = model.fc1(one_hot_tensor).squeeze(0).numpy()
            embeddings.append(embedding)

        return np.array(embeddings), words  

    def data(self):
        all_sampled_words = []
        
        for book in self.books:
            try:
                with open(book, "r", encoding='utf-8') as f:
                    corpus = f.read().strip().split()
                    cleaned_words = [re.sub(r'[^\w\s]', '', word.lower()) for word in corpus]
                    
                    if len(cleaned_words) > self.words:
                        sampled_words = random.sample(cleaned_words, self.words)
                    else:
                        sampled_words = cleaned_words
                        
                    all_sampled_words.extend(sampled_words)
            except FileNotFoundError:
                print(f"Warning: File {book} not found. Skipping.")
                continue
        
        unique_words = list(set(all_sampled_words))
        unique_words.append("<UNK>")
        self.vocab = unique_words
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
        
        print(f"Vocabulary size: {len(unique_words)} words")
        
        vocab_size = len(unique_words)
        one_hot_vectors = np.eye(vocab_size)
        word_vector_pairs = [(word, one_hot_vectors[self.word_to_idx[word]]) for word in unique_words]
        
        data = []
        for book in self.books:
            try:
                with open(book, "r", encoding='utf-8') as f:
                    corpus = f.read().strip().split()
                    word_array = [re.sub(r'[^\w\s]', '', word.lower()) for word in corpus]
                    
                    for i in range(2, len(word_array)-2):
                        center_word = word_array[i]
                        center_idx = self.word_to_idx.get(center_word, self.word_to_idx["<UNK>"])
                        
                        context_words = [
                            word_array[i-2], word_array[i-1], 
                            word_array[i+1], word_array[i+2]
                        ]
                        
                        context_indices = []
                        for context_word in context_words:
                            context_idx = self.word_to_idx.get(context_word, self.word_to_idx["<UNK>"])
                            context_indices.append(context_idx)
                        
                        data.append((center_idx, context_indices))
            except FileNotFoundError:
                continue
        
        return data, word_vector_pairs
    
    def train(self):
        lr = self.lr
        numOfEpochs = self.epoch
        negative_samples = self.k

        dataset, word_vector_pairs = self.data()
        vocab_size = len(self.vocab)

        model = word2vec_network(input_size=vocab_size, hidden_size=50)
        criterion = nn.BCEWithLogitsLoss()  
        optimizer = optim.SGD(model.parameters(), lr=lr)

        one_hot_vectors = np.eye(vocab_size)
        word_indices = list(range(vocab_size))

        for epoch in range(numOfEpochs):
            total_loss = 0.0
            random.shuffle(dataset)

            for center_idx, context_indices in dataset:
                # Zero gradients at the start of each sample
                optimizer.zero_grad()
                
                center_vec = one_hot_vectors[center_idx]
                center_tensor = torch.tensor(center_vec, dtype=torch.float32).unsqueeze(0)

                logits = model(center_tensor).squeeze(0) 
                
                batch_loss = 0.0
                
                # Process all context words for this center word first
                for target_idx in context_indices:
                    # Positive example
                    pos_score = logits[target_idx].unsqueeze(0)
                    pos_label = torch.tensor([1.0])  
                    loss = criterion(pos_score, pos_label)

                    # Negative sampling
                    neg_indices = random.sample(
                        [i for i in word_indices if i != target_idx],
                        min(negative_samples, vocab_size - 1)
                    )
                    
                    if neg_indices:
                        neg_scores = logits[neg_indices]
                        neg_labels = torch.zeros(len(neg_indices))
                        loss += criterion(neg_scores, neg_labels)
                    
                    batch_loss += loss

                # Backward pass once for all context words of this center word
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()

            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{numOfEpochs}, Loss: {avg_loss:.4f}")

        torch.save({
            'model_state_dict': model.state_dict(),
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab': self.vocab
        }, 'hp_embeddings.pth')

        return model, word_vector_pairs 

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

        # Convolutional layers - adapt to the actual embedding dimension
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, embedding_dim)) for k in [3,4,5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100 * len([3,4,5]), num_classes)

    def forward(self, x):
        x = self.embedding(x)          # [batch, seq_len, emb_dim]
        x = x.unsqueeze(1)             # [batch, 1, seq_len, emb_dim]
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
def load_word2vec_embeddings(checkpoint_path, word_to_idx):
    """
    Load embeddings from a trained Word2Vec model checkpoint
    
    Args:
        checkpoint_path: Path to the saved Word2Vec model (.pth file)
        word_to_idx: Current vocabulary mapping for CNN
    """
    # Load the Word2Vec checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get Word2Vec vocabulary and embeddings
    word2vec_word_to_idx = checkpoint['word_to_idx']
    word2vec_embedding_dim = checkpoint['model_state_dict']['fc1.weight'].shape[0]  # Get actual embedding dim
    
    print(f"Word2Vec embedding dim: {word2vec_embedding_dim}")
    
    # Initialize embeddings with random values for current CNN vocabulary
    # Use the SAME dimension as Word2Vec embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_idx), word2vec_embedding_dim))
    
    found = 0
    
    # Extract the embedding matrix from Word2Vec model
    # fc1.weight has shape [hidden_size, vocab_size] - we need to transpose it
    word2vec_embedding_matrix = checkpoint['model_state_dict']['fc1.weight'].numpy().T
    
    # Map words from current vocabulary to Word2Vec embeddings
    for word, cnn_idx in word_to_idx.items():
        if word in word2vec_word_to_idx:
            # Get the index in the Word2Vec vocabulary
            word2vec_idx = word2vec_word_to_idx[word]
            # Copy the embedding (word2vec_embedding_matrix is [vocab_size, embedding_dim])
            embeddings[cnn_idx] = word2vec_embedding_matrix[word2vec_idx]
            found += 1
        elif word in ["<UNK>", "<PAD>"]:
            # Keep random initialization for special tokens
            continue
    
    print(f"Found {found}/{len(word_to_idx)} words in Word2Vec embeddings")
    return torch.tensor(embeddings, dtype=torch.float32), word2vec_embedding_dim

# -----------------------------
# Training
# -----------------------------
def train_cnn(books, vocab_size, embedding_dim, num_classes, embeddings, word_to_idx, epochs=10, save_path="cnn_hp.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(vocab_size, embedding_dim, num_classes, embeddings).to(device)
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
    ###########
    # word2vec
    ###########

    # books = word2vec([
    #     "HP1.txt",
    #     "HP2.txt",
    #     "HP3.txt", 
    #     "HP4.txt",
    #     "HP5.txt",
    #     "HP6.txt",
    #     "HP7.txt"
    # ], 20_000, 5, 0.1, 30)
    
    # Train the model
    # print("Training model...")
    # model, word_vector_pairs = books.train()

    ######
    # CNN
    ######  

    books_list = ["harry_potter/HP1.txt","harry_potter/HP2.txt","harry_potter/HP3.txt","harry_potter/HP4.txt","harry_potter/HP5.txt","harry_potter/HP6.txt","harry_potter/HP7.txt"]
    word_to_idx = build_vocab(books_list)
    vocab_size = len(word_to_idx)
    embedding_dim = 50
    num_classes = 7

    word2vec_model = "Lab 2/hp_embeddings.pth"
    glove_file = "Lab 2/glove.6B.50d.txt"
    # embeddings = load_glove_embeddings(glove_file, word_to_idx, embedding_dim)
    embeddings, embedding_dim = load_word2vec_embeddings(word2vec_model, word_to_idx)

    train_cnn(books_list, vocab_size, embedding_dim, num_classes, embeddings, word_to_idx, epochs=10)

    ########################
    # Embeddings evaluation
    ########################
    """
    # Get embeddings
    print("Loading model...")
    checkpoint = torch.load("Lab 2/hp_embeddings.pth")
    model = word2vec_network(input_size=len(checkpoint['vocab']), hidden_size=50)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Restore the vocabulary mappings
    books.word_to_idx = checkpoint['word_to_idx']
    books.idx_to_word = checkpoint['idx_to_word'] 
    books.vocab = checkpoint['vocab']
    # Recreate word_vector_pairs for embedding extraction
    one_hot_vectors = np.eye(len(books.vocab))
    word_vector_pairs = [(word, one_hot_vectors[books.word_to_idx[word]]) for word in books.vocab]

    print("Extracting embeddings...")
    embeddings, words = books.get_embeddings(model, word_vector_pairs)
    
    # K-means clustering
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Dimensionality reduction
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot
    limit = 200
    limited_embeddings = reduced_embeddings[:limit]
    limited_labels = labels[:limit]
    limited_words = words[:limit]  

    plt.figure(figsize=(18, 12))
    scatter = plt.scatter(
        limited_embeddings[:, 0],
        limited_embeddings[:, 1],
        c=limited_labels,
        cmap="tab10",
        alpha=0.7
    )

    for i, word in enumerate(limited_words):
        plt.annotate(word, (limited_embeddings[i, 0], limited_embeddings[i, 1]), fontsize=8, alpha=0.7)

    plt.title("Word Embedding Clusters (PCA) - Limited to 200 Words")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label="Cluster")
    plt.savefig("word_embedding_clusters.png")
    plt.show()

    #############
    # Evaluation 
    #############

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-8) 

    related_words_pairs = [
        ("she", "her"), ("he", "him"), ("small", "thin"), ("mrs", "mr"),
        ("boring", "dull"), ("outside", "garden"), ("saw", "stared"),
        ("blonde", "light"), ("steering", "car"), ("opinion", "thought"),
        ("wand", "spell"), ("broomstick", "nimbus"), ("potions", "snape"), 
        ("gryffindor", "courage"), ("muggle", "nonmagic"), ("quidditch", "snitch"), 
        ("howler", "angry"), ("dementor", "despair"), ("sorting", "hat"),
        ("troll", "club"), ("mirror", "erised"), ("phoenix", "fawkes"), 
        ("potion", "polyjuice"), ("herbology", "sprout"), ("scar", "lightning"),
        ("ghost", "nearlyheadless"), ("portkey", "travel"), ("prophecy", "orb"), 
        ("werewolf", "lupin"), ("stone", "sorcerer") 
    ]
    
    unrelated_words_pairs = [
        ("she", "dursleys"), ("he", "mysterious"), ("small", "usual"),  
        ("mrs", "happily"), ("boring", "owl"), ("outside", "signs"),
        ("saw", "window"), ("blonde", "woke"), ("steering", "fashion"),
        ("opinion", "work"), ("wand", "dursleys"), ("broomstick", "pudding"),  
        ("potions", "muggle"), ("gryffindor", "socks"), ("muggle", "broomstick"),  
        ("quidditch", "homework"), ("howler", "hagrid"), ("dementor", "butterbeer"),  
        ("sorting", "quill"), ("troll", "library"), ("mirror", "kreacher"),  
        ("phoenix", "divination"), ("potion", "muggle"), ("herbology", "boring"), 
        ("scar", "feast"), ("ghost", "transfiguration"), ("portkey", "sock"),  
        ("prophecy", "owl"), ("werewolf", "gillyweed"), ("stone", "quidditch")  
    ]

    # Create mapping from word -> embedding
    word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}

    related_scores = []
    unrelated_scores = []
    
    # Test related pairs
    for w1, w2 in related_words_pairs:
        w1_lower, w2_lower = w1.lower(), w2.lower()
        if w1_lower in word_to_embedding and w2_lower in word_to_embedding:
            sim = cosine_similarity(word_to_embedding[w1_lower], word_to_embedding[w2_lower])
            related_scores.append(sim)

    # Test unrelated pairs  
    for w1, w2 in unrelated_words_pairs:
        w1_lower, w2_lower = w1.lower(), w2.lower()
        if w1_lower in word_to_embedding and w2_lower in word_to_embedding:
            sim = cosine_similarity(word_to_embedding[w1_lower], word_to_embedding[w2_lower])
            unrelated_scores.append(sim)
        
    # Calculate statistics
    if related_scores and unrelated_scores:
        related_mean = stats.mean(related_scores)
        related_var = stats.variance(related_scores) if len(related_scores) > 1 else 0
        unrelated_mean = stats.mean(unrelated_scores)
        unrelated_var = stats.variance(unrelated_scores) if len(unrelated_scores) > 1 else 0
        
        print("\n=== Results ===")
        print(f"Related pairs mean: {related_mean:.6f}")
        print(f"Related pairs variance: {related_var:.6f}")
        print(f"Unrelated pairs mean: {unrelated_mean:.6f}") 
        print(f"Unrelated pairs variance: {unrelated_var:.6f}")
        
        if related_mean > unrelated_mean:
            print("✓ SUCCESS: Related pairs have higher similarity")
        else:
            print("✗ FAILURE: Unrelated pairs have higher similarity")
    else:
        print("Not enough data to calculate statistics")
    """