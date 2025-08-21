import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import pymorphy2
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from syllables import estimate as syllable_count
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()

# Конфигурация
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
EMBEDDING_DIM = 256
LSTM_HIDDEN = 256
DENSE_HIDDEN = 128
DROPOUT_RATE = 0.5

class PoetryDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Токенизация текста
        tokens = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in text.split()[:self.max_len]]
        tokens = tokens + [self.word2idx["<pad>"]] * (self.max_len - len(tokens))
        
        # Извлечение стилистических признаков
        style_features = self.extract_style_features(text)
        
        return {
            'tokens': torch.LongTensor(tokens),
            'features': torch.FloatTensor(style_features),
            'label': torch.LongTensor([label])
        }
    
    def extract_style_features(self, text):
        # Анализ метра
        meter = self.detect_meter(text)
        
        # Части речи
        words = [w for w in re.findall(r'\b[а-яё]+\b', text.lower()) if len(w) > 1]
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        pos_tags = []
        for w in words[:300]:
            try:
                pos_tags.append(morph.parse(w)[0].tag.POS)
            except:
                continue
        pos_counts = Counter(pos_tags)
        total_pos = max(1, len(pos_tags))
        
        # Статистика по символам
        char_counts = Counter(text.lower())
        total_chars = max(1, sum(char_counts.values()))
        
        # Собираем все признаки
        features = [
            # Метрические (4)
            1.0 if meter == 'ямб' else 0.0,
            1.0 if meter == 'хорей' else 0.0,
            1.0 if meter == 'дактиль' else 0.0,
            1.0 if meter == 'анапест' else 0.0,
            
            # Морфологические (8)
            pos_counts.get('NOUN', 0) / total_pos,
            pos_counts.get('VERB', 0) / total_pos,
            pos_counts.get('ADJF', 0) / total_pos,
            pos_counts.get('ADJS', 0) / total_pos,
            pos_counts.get('COMP', 0) / total_pos,
            pos_counts.get('PRTF', 0) / total_pos,
            pos_counts.get('PRTS', 0) / total_pos,
            pos_counts.get('GRND', 0) / total_pos,
            
            # Статистические (6)
            len(words) / 200.0,
            len(set(words)) / len(words) if words else 0.0,
            np.mean([len(w) for w in words]) / 10.0 if words else 0.0,
            len(lines) / 20.0,
            sum(1 for c in text if c.isupper()) / len(text) if text else 0.0,
            sum(1 for c in text if c in '!?…') / len(text) if text else 0.0,
            
            # Символьные (10)
            char_counts.get('ё', 0) / total_chars,
            char_counts.get('ж', 0) / total_chars,
            char_counts.get('ф', 0) / total_chars,
            char_counts.get('щ', 0) / total_chars,
            char_counts.get('э', 0) / total_chars,
            char_counts.get('ъ', 0) / total_chars,
            char_counts.get('ы', 0) / total_chars,
            char_counts.get('ь', 0) / total_chars,
            char_counts.get('ю', 0) / total_chars,
            char_counts.get('я', 0) / total_chars,
        ]
        
        return features
    
    def detect_meter(self, text):
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 0]
        meter_scores = {'ямб':0, 'хорей':0, 'дактиль':0, 'анапест':0}
        
        for line in lines[:15]:
            words = line.split()
            if len(words) < 3: continue
            
            stresses = []
            for word in words[-3:]:
                syllables = syllable_count(word)
                stresses.append(min(2, syllables-1))
            
            if len(stresses) >= 2:
                if stresses[-2] < stresses[-1]:
                    meter_scores['ямб'] += 1
                elif stresses[-2] > stresses[-1]:
                    meter_scores['хорей'] += 1
                elif stresses[-2] == 0 and stresses[-1] == 0:
                    meter_scores['дактиль'] += 1
                elif stresses[-2] == 1 and stresses[-1] == 1:
                    meter_scores['анапест'] += 1
        
        return max(meter_scores.items(), key=lambda x: x[1])[0]

class AuthorClassifier(nn.Module):
    def __init__(self, vocab_size, num_features, num_classes):
        super().__init__()
        
        #эмбеддинг с dropout
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.embed_dropout = nn.Dropout(DROPOUT_RATE)
        
        #LSTM с двумя слоями
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, 
            LSTM_HIDDEN, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=DROPOUT_RATE if 2 > 1 else 0
        )
        
        # Механизм внимания
        self.attention = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, DENSE_HIDDEN),
            nn.Tanh(),
            nn.Linear(DENSE_HIDDEN, 1),
            nn.Dropout(DROPOUT_RATE)
        )
        
        #Обработка стилистических признаков
        self.style_net = nn.Sequential(
            nn.Linear(num_features, DENSE_HIDDEN * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(DENSE_HIDDEN * 2, DENSE_HIDDEN),
            nn.ReLU()
        )
        
        #Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2 + DENSE_HIDDEN, DENSE_HIDDEN * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(DENSE_HIDDEN * 2, num_classes)
        )
        
        # Инициализация весов
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    for i in range(0, param.size(0), param.size(1)):
                        nn.init.orthogonal_(param[i:i+param.size(1)])
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
    
    def forward(self, tokens, features):
        # Эмбеддинг слов
        x = self.embedding(tokens)
        x = self.embed_dropout(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Обработка стилистических признаков
        style_out = self.style_net(features)
        
        # Объединение и классификация
        combined = torch.cat([context_vector, style_out], dim=1)
        logits = self.classifier(combined)
        
        return logits

def load_data():
    df = pd.read_csv("preprocessed_poetry.csv")
    
    # Созд словарь
    word_counts = Counter()
    for text in df['text']:
        words = text.split()
        word_counts.update(words)
    
    # Отбираем наиболее частые слова
    vocab = ["<pad>", "<unk>"] + [word for word, count in word_counts.most_common(15000)]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Кодируем метки
    le = LabelEncoder()
    labels = le.fit_transform(df['author'])
    
    return df['text'].values, labels, word2idx, le

def generate_reports(le, y_true, y_pred, history):
    """Генерация всех отчетов и графиков"""
    # Отчет о классификации
    report = classification_report(
        y_true, y_pred, 
        target_names=le.classes_,
        output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv("classification_report.csv")
    
    # Матрица ошибок
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # График обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train')
    plt.plot(history['epoch'], history['val_loss'], label='Validation')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train')
    plt.plot(history['epoch'], history['val_acc'], label='Validation')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()

def train():
    # Загрузка данных
    texts, labels, word2idx, le = load_data()
    
    # Разделение на train/val (80/20)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Создание датасетов
    train_dataset = PoetryDataset(train_texts, train_labels, word2idx, MAX_LEN)
    val_dataset = PoetryDataset(val_texts, val_labels, word2idx, MAX_LEN)
    
    # DataLoader с увеличенным num_workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    #ализация модели
    num_features = len(train_dataset[0]['features'])  # Теперь 28 признаков
    model = AuthorClassifier(len(word2idx), num_features, len(le.classes_))
    
    #    оптимизатор с weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-5
    )
    
    # Планировщик скорости обучения
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Функция потерь с учетом дисбаланса классов
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    class_weights = torch.FloatTensor(class_weights / class_weights.sum())
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Обучение
    best_val_acc = 0.0
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    all_val_preds = []
    all_val_labels = val_labels
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            tokens = batch['tokens'].to('cpu')
            features = batch['features'].to('cpu')
            labels = batch['label'].squeeze().to('cpu')
            
            optimizer.zero_grad()
            logits = model(tokens, features)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
        
        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        val_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to('cpu')
                features = batch['features'].to('cpu')
                labels = batch['label'].squeeze().to('cpu')
                
                logits = model(tokens, features)
                val_loss += criterion(logits, labels).item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                val_preds.extend(preds.cpu().numpy())
        
        # Сохраняем метрики
        train_acc = correct / len(train_dataset)
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        
        history['epoch'].append(epoch+1)
        history['train_loss'].append(total_loss/len(train_loader))
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "poetry_model.pth")
            np.save("val_predictions.npy", val_preds)
            np.save("val_true.npy", all_val_labels)
            pd.DataFrame(history).to_csv("training_history.csv", index=False)
    
    # Генерация отчетов
    generate_reports(le, all_val_labels, val_preds, history)
    
    # Сохранение дополнительных данных
    with open('word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\nЛучшая Val Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()