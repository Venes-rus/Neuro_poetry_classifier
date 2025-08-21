import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pymorphy2
from syllables import estimate as syllable_count
import re
from collections import Counter
import pickle

class PoetryClassifier(nn.Module):
    def __init__(self, vocab_size, num_features, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.embed_dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Dropout(0.5)
        )
        self.style_net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, tokens, features):
        x = self.embedding(tokens)
        x = self.embed_dropout(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        style_out = self.style_net(features)
        combined = torch.cat([context, style_out], dim=1)
        return self.classifier(combined)
    
    def get_attention(self, tokens):
        x = self.embedding(tokens)
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        return attention_weights.squeeze().numpy()

def load_model():
    with open('word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    #создаем модель
    model = PoetryClassifier(
        vocab_size=len(word2idx),
        num_features=28,  
        num_classes=len(label_encoder.classes_)
    )
    
    # Загружаем веса модели
    model.load_state_dict(torch.load('poetry_model.pth', map_location='cpu'))
    model.eval()
    
    return model, word2idx, label_encoder

def detect_meter(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    meter_scores = {'ямб': 0, 'хорей': 0, 'дактиль': 0, 'анапест': 0}
    
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

def extract_features(text):
    morph = pymorphy2.MorphAnalyzer()
    words = [w for w in re.findall(r'\b[а-яё]+\b', text.lower()) if len(w) > 1]
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    meter = detect_meter(text)
    
    pos_tags = []
    for w in words[:300]:
        try:
            pos_tags.append(morph.parse(w)[0].tag.POS)
        except:
            continue
    pos_counts = Counter(pos_tags)
    total_pos = max(1, len(pos_tags))
    
    char_counts = Counter(text.lower())
    total_chars = max(1, sum(char_counts.values()))
    
    features = [
        # Метрика
        1.0 if meter == 'ямб' else 0.0,
        1.0 if meter == 'хорей' else 0.0,
        1.0 if meter == 'дактиль' else 0.0,
        1.0 if meter == 'анапест' else 0.0,
        
        # Части речи
        pos_counts.get('NOUN', 0) / total_pos,
        pos_counts.get('VERB', 0) / total_pos,
        pos_counts.get('ADJF', 0) / total_pos,
        pos_counts.get('ADJS', 0) / total_pos,
        pos_counts.get('COMP', 0) / total_pos,
        pos_counts.get('PRTF', 0) / total_pos,
        pos_counts.get('PRTS', 0) / total_pos,
        pos_counts.get('GRND', 0) / total_pos,
        
        # Статистика
        len(words) / 200.0,
        len(set(words)) / len(words) if words else 0.0,
        np.mean([len(w) for w in words]) / 10.0 if words else 0.0,
        len(lines) / 20.0,
        sum(1 for c in text if c.isupper()) / len(text) if text else 0.0,
        sum(1 for c in text if c in '!?…') / len(text) if text else 0.0,
        
        # Символы
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

def interpret_features(features, author):
    feature_names = [
        'ямб', 'хорей', 'дактиль', 'анапест',
        'доля_сущ', 'доля_глаг', 'доля_прил', 'доля_прил_кр',
        'доля_сравн', 'доля_прич', 'доля_прич_кр', 'доля_деепр',
        'длина_текста', 'лекс_разнообразие', 'ср_длина_слова',
        'строки', 'доля_заглавных', 'доля_знаков',
        'частота_ё', 'частота_ж', 'частота_ф', 'частота_щ',
        'частота_э', 'частота_ъ', 'частота_ы', 'частота_ь',
        'частота_ю', 'частота_я'
    ]
    
    interpretations = []
    feature_values = dict(zip(feature_names, features))
    
    # Метрические признаки
    if feature_values['ямб'] > 0.8 and 'пушкин' in author.lower():
        interpretations.append(f"Ямб ({feature_values['ямб']:.0%} строк) - характерно для Пушкина")
    
    if feature_values['хорей'] > 0.7 and 'есенин' in author.lower():
        interpretations.append(f"Хорей ({feature_values['хорей']:.0%} строк) - характерно для Есенина")
    
    # Морфологические признаки
    if feature_values['доля_сущ'] > 0.35 and 'ахматова' in author.lower():
        interpretations.append(f"Высокая доля существительных ({feature_values['доля_сущ']:.0%}) - свойственно Ахматовой")
    
    if feature_values['доля_глаг'] > 0.3 and 'цветаева' in author.lower():
        interpretations.append(f"Высокая доля глаголов ({feature_values['доля_глаг']:.0%}) - свойственно Цветаевой")
    
    # Символьные признаки
    if feature_values['частота_ё'] > 0.004 and 'цветаева' in author.lower():
        interpretations.append(f"Частое использование 'ё' ({feature_values['частота_ё']:.3%}) - маркер Цветаевой")
    
    if feature_values['частота_ф'] > 0.002 and 'блок' in author.lower():
        interpretations.append(f"Использование редкой буквы 'ф' ({feature_values['частота_ф']:.3%}) - характерно для Блока")
    
    # Статистические признаки
    if feature_values['лекс_разнообразие'] > 0.6 and 'пушкин' in author.lower():
        interpretations.append(f"Высокое лексическое разнообразие ({feature_values['лекс_разнообразие']:.0%}) - свойственно Пушкину")
    
    if feature_values['ср_длина_слова'] > 0.6 and 'блок' in author.lower():
        interpretations.append(f"Длинные слова (ср. {feature_values['ср_длина_слова']*10:.1f} букв) - характерно для Блока")
    
    return {
        'feature_names': feature_names,
        'feature_values': feature_values,
        'interpretations': interpretations
    }

def predict_author(text, model, word2idx, label_encoder):
    if len(text) < 50:
        return {"error": "Текст слишком короткий (минимум 50 символов)."}
    
    # Токенизация
    tokens = [word2idx.get(word, word2idx['<unk>']) for word in text.split()[:256]]
    tokens += [word2idx['<pad>']] * (256 - len(tokens))
    tokens = torch.LongTensor([tokens])
    
    # Признаки
    features = torch.FloatTensor([extract_features(text)])
    
    # Предсказание
    with torch.no_grad():
        logits = model(tokens, features)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    
    # Результат
    author_idx = np.argmax(probs)
    author = label_encoder.classes_[author_idx]
    confidence = " (высокая)" if probs[author_idx] > 0.7 else " (средняя)" if probs[author_idx] > 0.5 else " (низкая)"
    
    # Интерпретация признаков
    interpretation = interpret_features(features.numpy()[0], author)
    
    return {
        'author': author + confidence,
        'probs': list(zip(label_encoder.classes_, probs)),
        'features': interpretation
    }