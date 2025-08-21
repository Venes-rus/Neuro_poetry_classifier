import pandas as pd
import re
import numpy as np
from collections import Counter
import pymorphy2
from syllables import estimate as syllable_count
from sklearn.preprocessing import LabelEncoder
import pickle

morph = pymorphy2.MorphAnalyzer()

def detect_meter(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    meter_scores = {'ямб': 0, 'хорей': 0, 'дактиль': 0, 'анапест': 0}
    
    for line in lines[:15]:
        words = line.split()
        if len(words) < 3:
            continue
        
        stresses = []
        for word in words[-3:]:
            syllables = syllable_count(word)
            stresses.append(min(2, syllables - 1))
        
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
    
    features = {
        # Метрические (4)
        'meter_ямб': 1.0 if meter == 'ямб' else 0.0,
        'meter_хорей': 1.0 if meter == 'хорей' else 0.0,
        'meter_дактиль': 1.0 if meter == 'дактиль' else 0.0,
        'meter_анапест': 1.0 if meter == 'анапест' else 0.0,
        
        # Морфологические (8)
        'noun_ratio': pos_counts.get('NOUN', 0) / total_pos,
        'verb_ratio': pos_counts.get('VERB', 0) / total_pos,
        'adj_ratio': pos_counts.get('ADJF', 0) / total_pos,
        'adj_short_ratio': pos_counts.get('ADJS', 0) / total_pos,
        'comp_ratio': pos_counts.get('COMP', 0) / total_pos,
        'participle_ratio': pos_counts.get('PRTF', 0) / total_pos,
        'participle_short_ratio': pos_counts.get('PRTS', 0) / total_pos,
        'gerund_ratio': pos_counts.get('GRND', 0) / total_pos,
        
        # Статистические (6)
        'word_count': len(words),
        'lex_diversity': len(set(words)) / len(words) if words else 0.0,
        'avg_word_len': np.mean([len(w) for w in words]) if words else 0.0,
        'lines_count': len(lines),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0.0,
        'punctuation_ratio': sum(1 for c in text if c in '!?…') / len(text) if text else 0.0,
        
        # Символьные (10)
        'char_ё': char_counts.get('ё', 0) / total_chars,
        'char_ж': char_counts.get('ж', 0) / total_chars,
        'char_ф': char_counts.get('ф', 0) / total_chars,
        'char_щ': char_counts.get('щ', 0) / total_chars,
        'char_э': char_counts.get('э', 0) / total_chars,
        'char_ъ': char_counts.get('ъ', 0) / total_chars,
        'char_ы': char_counts.get('ы', 0) / total_chars,
        'char_ь': char_counts.get('ь', 0) / total_chars,
        'char_ю': char_counts.get('ю', 0) / total_chars,
        'char_я': char_counts.get('я', 0) / total_chars,
    }
    return features

def preprocess_data():
    df = pd.read_csv("poetry_dataset.csv")
    
    features = []
    for text in df['text']:
        features.append(extract_features(text))
    
    features_df = pd.DataFrame(features)
    final_df = pd.concat([df, features_df], axis=1)
    
    # Удаление строк с NaN (если признаки не извлеклись)
    final_df = final_df.dropna()
    
    final_df.to_csv("preprocessed_poetry.csv", index=False)
    
    le = LabelEncoder()
    le.fit(df['author'])
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("Данные успешно предобработаны. Сохранено признаков:", len(features_df.columns))

if __name__ == "__main__":
    preprocess_data()