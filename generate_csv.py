import os
import pandas as pd

def generate_dataset(data_dir="data"):
    authors = ["pushkin", "yesenin", "akhmatova", "blok", "tsvetaeva"]
    data = []
    
    for author in authors:
        author_dir = os.path.join(data_dir, author)
        if not os.path.exists(author_dir):
            continue
            
        for filename in os.listdir(author_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(author_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    data.append({
                        "author": author,
                        "text": text,
                        "filename": filename
                    })
    
    df = pd.DataFrame(data)
    df.to_csv("poetry_dataset.csv", index=False)
    print(f"Создан датасет с {len(df)} стихотворениями")

if __name__ == "__main__":
    generate_dataset()