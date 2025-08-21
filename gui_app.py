import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from predict import predict_author, load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class PoetryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Определение автора стихотворения")
        self.root.geometry("1100x800")
        
        # Загрузка модели
        self.model, self.word2idx, self.label_encoder = load_model()
        
        # Настройка стилей
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', font=('Helvetica', 10), padding=5)
        style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        
        # Основной контейнер
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Текстовое поле для ввода
        self.text_frame = ttk.Frame(self.main_frame)
        self.text_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.text_label = ttk.Label(self.text_frame, text="Введите текст стихотворения:")
        self.text_label.pack(anchor=tk.W)
        
        self.text_area = tk.Text(self.text_frame, height=15, font=('Helvetica', 10))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Кнопки
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        self.analyze_btn = ttk.Button(self.button_frame, text="Анализировать", command=self.analyze_text)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_btn = ttk.Button(self.button_frame, text="Загрузить из файла", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(self.button_frame, text="Очистить", command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Панель результатов
        self.result_paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.result_paned.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель - результаты и график
        self.left_panel = ttk.Frame(self.result_paned)
        self.result_paned.add(self.left_panel, weight=2)
        
        self.result_frame = ttk.LabelFrame(self.left_panel, text="Результаты анализа")
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(self.result_frame, height=8, font=('Helvetica', 10))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.figure = plt.figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Правая панель - ключевые признаки
        self.right_panel = ttk.Frame(self.result_paned)
        self.result_paned.add(self.right_panel, weight=1)
        
        self.feature_frame = ttk.LabelFrame(self.right_panel, text="Ключевые признаки")
        self.feature_frame.pack(fill=tk.BOTH, expand=True)
        
        self.feature_text = tk.Text(self.feature_frame, height=20, font=('Helvetica', 9))
        self.feature_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Полоса прокрутки для признаков
        scrollbar = ttk.Scrollbar(self.feature_frame, orient="vertical", command=self.feature_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.feature_text.configure(yscrollcommand=scrollbar.set)
    
    def analyze_text(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if len(text) < 50:
            messagebox.showwarning("Ошибка", "Текст слишком короткий (минимум 50 символов)")
            return
        
        try:
            result = predict_author(text, self.model, self.word2idx, self.label_encoder)
            self.show_results(result)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка анализа: {str(e)}")
    
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert("1.0", f.read())
    
    def clear_text(self):
        self.text_area.delete("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)
        self.feature_text.delete("1.0", tk.END)
        self.figure.clear()
        self.canvas.draw()
    
    def show_results(self, result):
        self.result_text.delete("1.0", tk.END)
        self.feature_text.delete("1.0", tk.END)
        
        if 'error' in result:
            self.result_text.insert(tk.END, result['error'])
            return
        
        # Вывод автора и вероятностей
        self.result_text.insert(tk.END, f"Предполагаемый автор: {result['author']}\n\n")
        self.result_text.insert(tk.END, "Вероятности по авторам:\n")
        
        # Сортируем по убыванию вероятности
        sorted_probs = sorted(result['probs'], key=lambda x: -x[1])
        for author, prob in sorted_probs[:5]:
            self.result_text.insert(tk.END, f"{author}: {prob:.2%}\n")
        
        # График
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        authors = [x[0] for x in sorted_probs[:5]]
        probs = [x[1] for x in sorted_probs[:5]]
        
        bars = ax.barh(authors, probs, color='skyblue')
        ax.set_xlabel('Вероятность')
        ax.set_title('Распределение вероятностей')
        
        # Добавляем значения на столбцы
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.2%}',
                    va='center', ha='left')
        
        self.canvas.draw()
        
        # Вывод ключевых признаков
        if 'features' in result:
            features = result['features']
            self.feature_text.insert(tk.END, "Основные стилистические признаки:\n\n")
            
            if features['interpretations']:
                self.feature_text.insert(tk.END, "Наиболее значимые:\n")
                for item in features['interpretations']:
                    self.feature_text.insert(tk.END, f"• {item}\n")
                self.feature_text.insert(tk.END, "\n")
            
            # Дополнительные признаки
            self.feature_text.insert(tk.END, "Все признаки:\n")
            for name, value in zip(features['feature_names'], features['feature_values'].values()):
                self.feature_text.insert(tk.END, f"{name}: {value:.3f}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = PoetryApp(root)
    root.mainloop()