# Анализ тональности русскоязычных текстов

## Описание проекта
Данный проект представляет собой систему анализа тональности (sentiment analysis) текстов на русском языке. Система использует модель BERT (bert-base-multilingual-cased) и способна классифицировать тексты на три категории:
- Позитивные
- Негативные
- Нейтральные

## Основные возможности
- Предварительная обработка текста (удаление HTML-тегов, ссылок, специальных символов)
- Балансировка классов в датасете
- Аугментация данных для улучшения обучения
- Классификация текстов с использованием BERT
- Визуализация результатов обучения

## Технологии
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- pandas
- scikit-learn
- matplotlib
- tqdm

## Установка проекта

### Предварительные требования
- Python 3.8 или выше
- pip
- Git

### Шаги по установке
1. Клонируйте репозиторий:
```bash
git clone https://github.com/ваш-username/russian-sentiment-analysis.git
cd russian-sentiment-analysis
```

2. Создайте виртуальное окружение:
```
python -m venv venv
```
3. Активируйте виртуальное окружение: Для Windows:
```
venv\Scripts\activate
```
Для Linux/Mac:
```
source venv/bin/activate
```
4. Установите необходимые зависимости:
```
pip install -r requirements.txt
```
## Использование

### Пример базового использования
```
from src.model import SentimentClassifier
from src.data_processing import clean_text
from transformers import BertTokenizer

# Инициализация модели и токенизатора
model = SentimentClassifier('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Пример обработки текста
text = "Я в жизни ничего глупее не слышал!"
```

### Обучение модели
```
from src.training import train_model
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Настройка оптимизатора и функции потерь
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()

# Обучение модели
train_losses, train_accuracies, test_accuracies = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=3
)
```

### Структура проекта
```
russian-sentiment-analysis/
├── data/               # Папка для датасетов
├── models/            # Папка для сохранения обученных моделей
├── src/               # Исходный код
│   ├── data_processing.py  # Обработка данных
│   ├── model.py           # Определение модели
│   ├── dataset.py         # Класс датасета
│   ├── training.py        # Функции для обучения
│   └── utils.py           # Вспомогательные функции
├── tests/             # Тесты

```
### Автор

Fireman52
