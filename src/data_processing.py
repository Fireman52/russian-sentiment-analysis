import re
import pandas as pd
import random
from typing import Dict, Any


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, HTML tags, and special characters.

    Args:
        text (str): Input text to clean

    Returns:
        str: Cleaned text
    """
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance classes in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame with 'sentiment' column

    Returns:
        pd.DataFrame: Balanced DataFrame
    """
    neutral = df[df['sentiment'] == 'neutral']
    positive = df[df['sentiment'] == 'positive']
    negative = df[df['sentiment'] == 'negative']

    target_size = max(len(positive), len(negative))
    neutral_sampled = neutral.sample(n=target_size, random_state=42)
    balanced_df = pd.concat([neutral_sampled, positive, negative], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df
