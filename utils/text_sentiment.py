from transformers import pipeline

# Load sentiment analysis model once
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment_bert(text):
    if not text.strip():
        return "Neutral"
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    if "POS" in label.upper():
        return "Positive"
    elif "NEG" in label.upper():
        return "Negative"
    else:
        return "Neutral"
