import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load results
df = pd.read_csv(r"C:\Users\PRIYANSHU\OneDrive\Desktop\ESA_Project\fusion\results.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')  # convert manually

# Bar Chart: Emotion Frequency
emotion_counts = df[["FaceEmotion", "VoiceEmotion", "TextSentiment"]].melt()["value"].value_counts()

plt.figure(figsize=(8, 5))
emotion_counts.plot(kind="bar", color="skyblue")
plt.title("Emotion Frequency (All Sources)")
plt.ylabel("Count")
plt.xlabel("Emotion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/emotion_bar.png")
plt.show()

# Line Chart: Emotions over Time
df["Date"] = df["Timestamp"].dt.date
daily_emotions = df.groupby("Date")[["FaceEmotion", "VoiceEmotion"]].agg(lambda x: x.mode()[0])

plt.figure(figsize=(10, 5))
for col in daily_emotions.columns:
    daily_emotions[col].value_counts().sort_index().plot(label=col)
plt.title("Dominant Emotions Over Time")
plt.xlabel("Date")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("charts/emotion_trends.png")
plt.show()

# Pie Chart: Text Sentiment
plt.figure(figsize=(5, 5))
df["TextSentiment"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, shadow=True)
plt.title("Text Sentiment Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/sentiment_pie.png")
plt.show()
