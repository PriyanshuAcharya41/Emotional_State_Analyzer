from utils.text_sentiment import analyze_sentiment_bert

print("Text Sentiment Tester (BERT-based)")
print("Type something and press Enter. Type 'exit' to quit.\n")

while True:
    user_input = input(" You said: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    sentiment = analyze_sentiment_bert(user_input)
    print(f" Detected Sentiment: {sentiment}\n")
