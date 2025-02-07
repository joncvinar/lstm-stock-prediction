from flask import Flask, request, jsonify
from model import train_model, predict_stock
from scraper import get_cnn_news, get_barrons_news  # Removed Reddit

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get("ticker")

    model, scaler = train_model(ticker)
    predicted_price = predict_stock(model, scaler, ticker)

    news = get_cnn_news(ticker) + get_barrons_news(ticker)

    return jsonify({
        "ticker": ticker,
        "predicted_price": float(predicted_price),  # Convert float32 -> float
        "news": news,
        "recommendation": "BUY" if float(predicted_price) > 100 else "SELL"
    })

if __name__ == "__main__":
    app.run(debug=True)
