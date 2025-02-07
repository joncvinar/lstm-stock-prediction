import sys
import requests
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt6.QtCore import Qt


class StockPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Stock Predictor")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        self.label = QLabel("Enter Stock Tickers (comma separated):")
        layout.addWidget(self.label)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("AAPL, TSLA, GOOG")
        layout.addWidget(self.text_input)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.get_predictions)
        layout.addWidget(self.predict_button)

        self.result_label = QLabel("Results will appear here.")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def get_predictions(self):
        tickers = self.text_input.toPlainText().strip()
        if not tickers:
            self.result_label.setText("Please enter stock tickers.")
            return

        results = ""
        for ticker in tickers.split(","):
            ticker = ticker.strip().upper()
            response = requests.post("http://127.0.0.1:5000/predict", json={"ticker": ticker})
            if response.status_code == 200:
                data = response.json()
                results += f"\nTicker: {data['ticker']}\nPredicted Price: {data['predicted_price']}\nRecommendation: {data['recommendation']}\nNews: {', '.join(data['news'])}\n---\n"
            else:
                results += f"\nError fetching prediction for {ticker}\n"

        self.result_label.setText(results)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictorApp()
    window.show()
    sys.exit(app.exec())
