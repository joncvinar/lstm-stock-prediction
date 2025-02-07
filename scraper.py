import requests
from bs4 import BeautifulSoup

# Scrape CNN
def get_cnn_news(ticker):
    url = f"https://www.cnn.com/search?q={ticker}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [h.text for h in soup.find_all('h3')][:5]
    return headlines

# Scrape Barron's
def get_barrons_news(ticker):
    url = f"https://www.barrons.com/search?q={ticker}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [h.text for h in soup.find_all('h2')][:5]
    return headlines

