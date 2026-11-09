"""LANGKAH 1 - CRAWLING ulasan dari Google Play Store."""
from google_play_scraper import reviews, Sort
import pandas as pd

APP_ID = "com.gojek.app"   # id url aplikasi pada play store
JUMLAH = 5000

hasil, _ = reviews(
    APP_ID,
    lang="id",        # bahasa ulasan
    country="id",     # negara
    sort=Sort.NEWEST, # ambil ulasan terbaru
    count=JUMLAH,
)

df = pd.DataFrame(hasil)[["content", "score"]]
df.columns = ["ulasan", "rating"]
df.to_csv("data_ulasan.csv", index=False)
print(f"Tersimpan {len(df)} ulasan ke data_ulasan.csv")
print(df.head())
