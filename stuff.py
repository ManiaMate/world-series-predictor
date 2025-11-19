import ssl
import pandas as pd
from pybaseball import team_game_logs

batting_logs = team_game_logs(2019, "ATL")
print(batting_logs.head())

ssl._create_default_https_context = ssl._create_unverified_context

# Example of getting Tark Skubal's stats
url = "https://www.baseball-reference.com/players/gl.fcgi?id=skubata01&t=p&year=2025"
df = pd.read_html(url)[0]

# Clean table
df = df[df["Rk"].astype(str) != "Rk"]

df = df[~df["Rk"].isna()]

df["Rk"] = pd.to_numeric(df["Rk"], errors="coerce")

df = df.dropna(subset=["Rk"])

df.reset_index(drop=True, inplace=True)

df.to_csv("help.csv", index=False)

