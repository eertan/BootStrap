from src.model import BootModel
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/erol/Projects/stocks/Data/test_dataset.csv")
df = df.loc[df['Symbol'] == 'AAPL', ["Date","Close","ema50",'Volume','bol_up','bol_down']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index().drop(columns=['Date','index'])
print(df)

mod_obj = BootModel(df,['ema50','Volume','bol_up','bol_down'],'Close',intercept=True)
aa = mod_obj.get_coef_dists()
print(aa)
plt.plot(aa)
plt.show()