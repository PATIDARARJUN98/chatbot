


import pandas as pd
df = pd.read_excel("Book1.xlsx")
n = input("Enter your Invoice no. :")
print(df[df['Invoice no.']==n].index.tolist())
#df["Remarks"] = df["Invoice no."].str.finf(n)
#df[df.apply(lambda row: row.astype(str).str.contains('DEF').any().axis=1)])




