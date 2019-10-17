import pandas as pd

for x in range(3):
    u=input("User:")
    b=input("Bot:")
    dfm = pd.read_excel("conv_logs.xlsx", "Sheet1")
    # dfm = pd.read_csv("conv_logs.csv")
    df1=pd.Series([u,b],index=['user','bot'])
    dfm=dfm.append(df1,ignore_index=True)
    #print(dfm)
    # dfm.to_csv('conv_logs.csv',header=True)
    dfm.to_excel("conv_logs.xlsx", sheet_name="Sheet1",index = False)
    # with pd.ExcelWriter('conv_logs.xlsx',index = False) as writer:

print('done')