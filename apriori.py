from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd

dataset = [['Milk','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
           ['Dill','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
           ['Milk','Apple','Kidney Beans','Eggs'],
           ['Milk','Unicorn','Corn','Kidney Beans','Yogurt'],
           ['Corn','Onion','Kidney Beans','Ice cream','Eggs']]

te=TransactionEncoder()
Trans_array=te.fit(dataset).transform(dataset)
df=pd.DataFrame(Trans_array,columns=te.columns_)
#print(df)

ap=apriori(df,min_support=0.6,use_colnames=True)


ap['length']=ap['itemsets'].apply(lambda x:len(x))
print(ap)
print(ap[(ap['length']==2) & (ap['support']>=0.8)])
