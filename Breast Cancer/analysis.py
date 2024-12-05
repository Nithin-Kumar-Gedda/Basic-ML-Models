import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pycaret.classification import * # type: ignore

warnings.filterwarnings('ignore')

df = pd.read_csv("data.csv")
df=df.drop(columns=['id','Unnamed: 32'], axis= 1)
# print(df.info())

# Subplots
df_temp = df.drop(columns=['diagnosis'], axis=1)
fig, ax = plt.subplots(ncols=6, nrows= 5, figsize=(20,10))
index = 0
ax = ax.flatten()
for col in df_temp.columns:
    # sns.displot(df[col], ax = ax[index])
    index +=1
# plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
# plt.show()

clf = setup(df, target='diagnosis') # type: ignore 
# best_model=compare_models() # type: ignore
model = create_model('ada') # type: ignore
best_model = tune_model(model) # type: ignore
# evaluate_model(best_model) # type: ignore

plot_model(estimator=best_model, plot='confusion_matrix')