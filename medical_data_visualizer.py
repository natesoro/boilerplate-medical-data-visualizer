import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def normalize_data(value):
    ret = value
    if value == 1: ret = 0
    elif value > 1: ret = 1
    return ret

# 1
df = pd.read_csv('/workspace/boilerplate-medical-data-visualizer/medical_examination.csv')

# 2
df['bmi'] = df['weight'] / (df['height']/100)**2 #create bmi column first for debugging purpose
df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25.0 else 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
normalizer = np.vectorize(normalize_data)
df['gluc'] = df.apply(lambda v: normalizer(v['gluc']), axis=1)
df['cholesterol'] = df.apply(lambda v: normalizer(v['cholesterol']), axis=1)
df=df.astype({'overweight':int, 'cholesterol':int, 'gluc':int})#normalize datatypes for these columns so that they are treated the same during grouping

# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(id_vars=['cardio'],value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6
    df_cat['total']=1
    df_cat=df_cat.groupby(['cardio', 'variable', 'value'])['total'].sum().reset_index()

    # 7
    plot = sns.catplot(
    data=df_cat, kind="bar",
    x="variable", y="total", col="cardio", hue='value',
    errorbar="sd", palette="dark", alpha=.6, height=6)
    

    # 8
    fig = plot.figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo']<=df['ap_hi']) &
        (df['height']>=df['height'].quantile(0.025)) & 
        (df['height']<=df['height'].quantile(0.975)) &
        (df['weight']>=df['weight'].quantile(0.025)) &
        (df['weight']<=df['weight'].quantile(0.975))].reset_index(drop=True)
    df_heat.rename(columns={'sex':'gender'}, inplace=True)
    df_heat = df_heat.drop('bmi', axis=1)

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr))

    # 14
    fig, ax = plt.subplots(figsize=(9,6))


    # 15
    fig=sns.heatmap(corr, mask=mask,annot=True, fmt=".1f",vmin=corr.min().min(), vmax=.3, 
                cbar_kws={'shrink': 0.5, 'ticks':[.24,.16,.08,.00,-0.08]}, ax=ax, center=0).figure


    # 16
    fig.savefig('heatmap.png')
    return fig
