import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def normalize_data(value):
    ret = value
    if value == 1: ret = 0
    elif value > 1: ret = 1
    return ret

# Import data
df = pd.read_csv('/workspace/boilerplate-medical-data-visualizer/medical_examination.csv')

# Add 'overweight' column
df['bmi'] = df['weight'] / (df['height']/100)**2 #create bmi column first for debugging purpose
df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25.0 else 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
normalizer = np.vectorize(normalize_data)
df['gluc'] = df.apply(lambda v: normalizer(v['gluc']), axis=1)
df['cholesterol'] = df.apply(lambda v: normalizer(v['cholesterol']), axis=1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars=['cardio'],value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat.rename(columns={'value':'total'}, inplace=True)
    #cardio 1, value 1
    df_cat1=df_cat[(df_cat['cardio']==1) & (df_cat['total']==1)].groupby(['variable']).count().reset_index()
    df_cat1["cardio"]=1
    df_cat1["value"]=1
    #cardio 1, value 0
    df_cat2=df_cat[(df_cat['cardio']==1) & (df_cat['total']==0)].groupby(['variable']).count().reset_index()
    df_cat2["cardio"]=1
    df_cat2["value"]=0
    #cardio 0, value=1
    df_cat3=df_cat[(df_cat['cardio']==0) & (df_cat['total']==1)].groupby(['variable']).count().reset_index()
    df_cat3["cardio"]=0
    df_cat3["value"]=1
    #cardio 0, value=0
    df_cat4=df_cat[(df_cat['cardio']==0) & (df_cat['total']==0)].groupby(['variable']).count().reset_index()
    df_cat4["cardio"]=0
    df_cat4["value"]=0
    #combine to finalize
    df_cat=pd.concat([df_cat1, df_cat2, df_cat3, df_cat4])


    # Draw the catplot with 'sns.catplot()'
    plot = sns.catplot(
    data=df_cat, kind="bar",
    x="variable", y="total", col="cardio", hue='value',
    errorbar="sd", palette="dark", alpha=.6, height=6)
    

    # Get the figure for the output
    fig = plot.figure


    # Do not modify the next two lines
    fig.savefig('catplot2.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['height']>=df['height'].quantile(0.025)) & 
                (df['height']<=df['height'].quantile(0.975)) &
                (df['weight']>=df['weight'].quantile(0.025)) &
                (df['weight']<=df['weight'].quantile(0.975))].reset_index(drop=True)
    df_heat.rename(columns={'sex':'gender'}, inplace=True)
    df_heat = df_heat.drop('bmi', axis=1)

    """df_heat = df[(df['height']>=df['height'].quantile(0.025))]
    df_heat = df_heat[(df_heat['height']>=df_heat['height'].quantile(0.975))]
    df_heat = df_heat[(df_heat['weight']>=df_heat['weight'].quantile(0.025))]
    df_heat = df_heat[(df_heat['weight']<=df_heat['weight'].quantile(0.975))]
    df_heat.reset_index(drop=True)"""


    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))

    # Set up the matplotlib figure
    fig, ax = plt.subplots()


    # Draw the heatmap with 'sns.heatmap()'
    fig=sns.heatmap(corr, mask=mask, annot=True, fmt=".1f")
    #fig=fig.figure

    # Do not modify the next two lines
    fig.figure.savefig('heatmap.png')
    return fig
