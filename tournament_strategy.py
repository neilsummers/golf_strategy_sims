import os

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

class Baseline(object):
    def __init__(self):
        self.interpolators = {}
        filedir = os.path.dirname(__file__)
        for lie in ['fairway', 'green', 'recovery', 'rough', 'sand', 'tee']:
            data = pd.read_csv(os.path.join(filedir, 'strokes_gained_data', f'{lie}.csv'))
            self.__setattr__(lie, data)
            interpolator = sp.interpolate.interp1d(data.Dist, data.Strokes, fill_value='extrapolate',
                                                   assume_sorted=data.Dist.is_monotonic_increasing)
            self.interpolators[lie] = interpolator

    def strokes_from(self, lie, distance):
        return round(float(self.interpolators[lie.lower()](distance)), 3)

def generate_lie(x):
    if x > 25:
        return 'Rough'
    elif x > 24:
        return 'Fairway'
    elif x < -6:
        return 'Sand'
    elif x < -5:
        return 'Fairway'
    else:
        return 'Green'

def mc_predicted_score(target=0, std=1):
    baseline = Baseline()
    df = pd.DataFrame(sp.stats.norm(loc=target, scale=std).rvs(18), columns=['distance'], index=pd.Index(range(1, 19), name='hole'))
    df = df.assign(lie=df['distance'].apply(generate_lie))
    df = df.assign(strokes_from=df.apply(lambda x: baseline.strokes_from(x['lie'], abs(x['distance'])), axis=1))
    return int(round(df.strokes_from.sum(), 0)+40)

dtypes = {
    'target': pd.Int64Dtype(),
    'std': np.dtype('float64'),
    'round1': pd.Int64Dtype(),
    'round2': pd.Int64Dtype(),
    'score36': pd.Int64Dtype(),
    'round3': pd.Int64Dtype(),
    'round4': pd.Int64Dtype(),
    'score72': pd.Int64Dtype()
}

#https://golf.com/lifestyle/2023-players-championship-purse-payout-money/
# 75 places
payouts = np.array([4500000, 2725000, 1725000, 1225000, 1025000,  906250,  843750,
                    781250,  731250,  681250,  631250,  581250,  531250,  481250,
                    456250,  431250,  406250,  381250,  356250,  331250,  306250,
                    281250,  261250,  241250,  221250,  201250,  193750,  186250,
                    178750,  171250,  163750,  156250,  148750,  142500,  136250,
                    130000,  123750,  118750,  113750,  108750,  103750,   98750,
                     93750,   88750,   83750,   78750,   73750,   69750,   66250,
                     64250,   62750,   61250,   60250,   59250,   58750,   58250,
                     57750,   57250,   56750,   56250,   55750,   55250,   54750,
                     54250,   53750,   53250,   52750,   52250,   51750,   51250,
                     50750,   50250,   49750,   49250,   48750])
for i in range(75):
    payouts = np.append(payouts, payouts[-1]-500)
payouts = (payouts/2).astype(int)

def prize_money(df):
    df['prize_money'] = payouts[:len(df)]
    df.iloc[df.dropna().shape[0]:].loc[:,'prize_money'] = 0
    prize_money = df.groupby('score72')['prize_money'].mean()
    return df.drop('prize_money', axis=1).join(prize_money, on='score72').fillna({'prize_money':0}).astype({'prize_money': int})

def position(df):
    df['position'] = df.reset_index().index+1
    position = df.groupby('score72')['position'].first()
    return df.drop('position', axis=1).join(position, on='score72').astype({'position': pd.Int64Dtype()})

def tournament(df):
    df = df.assign(round1=df.apply(lambda row: mc_predicted_score(row['target'], row['std']), axis=1))
    df = df.assign(round2=df.apply(lambda row: mc_predicted_score(row['target'], row['std']), axis=1))
    df = df.eval('score36 = round1 + round2')
    df.sort_values('score36', ascending=True, inplace=True)
    cut = df.iloc[70]['score36']
    df_cut = df.query('score36 > @cut')
    df = df.query('score36 <= @cut')
    df = df.assign(round3=df.apply(lambda row: mc_predicted_score(row['target'], row['std']), axis=1))
    df = df.assign(round4=df.apply(lambda row: mc_predicted_score(row['target'], row['std']), axis=1))
    df = df.eval('score72 = score36 + round3 + round4')
    df.sort_values('score72', ascending=True, inplace=True)
    df = pd.concat([df, df_cut]).astype(dtypes)
    df = position(df)
    df = prize_money(df)
    return df

if __name__ == '__main__':
    pd.options.display.max_rows = 20
    np.random.seed(0)
    std = 8/sp.stats.norm.ppf(0.8)

    x = np.arange(0., 10.1, 0.1)

    fig = plt.figure(figsize=(6.4, 6.4*1))
    ax1 = fig.add_subplot(1, 1, 1)
    step = 0.2
    baseline = Baseline()
    yy = np.array([
        np.array([baseline.strokes_from(generate_lie(z), abs(z)) * sp.stats.norm(loc=x_, scale=std).pdf(z)
                  for z in np.arange(-40, 40+step, step)]).sum()*step
        for x_ in x
    ])
    ax1.plot(x, yy, label='∫ P(x|target)*SG(x) dx')
    ax1.set_xlim([x.min(), x.max()])
    ax1.set_xlabel('target (yards)')
    ax1.set_ylabel('∫ P(x|target)*SG(x) dx')
    ax1.set_title('Strokes gained averaged over shot distribution for a given target')
    fig.tight_layout()
    fig.savefig('optimal_target.jpeg')

    #num_players = 140
    #players = pd.DataFrame(list([5]+[0]*(num_players-1)), columns=['target'], index=pd.Index(range(1, num_players+1), name='player'))
    #players = pd.DataFrame(list([5]*70 + [0]*70), columns=['target'], index=pd.Index(range(1, num_players+1), name='player'))
    num_players = 143
    players = pd.DataFrame(list(range(0, 11))*int(num_players/11), columns=['target'], index=pd.Index(range(1, num_players+1), name='player'))
    players['std'] = std
    #df = tournament(players)
    #print(df.head(10))
    #df.dropna().groupby('target').score72.mean().plot()
    #plt.scatter(df.dropna()['target'], df.dropna()['score72'])
    season = []
    for _ in range(30):
        df = tournament(players)
        season.append(df)

    total_prize_money = pd.DataFrame(pd.concat([t['prize_money'] for t in season], axis=1).sort_index().sum(axis=1).sort_values(ascending=False), columns=['prize_money']).join(players)
    wins = pd.concat([t[t.position<=1] for t in season]).index.to_series().value_counts().sort_index().rename('wins')
    top3 = pd.concat([t[t.position<=3] for t in season]).index.to_series().value_counts().sort_index().rename('top 3')
    top10 = pd.concat([t[t.position <= 10] for t in season]).index.to_series().value_counts().sort_index().rename('top 10')
    top25 = pd.concat([t[t.position <= 25] for t in season]).index.to_series().value_counts().sort_index().rename('top 25')
    cuts_made = pd.concat([t.dropna(subset=['score72']) for t in season]).index.to_series().value_counts().sort_index().rename('cuts made')
    summary = pd.concat([wins, top3, top10, top25, cuts_made], axis=1).sort_index().fillna(0).astype(int).join(players)

    print(total_prize_money.groupby('target').sum()['prize_money'])
    fig = plt.figure(figsize=(6.4, 6.4))
    ax2 = total_prize_money.groupby('target').sum()['prize_money'].plot()
    ax2.set_xlim([0, 10])
    ax2.set_xlabel('target (yards)')
    ax2.set_ylabel('total prize money ($)')
    ax2.set_title('Total prize money for a season per target')
    fig.tight_layout()
    fig.savefig('prize_money.jpeg')

    print(summary.groupby('target').sum()[['wins', 'top 3', 'top 10', 'top 25', 'cuts made']])
    print(summary.groupby('target').sum()[['wins', 'top 3', 'top 10', 'top 25', 'cuts made']].to_markdown())
    ax3s = summary.groupby('target').sum()[['wins', 'top 3', 'top 10', 'top 25', 'cuts made']].plot(subplots=True, sharex=True, figsize=(6.4, 6.4*2))
    ax3 = ax3s[0]
    fig = ax3.get_figure()
    fig.subplots_adjust(wspace=0, hspace=0)
    ax3.set_xlim([0, 10])
    ax3s[-1].set_xlabel('target (yards)')
    ax3.set_title('Number of placed finishes per target')
    fig.tight_layout()
    fig.savefig('places.jpeg')

    fig = plt.figure(figsize=(8.5, 6))
    z = np.arange(-10, 30+step, step)
    y0 = sp.stats.norm(loc=0, scale=std).pdf(z)
    y5 = sp.stats.norm(loc=4.6, scale=std).pdf(z)
    ax = fig.add_subplot()
    ax.plot(z, y0, '--', color='b', label='aim at pin')
    ax.plot(z, y5, '--', color='orange', label='aim at optimal target')
    ax.legend()
    ax.set_xlim([-10, 30])
    ax.set_ylim([0, 0.05])
    ax.plot([-6, -6], [0, 0.015], '-', color='y')
    ax.plot([-5, -5], [0, 0.015], '-', color='g')
    ax.plot([24, 24], [0, 0.015], '-', color='g')
    ax.plot([25, 25], [0, 0.015], '-', color='g')
    ax.set_xlabel('distance from pin (yards)')
    ax.set_ylabel('Probability')
    ax.plot([0, 0], [0, sp.stats.norm(loc=0, scale=std).pdf(0)], 'b--')
    ax.plot([4.6, 4.6], [0, sp.stats.norm(loc=4.6, scale=std).pdf(4.6)], '--', color='orange')
    ax.text(-7, 0.01, 'SAND', rotation='vertical', color='y', fontsize='small')
    ax.text(-6, 0.01, 'FAIRWAY', rotation='vertical', color='g', fontsize='small')
    ax.text(-5, 0.01, 'GREEN', rotation='vertical', color='g', fontsize='small')
    ax.text(23, 0.01, 'GREEN', rotation='vertical', color='g', fontsize='small')
    ax.text(24, 0.01, 'FAIRWAY', rotation='vertical', color='g', fontsize='small')
    ax.text(25, 0.01, 'ROUGH', rotation='vertical', color='g', fontsize='small')
    ax.set_title('Shot Distributions and Target Description')
    fig.tight_layout()
    fig.savefig('approach.jpeg')
