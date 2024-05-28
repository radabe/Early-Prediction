import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 442)

case_length_no={1:0,2:257,3:445,4:910,5:726,6:1525,7:3053,8:3576,9:3891,10:3683,11:4332,12:8221,13:11690,14:11621,15:7574,16:8085,17:6835,
 18:3816,19:3608,20:3508,21:2197,22:1683,23:1821,24:2298,25:1554,26:1186,27:1260,28:1513,29:10472,30:0}

palette = ['#b2df8a','#33a02c','#a6cee3','#1f78b4','#cab2d6','#6a3d9a','#fb9a99','#e31a1c','#fdbf6f']#'#ffff99','#ff7f00'

sns.barplot(data=pd.DataFrame(np.cumsum(pd.DataFrame(data=case_length_no, index=(range(1))).loc[0])).T)
sns.barplot(pd.DataFrame(data=case_length_no, index=(range(1))))


alg_list=['rf','xgb','lstm']
app_list=['_agg_base','_agg','_plus','_idx']
app_list3=['_AGG','_AAGG','_CAGG','_IDX']
fold_list=['Results_RF','Results_XGB', 'Results_LSTM']
event_max=31

folders=3
appr=4
eval_columns=[]
cls_rows=['E 1']

for i in range(1,30):
    eval_columns.append('case_length_{}'.format(i))
    cls_rows.append('E {}'.format(i+1))
    
eval_columns.append('Approach')
eval_columns.append('Prediction Point')

full_set=pd.DataFrame(columns=eval_columns)

for f in range(folders):
    
    folder=fold_list[f]
    alg=alg_list[f]
    
    for a in range(appr):
        
        app=app_list[a]
        app3=app_list3[a]
        print(alg+app)
        app2=app
        i=0
        for cls in ['','_c','_c2']:
            eval_set=pd.DataFrame(columns=range(0,29))
            av_list=[]
            for event_nr in range(1,event_max):
                if event_nr==31:
                    event_nr='all'
                    
                if alg=='lstm':
                    app='_lstm'
                    app2=''
                    app3=''
                    
                file_read='Results/{}{}/av_roc{}{}_{}.csv'.format(folder,app3,cls,app,event_nr)
                eval_set=pd.concat([eval_set,pd.read_csv(file_read, header=None)])
                eval_set['Approach']='{}{}{}'.format(alg.upper(),app3,cls)
                
                if alg=='lstm':
                    eval_set['Approach']='LSTM'


            eval_set['Prediciton Point']=cls_rows
            eval_set.columns=eval_columns
            eval_set.reset_index(drop=True, inplace=True)
            for i in range(1,30):
                eval_set.loc[i:,'case_length_{}'.format(i)]=np.NaN

            full_set=pd.concat([full_set,eval_set])
    
            if alg=='lstm':
                break
        if alg=='lstm':
            break


full_set.reset_index(drop=True, inplace=True)


app_list=full_set['Approach'].unique()
app_list=['RF_AGG','RF_AAGG','RF_CAGG','RF_IDX','XGB_AGG','XGB_AAGG','XGB_CAGG','XGB_IDX','LSTM']
app_list

long_set=pd.DataFrame(columns=['Approach','Case_Length','ROC','Prediction Point'])


for approach in app_list:#list(full_set['Approach'].unique()):
    for i in range(2,30):
        temp=full_set[full_set['Approach']==approach]
        temp['ROC']=temp['case_length_{}'.format(i)]
        temp['Case_Length']='CL_{}'.format(i)
        temp=temp[['ROC','Case_Length','Prediction Point','Approach']]
        long_set=pd.concat([long_set,temp])
    
long_set.reset_index(drop=True, inplace=True)

long_set.dropna(how='any', inplace=True)
long_set.reset_index(inplace=True, drop=True)
#long_set['ROC']=np.where(long_set['ROC']<0.5,1-long_set['ROC'],long_set['ROC'])

long_set_weighted=long_set.copy(deep=True)
total=111340
for i in range(2,30):
    long_set_weighted['ROC']=np.where(long_set_weighted['Case_Length']=='CL_{}'.format(i),long_set_weighted['ROC']*case_length_no[i],long_set_weighted['ROC'])

    
for i in range(1,30):
    long_set_weighted['ROC']=np.where(long_set_weighted['Prediction Point']=='E {}'.format(i),long_set_weighted['ROC']/total,long_set_weighted['ROC'])
    total-=case_length_no[i]

long_set.dropna(how='any', inplace=True)
long_set.reset_index(inplace=True, drop=True)


mean_set_pp=long_set.groupby(['Approach','Prediction Point']).mean(['ROC'])
mean_set_pp.reset_index(inplace=True)

mean_set_pp_w=long_set_weighted.groupby(['Approach','Prediction Point']).sum(['ROC'])
mean_set_pp_w.reset_index(inplace=True)

mean_set_cl=long_set.groupby(['Approach','Case_Length']).mean(['ROC'])
mean_set_cl.reset_index(inplace=True)

approach_list=list(mean_set_cl['Approach'].unique())
cl_list=list(mean_set_cl['Case_Length'].unique())
pp_list=list(mean_set_pp['Prediction Point'].unique())

mean_set_pp_w

display_df_cl=pd.DataFrame(columns=['Approach'])


for app in app_list:
    temp=mean_set_cl[mean_set_cl['Approach']==app]
    temp=temp[['Case_Length','ROC']].T
    temp.columns=temp.iloc[0,:]
    temp.reset_index(drop=True, inplace=True)
    temp.drop(0, inplace=True)
    temp['Approach']=app
    display_df_cl=pd.concat([display_df_cl,temp])
    
display_df_cl.set_index('Approach',inplace=True)
display_df_cl=display_df_cl[sorted(display_df_cl.columns, key=lambda x: int(x[3:]))]
display_df_cl


display_df_pp=pd.DataFrame(columns=['Approach'])


for app in app_list:
    temp=mean_set_pp[mean_set_pp['Approach']==app]
    temp=temp[['Prediction Point','ROC']].T
    temp.columns=temp.iloc[0,:]
    temp.reset_index(drop=True, inplace=True)
    temp.drop(0, inplace=True)
    temp['Approach']=app
    display_df_pp=pd.concat([display_df_pp,temp])
    
display_df_pp.set_index('Approach',inplace=True)
display_df_pp=display_df_pp[sorted(display_df_pp.columns, key=lambda x: int(x[2:]))]
display_df_pp.to_excel('display_pp_roc.xlsx')

display_df_pp_w=pd.DataFrame(columns=['Approach'])


for app in app_list:
    temp=mean_set_pp_w[mean_set_pp_w['Approach']==app]
    temp=temp[['Prediction Point','ROC']].T
    temp.columns=temp.iloc[0,:]
    temp.reset_index(drop=True, inplace=True)
    temp.drop(0, inplace=True)
    temp['Approach']=app
    display_df_pp_w=pd.concat([display_df_pp_w,temp])
    
display_df_pp_w.set_index('Approach',inplace=True)
display_df_pp_w=display_df_pp_w[sorted(display_df_pp_w.columns, key=lambda x: int(x[2:]))]
display_df_pp_w.columns=range(1,30)
display_df_pp_w.T

sns.set(rc={'figure.figsize':(3,3), 'xtick.top' : False, 'figure.dpi':1200})
sns.set_theme(style="whitegrid", rc={'grid.linewidth':0.2})
plt.ticklabel_format(style='plain', axis='y')
sns.set_palette(palette)
sns.set_context("paper")
s=sns.lineplot(data=display_df_pp_w.T,legend=True, dashes=False, markers=True, markevery=4, linewidth= 1,markersize=5,markeredgewidth=0.1)
s.set(ylim=(0.4,0.85))
s.set(xlim=(1,29))

s.axes.set_title("AUC-ROC vs. prefix length for all approaches",fontsize=10)
s.set_xlabel("Prefix length",fontsize=8)
s.set_ylabel("AUC-ROC",fontsize=8)
s.tick_params(labelsize=5)

s.spines['left'].set_linewidth(0.5)
s.spines['bottom'].set_linewidth(0.5)
s.spines['top'].set_linewidth(0.5)
s.spines['right'].set_linewidth(0.5)
plt.xticks(rotation=90)
sns.move_legend(s, "center",bbox_to_anchor=(1.2, 0.5), ncol=1, title='Approach', frameon=False,fontsize=6,title_fontsize=7)

meanlong_auc_roc_df=long_set.groupby(['Approach','Case_Length'],sort=False).mean('ROC').reset_index()
mean_auc_roc_df=long_set.groupby(['Approach','Case_Length'],sort=False).mean('ROC').reset_index().pivot(index='Approach', columns='Case_Length')

mean_auc_roc_df

sns.set(rc={'figure.figsize':(7,1.5), 'xtick.top' : False, 'figure.dpi':1200})
sns.set_theme(style="whitegrid", rc={'grid.linewidth':0.2})
plt.ticklabel_format(style='plain', axis='y')
sns.set_palette(palette)
sns.set_context("paper")

s=sns.lineplot(data=meanlong_auc_roc_df, x='Case_Length',y='ROC', hue='Approach', style='Approach',
               legend=True, dashes=False, markers=True, markevery=3, linewidth= 1,markersize=5,markeredgewidth=0.1)

s.axes.set_title("Mean AUC-ROC vs. Case Length",fontsize=10)
s.set_xlabel("Case Length",fontsize=8)
s.set_ylabel("Mean AUC-ROC",fontsize=8)
s.tick_params(labelsize=5)
#plt.legend()
sns.move_legend(s, "lower center",bbox_to_anchor=(0.5,-0.8), ncol=9, title='Approach', frameon=False,fontsize=6,title_fontsize=7)
s.spines['left'].set_linewidth(0.5)
s.spines['bottom'].set_linewidth(0.5)
s.spines['top'].set_linewidth(0.5)
s.spines['right'].set_linewidth(0.5)
plt.xticks(rotation=90)

plt.savefig('Mean_AUC_ROC.svg',bbox_inches='tight')
long_set['Prefix Length']=long_set['Prediction Point'].apply(lambda x: int(x[2:]))

c=1

for i in range (2,30):

    if i%4==0:
        c+=1
    sns.set(rc={ 'figure.figsize':(1,1), 'xtick.top' : False, 'figure.dpi':600})
    sns.set_theme(style="whitegrid", rc={'grid.linewidth':0.2})
    plt.ticklabel_format(style='plain', axis='y')
    sns.set_palette(palette)
    sns.set_context("paper")
    sns.despine()
    s=sns.lineplot(data=long_set[long_set['Case_Length']=='CL_{}'.format(i)], 
                   x='Prefix Length', y="ROC", hue='Approach',style='Approach',
                   dashes=False, markers=True, markevery=c,legend=False,
                   linewidth= 0.5,markersize=2.5,markeredgewidth=0.1, errorbar=None)
    
    
    s.set(ylim=(0.4,1))
    s.set(xlim=(1,i))
    s.axes.set_title("Case Length {}".format(i),fontsize=8)
    s.set_xlabel(None)
    s.set_ylabel(None)
    s.tick_params(labelsize=5)
    s.set_xticks(range(1,i+1,c))
    s.spines['left'].set_linewidth(0.3)
    s.spines['bottom'].set_linewidth(0.3)

    plt.savefig('grid_CL_{}.svg'.format(i),bbox_inches='tight')
    plt.clf()

i=2
c=1

if i%4==0:
    c+=1
sns.set(rc={ 'figure.figsize':(1,1), 'xtick.top' : False, 'figure.dpi':600})
sns.set_theme(style="whitegrid", rc={'grid.linewidth':0.2})
plt.ticklabel_format(style='plain', axis='y')
sns.set_palette(palette)
sns.set_context("paper")
sns.despine()
s=sns.lineplot(data=long_set[long_set['Case_Length']=='CL_{}'.format(i)], 
               x='Prefix Length', y="ROC", hue='Approach',style='Approach',legend="full",
               dashes=False, markers=True, markevery=c,
               linewidth= 0.5,markersize=2.5,markeredgewidth=0.1, errorbar=None)


s.set(ylim=(0.4,1))
s.set(xlim=(1,i))
s.axes.set_title("Case Length {}".format(i),fontsize=8)
s.set_xlabel(None)
s.set_ylabel(None)
s.tick_params(labelsize=5)
s.set_xticks(range(1,i+1,c))
s.spines['left'].set_linewidth(0.3)
s.spines['bottom'].set_linewidth(0.3)

sns.move_legend(s, "lower center",bbox_to_anchor=(0, -1), ncol=9, title='Approach', frameon=False,fontsize=6,title_fontsize=7)

plt.savefig('grid_CL_Axis.svg'.format(i),bbox_inches='tight')


num_alg=len(approach_list)
num_runs=len(cl_list)

score_name=['Mean Value depending on case length']

friedmann_df=pd.DataFrame(data=None, index=approach_list, columns=cl_list)
i=0
for app in approach_list:
    friedmann_df.loc[app,:]=np.around(display_df_cl.loc[app,:].astype(float), decimals=4)

friedmann_rank=pd.DataFrame(data=None, index=approach_list)



for cl in cl_list:
    friedmann_temp=friedmann_df.loc[:,cl].sort_values( kind='mergesort', ascending=False).reset_index()
    friedmann_temp['rank_score_%s'%cl]=range(1,num_alg+1)
    friedmann_rank=friedmann_rank.join(friedmann_temp[['index','rank_score_%s'%cl]].set_index('index'))
    
    if len(set(friedmann_df.loc[:,cl]))==1:
        friedmann_rank['rank_score_%s'%cl]=5
    elif len(set(friedmann_df.loc[:,cl]))<9:
        temp_df=friedmann_df.loc[:,cl].value_counts().reset_index().sort_values(by='index', ascending=False).reset_index(drop=True)
        pre_count=0
        for i in range(len(temp_df)):
            value=temp_df.iloc[i,0]
            count=temp_df.iloc[i,1]
            rank=np.cumsum(range(count+1))[-1]/count+pre_count
            temp_alg_list=list(friedmann_df[friedmann_df.loc[:,cl]==value].loc[:,cl].reset_index()['index'])
            for alg in temp_alg_list:
                friedmann_rank.loc[alg,'rank_score_%s'%cl]=rank
                
            pre_count+=count


friedmann_rank_average=pd.DataFrame(data=None, index=approach_list, columns=['average_rank'])

for alg in approach_list:
    friedmann_rank_average.loc[alg,'average_rank']=np.mean(friedmann_rank.loc[alg])

nemenyi = np.array([friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5],friedmann_df.iloc[6],friedmann_df.iloc[7],friedmann_df.iloc[8]])

friedmann_stats=stats.friedmanchisquare(friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5],friedmann_df.iloc[6],friedmann_df.iloc[7],friedmann_df.iloc[8])
CD=2.728*(((num_alg*(num_alg+1))/(6*num_runs))**(1/2))

print(friedmann_stats)
print(CD)

sns.set(font_scale=1.2,rc={"lines.linewidth": 2, 'figure.figsize':(10,10), 'figure.dpi':100, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":9  })
sns.set_style("white")

sorted_friedmann=friedmann_rank_average.sort_values(by='average_rank').reset_index()

cd = CD


rank_1_value=sorted_friedmann.iloc[0,1]
rank_2_value=sorted_friedmann.iloc[1,1]
rank_3_value=sorted_friedmann.iloc[2,1]
rank_4_value=sorted_friedmann.iloc[3,1]
rank_5_value=sorted_friedmann.iloc[4,1]
rank_6_value=sorted_friedmann.iloc[5,1]
rank_7_value=sorted_friedmann.iloc[6,1]
rank_8_value=sorted_friedmann.iloc[7,1]
rank_9_value=sorted_friedmann.iloc[8,1]



rank_1=sorted_friedmann.iloc[0,0]
rank_2=sorted_friedmann.iloc[1,0]
rank_3=sorted_friedmann.iloc[2,0]
rank_4=sorted_friedmann.iloc[3,0]
rank_5=sorted_friedmann.iloc[4,0]
rank_6=sorted_friedmann.iloc[5,0]
rank_7=sorted_friedmann.iloc[6,0]
rank_8=sorted_friedmann.iloc[7,0]
rank_9=sorted_friedmann.iloc[8,0]


value_list=[rank_1_value,rank_2_value,rank_3_value,rank_4_value,rank_5_value,rank_6_value,rank_7_value,rank_8_value,rank_9_value]
rank_list=[rank_1,rank_2,rank_3,rank_4,rank_5,rank_6,rank_7,rank_8,rank_9]

limits=(1,9)

fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(left=0.2, right=0.8)


ax.set_xlim(limits)
ax.set_ylim(0,1)
ax.spines['top'].set_position(('axes', 0.6))
#ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_visible(False)
for pos in ["bottom", "left", "right"]:
    ax.spines[pos].set_visible(False)

ax.plot([limits[0],limits[0]+cd], [.8,.8], color="k")
ax.plot([limits[0],limits[0]], [.8-0.03,.8+0.03], color="k")
ax.plot([limits[0]+cd,limits[0]+cd], [.8-0.03,.8+0.03], color="k") 
ax.text(limits[0]+cd/2., 0.82, "CD", ha="center", va="bottom") 



bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.0)
arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90", color='black')
kw = dict(xycoords='data',textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, va="center")
ax.annotate(rank_1, xy=(rank_1_value, 0.6), xytext=(0,0.55),ha="right",  **kw)
ax.annotate(rank_2, xy=(rank_2_value, 0.6), xytext=(0,0.45),ha="right",  **kw)
ax.annotate(rank_3, xy=(rank_3_value, 0.6), xytext=(0,0.35),ha="right",  **kw)
ax.annotate(rank_4, xy=(rank_4_value, 0.6), xytext=(0,0.25),ha="right",  **kw)
ax.annotate(rank_5, xy=(rank_5_value, 0.6), xytext=(0,0.15),ha="right",  **kw)
ax.annotate(rank_6, xy=(rank_6_value, 0.6), xytext=(1.,0.15),ha="left",  **kw)
ax.annotate(rank_7, xy=(rank_7_value, 0.6), xytext=(1.,0.25),ha="left",  **kw)
ax.annotate(rank_8, xy=(rank_8_value, 0.6), xytext=(1.,0.35),ha="left",  **kw)
ax.annotate(rank_9, xy=(rank_9_value, 0.6), xytext=(1.,0.45),ha="left",  **kw)



k=0
row=0
for i in range(num_alg):
    value_idx=np.where((np.array(value_list[i:])<(value_list[i]+CD)))
    if ((len(value_idx[0]) > 0) & (k==0)):
        k=value_idx[0][-1]
        alg_1=value_list[i]
        alg_2=value_list[i+k]

        ax.plot([alg_1,alg_2],[0.55-(0.05*row),0.55-(0.05*row)], color="k", lw=3)
        row+=1
    k-=1
    if k<0:
        k=0
num_alg=len(approach_list)
num_runs=len(pp_list)

score_name=['Mean Value by prediction point , unweighted']

friedmann_df=pd.DataFrame(data=None, index=approach_list, columns=pp_list)
i=0
for app in approach_list:
    friedmann_df.loc[app,:]=np.around(display_df_pp.loc[app,:].astype(float), decimals=4)

friedmann_rank=pd.DataFrame(data=None, index=approach_list)



for cl in pp_list:
    friedmann_temp=friedmann_df.loc[:,cl].sort_values( kind='mergesort', ascending=False).reset_index()
    friedmann_temp['rank_score_%s'%cl]=range(1,num_alg+1)
    friedmann_rank=friedmann_rank.join(friedmann_temp[['index','rank_score_%s'%cl]].set_index('index'))
    
    if len(set(friedmann_df.loc[:,cl]))==1:
        friedmann_rank['rank_score_%s'%cl]=5
    elif len(set(friedmann_df.loc[:,cl]))<9:
        temp_df=friedmann_df.loc[:,cl].value_counts().reset_index().sort_values(by='index', ascending=False).reset_index(drop=True)
        pre_count=0
        for i in range(len(temp_df)):
            value=temp_df.iloc[i,0]
            count=temp_df.iloc[i,1]
            rank=np.cumsum(range(count+1))[-1]/count+pre_count
            temp_alg_list=list(friedmann_df[friedmann_df.loc[:,cl]==value].loc[:,cl].reset_index()['index'])
            for alg in temp_alg_list:
                friedmann_rank.loc[alg,'rank_score_%s'%cl]=rank
                
            pre_count+=count


friedmann_rank_average=pd.DataFrame(data=None, index=approach_list, columns=['average_rank'])

for alg in approach_list:
    friedmann_rank_average.loc[alg,'average_rank']=np.mean(friedmann_rank.loc[alg])

nemenyi = np.array([friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5],friedmann_df.iloc[6],friedmann_df.iloc[7],friedmann_df.iloc[8]])

friedmann_stats=stats.friedmanchisquare(friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5],friedmann_df.iloc[6],friedmann_df.iloc[7],friedmann_df.iloc[8])
CD=2.728*(((num_alg*(num_alg+1))/(6*num_runs))**(1/2))

print(friedmann_stats)
print(CD)

sns.set(font_scale=1.2,rc={"lines.linewidth": 2, 'figure.figsize':(10,10), 'figure.dpi':100, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":9  })
sns.set_style("white")

sorted_friedmann=friedmann_rank_average.sort_values(by='average_rank').reset_index()

cd = CD


rank_1_value=sorted_friedmann.iloc[0,1]
rank_2_value=sorted_friedmann.iloc[1,1]
rank_3_value=sorted_friedmann.iloc[2,1]
rank_4_value=sorted_friedmann.iloc[3,1]
rank_5_value=sorted_friedmann.iloc[4,1]
rank_6_value=sorted_friedmann.iloc[5,1]
rank_7_value=sorted_friedmann.iloc[6,1]
rank_8_value=sorted_friedmann.iloc[7,1]
rank_9_value=sorted_friedmann.iloc[8,1]



rank_1=sorted_friedmann.iloc[0,0]
rank_2=sorted_friedmann.iloc[1,0]
rank_3=sorted_friedmann.iloc[2,0]
rank_4=sorted_friedmann.iloc[3,0]
rank_5=sorted_friedmann.iloc[4,0]
rank_6=sorted_friedmann.iloc[5,0]
rank_7=sorted_friedmann.iloc[6,0]
rank_8=sorted_friedmann.iloc[7,0]
rank_9=sorted_friedmann.iloc[8,0]


value_list=[rank_1_value,rank_2_value,rank_3_value,rank_4_value,rank_5_value,rank_6_value,rank_7_value,rank_8_value,rank_9_value]
rank_list=[rank_1,rank_2,rank_3,rank_4,rank_5,rank_6,rank_7,rank_8,rank_9]

limits=(1,9)

fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(left=0.2, right=0.8)


ax.set_xlim(limits)
ax.set_ylim(0,1)
ax.spines['top'].set_position(('axes', 0.6))
#ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_visible(False)
for pos in ["bottom", "left", "right"]:
    ax.spines[pos].set_visible(False)

ax.plot([limits[0],limits[0]+cd], [.8,.8], color="k")
ax.plot([limits[0],limits[0]], [.8-0.03,.8+0.03], color="k")
ax.plot([limits[0]+cd,limits[0]+cd], [.8-0.03,.8+0.03], color="k") 
ax.text(limits[0]+cd/2., 0.82, "CD", ha="center", va="bottom") 



bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.0)
arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90", color='black')
kw = dict(xycoords='data',textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, va="center")
ax.annotate(rank_1, xy=(rank_1_value, 0.6), xytext=(0,0.55),ha="right",  **kw)
ax.annotate(rank_2, xy=(rank_2_value, 0.6), xytext=(0,0.45),ha="right",  **kw)
ax.annotate(rank_3, xy=(rank_3_value, 0.6), xytext=(0,0.35),ha="right",  **kw)
ax.annotate(rank_4, xy=(rank_4_value, 0.6), xytext=(0,0.25),ha="right",  **kw)
ax.annotate(rank_5, xy=(rank_5_value, 0.6), xytext=(0,0.15),ha="right",  **kw)
ax.annotate(rank_6, xy=(rank_6_value, 0.6), xytext=(1.,0.15),ha="left",  **kw)
ax.annotate(rank_7, xy=(rank_7_value, 0.6), xytext=(1.,0.25),ha="left",  **kw)
ax.annotate(rank_8, xy=(rank_8_value, 0.6), xytext=(1.,0.35),ha="left",  **kw)
ax.annotate(rank_9, xy=(rank_9_value, 0.6), xytext=(1.,0.45),ha="left",  **kw)



k=0
row=0
for i in range(num_alg):
    value_idx=np.where((np.array(value_list[i:])<(value_list[i]+CD)))
    if ((len(value_idx[0]) > 0) & (k==0)):
        k=value_idx[0][-1]
        alg_1=value_list[i]
        alg_2=value_list[i+k]

        ax.plot([alg_1,alg_2],[0.55-(0.05*row),0.55-(0.05*row)], color="k", lw=3)
        row+=1
    k-=1
    if k<0:
        k=0
num_alg=len(approach_list)
num_runs=len(pp_list)

score_name=['Mean Value by prediction point, weighted']

friedmann_df=pd.DataFrame(data=None, index=approach_list, columns=pp_list)
i=0
for app in approach_list:
    friedmann_df.loc[app,:]=np.around(display_df_pp_w.loc[app,:].astype(float), decimals=4)

friedmann_rank=pd.DataFrame(data=None, index=approach_list)



for cl in pp_list:
    friedmann_temp=friedmann_df.loc[:,cl].sort_values( kind='mergesort', ascending=False).reset_index()
    friedmann_temp['rank_score_%s'%cl]=range(1,num_alg+1)
    friedmann_rank=friedmann_rank.join(friedmann_temp[['index','rank_score_%s'%cl]].set_index('index'))
    
    if len(set(friedmann_df.loc[:,cl]))==1:
        friedmann_rank['rank_score_%s'%cl]=5
    elif len(set(friedmann_df.loc[:,cl]))<9:
        temp_df=friedmann_df.loc[:,cl].value_counts().reset_index().sort_values(by='index', ascending=False).reset_index(drop=True)
        pre_count=0
        for i in range(len(temp_df)):
            value=temp_df.iloc[i,0]
            count=temp_df.iloc[i,1]
            rank=np.cumsum(range(count+1))[-1]/count+pre_count
            temp_alg_list=list(friedmann_df[friedmann_df.loc[:,cl]==value].loc[:,cl].reset_index()['index'])
            for alg in temp_alg_list:
                friedmann_rank.loc[alg,'rank_score_%s'%cl]=rank
                
            pre_count+=count


friedmann_rank_average=pd.DataFrame(data=None, index=approach_list, columns=['average_rank'])

for alg in approach_list:
    friedmann_rank_average.loc[alg,'average_rank']=np.mean(friedmann_rank.loc[alg])

nemenyi = np.array([friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5],friedmann_df.iloc[6],friedmann_df.iloc[7],friedmann_df.iloc[8]])

friedmann_stats=stats.friedmanchisquare(friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5],friedmann_df.iloc[6],friedmann_df.iloc[7],friedmann_df.iloc[8])
CD=3.102*(((num_alg*(num_alg+1))/(6*num_runs))**(1/2))

print(friedmann_stats)
print(CD)

sns.set(font_scale=1.2,rc={"lines.linewidth": 2, 'figure.figsize':(10,10), 'figure.dpi':100, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":9  })
sns.set_style("white")

sorted_friedmann=friedmann_rank_average.sort_values(by='average_rank').reset_index()

cd = CD


rank_1_value=sorted_friedmann.iloc[0,1]
rank_2_value=sorted_friedmann.iloc[1,1]
rank_3_value=sorted_friedmann.iloc[2,1]
rank_4_value=sorted_friedmann.iloc[3,1]
rank_5_value=sorted_friedmann.iloc[4,1]
rank_6_value=sorted_friedmann.iloc[5,1]
rank_7_value=sorted_friedmann.iloc[6,1]
rank_8_value=sorted_friedmann.iloc[7,1]
rank_9_value=sorted_friedmann.iloc[8,1]



rank_1=sorted_friedmann.iloc[0,0]
rank_2=sorted_friedmann.iloc[1,0]
rank_3=sorted_friedmann.iloc[2,0]
rank_4=sorted_friedmann.iloc[3,0]
rank_5=sorted_friedmann.iloc[4,0]
rank_6=sorted_friedmann.iloc[5,0]
rank_7=sorted_friedmann.iloc[6,0]
rank_8=sorted_friedmann.iloc[7,0]
rank_9=sorted_friedmann.iloc[8,0]


value_list=[rank_1_value,rank_2_value,rank_3_value,rank_4_value,rank_5_value,rank_6_value,rank_7_value,rank_8_value,rank_9_value]
rank_list=[rank_1,rank_2,rank_3,rank_4,rank_5,rank_6,rank_7,rank_8,rank_9]

limits=(1,9)

fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(left=0.2, right=0.8)


ax.set_xlim(limits)
ax.set_ylim(0,1)
ax.spines['top'].set_position(('axes', 0.6))
#ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_visible(False)
for pos in ["bottom", "left", "right"]:
    ax.spines[pos].set_visible(False)

ax.plot([limits[0],limits[0]+cd], [.8,.8], color="k")
ax.plot([limits[0],limits[0]], [.8-0.03,.8+0.03], color="k")
ax.plot([limits[0]+cd,limits[0]+cd], [.8-0.03,.8+0.03], color="k") 
ax.text(limits[0]+cd/2., 0.82, "CD", ha="center", va="bottom") 



bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.0)
arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90", color='black')
kw = dict(xycoords='data',textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, va="center")
ax.annotate(rank_1, xy=(rank_1_value, 0.6), xytext=(0,0.55),ha="right",  **kw)
ax.annotate(rank_2, xy=(rank_2_value, 0.6), xytext=(0,0.45),ha="right",  **kw)
ax.annotate(rank_3, xy=(rank_3_value, 0.6), xytext=(0,0.35),ha="right",  **kw)
ax.annotate(rank_4, xy=(rank_4_value, 0.6), xytext=(0,0.25),ha="right",  **kw)
ax.annotate(rank_5, xy=(rank_5_value, 0.6), xytext=(0,0.15),ha="right",  **kw)
ax.annotate(rank_6, xy=(rank_6_value, 0.6), xytext=(1.,0.15),ha="left",  **kw)
ax.annotate(rank_7, xy=(rank_7_value, 0.6), xytext=(1.,0.25),ha="left",  **kw)
ax.annotate(rank_8, xy=(rank_8_value, 0.6), xytext=(1.,0.35),ha="left",  **kw)
ax.annotate(rank_9, xy=(rank_9_value, 0.6), xytext=(1.,0.45),ha="left",  **kw)



k=0
row=0
for i in range(num_alg):
    value_idx=np.where((np.array(value_list[i:])<(value_list[i]+CD)))
    if ((len(value_idx[0]) > 0) & (k==0)):
        k=value_idx[0][-1]
        alg_1=value_list[i]
        alg_2=value_list[i+k]

        ax.plot([alg_1,alg_2],[0.55-(0.05*row),0.55-(0.05*row)], color="k", lw=3)
        row+=1
    k-=1
    if k<0:
        k=0

#Temporal Stability RF_AAGG


app_list_2=['_agg_base','_agg','_plus','_idx','_agg_base','_agg','_plus','_idx','_lstm']

collect_TS_df=pd.DataFrame(data=None, columns=['case length'])

for folder,app,colour in zip(app_list,app_list_2,palette):
    full_df=pd.DataFrame(data=None, columns=['case_length'])
    print(app)
    print(folder)
    for case_len in range (2,30):
        if ((app=='_lstm') & (case_len==26)):
            break
        else:
            for pred_point in range (1,case_len+1): 

                if pred_point==1:
                    file_read='Results/Results_Probs/{}/probs{}_{}_{}.csv'.format(folder,app,pred_point,case_len)
                    temp_df=pd.read_csv(file_read, header=None)
                    temp_df.columns=['pred_point_{}'.format(pred_point)]
                else:
                    file_read='Results/Results_Probs/{}/probs{}_{}_{}.csv'.format(folder,app,pred_point,case_len)
                    temp_df['pred_point_{}'.format(pred_point)]=pd.read_csv(file_read, header=None)

            temp_df['case_length']=case_len
            full_df=pd.concat([full_df,temp_df])
            
            del temp_df

    case_length_val=[]
    TS_val=[]
        
    for Ti in range(2,30):
        if ((app=='_lstm') & (Ti==26)):
            break
        else:
            temp_df=full_df[full_df['case_length']==Ti]
            n=len(temp_df)
            summed_diff=0
            for t in range(2,Ti+1):
                summed_diff+=sum(abs(temp_df['pred_point_{}'.format(t)]-temp_df['pred_point_{}'.format(t-1)]))
                #print(summed_diff)

            TS=1-(1/n)*(1/(Ti-1))*summed_diff
            case_length_val.append(Ti)
            TS_val.append(TS)

    TS_df=pd.DataFrame(data=TS_val, index=case_length_val, columns=['TS'])
    TS_df.reset_index(inplace=True)
    
    del TS_val, case_length_val, summed_diff
    
    sns.set(rc={'figure.figsize':(1.5,1.5), 'xtick.top' : False, 'figure.dpi':1200})
    sns.set_theme(style="whitegrid", rc={'grid.linewidth':0.2})
    plt.ticklabel_format(style='plain', axis='y')
    sns.set_context("paper")

    s=sns.barplot(data=TS_df, x='index', y='TS', color=colour, width=1)
    s.axes.set_title("{}".format(folder),fontsize=8)
    s.set(ylim=(0.5,1))   
    s.set_xlabel(None)
    s.set_ylabel(None)
    s.set(xticklabels=[])
    s.set(yticklabels=[])
    s.spines['left'].set_linewidth(0.5)
    s.spines['bottom'].set_linewidth(0.5)
    s.spines['top'].set_linewidth(0.5)
    s.spines['right'].set_linewidth(0.5)
    
    collect_TS_df[folder]=TS_df['TS']
    collect_TS_df['case length']=range(2,30)
    plt.clf()

collect_TS_df.set_index('case length', inplace=True)
weighted_TS_df=collect_TS_df.copy(deep=True)
weighted_sum_TS_df=collect_TS_df.copy(deep=True).head(1)
total=111340

for i in range(2,30):
    weighted_TS_df.loc[i]=collect_TS_df.loc[i]*case_length_no[i]/total
    
for col in weighted_TS_df.columns:
    weighted_sum_TS_df.loc[:,col]=weighted_TS_df.loc[:,col].sum()
    
weighted_sum_TS_df.reset_index(drop=True, inplace=True)
weighted_sum_TS_df.sort_values(by=0,axis=1,ascending=False, inplace=True)


sns.set(rc={'figure.figsize':(4,1.5), 'xtick.top' : False, 'figure.dpi':1200})
sns.set_theme(style="whitegrid", rc={'grid.linewidth':0.2})
plt.ticklabel_format(style='plain', axis='y')
sns.set_palette(palette)
sns.set_context("paper")

s=sns.barplot(data=weighted_sum_TS_df, palette=palette2, width=1)


s.set(ylim=(0.5,1))
s.set_xlabel("Approach",fontsize=8)
s.set_ylabel("Temporal Stability",fontsize=8)
s.tick_params(labelsize=5)
s.spines['left'].set_linewidth(0.5)
s.spines['bottom'].set_linewidth(0.5)
s.spines['top'].set_linewidth(0.5)
s.spines['right'].set_linewidth(0.5)
plt.xticks(rotation=90)
palette2 = ['#33a02c','#b2df8a','#1f78b4','#a6cee3','#e31a1c','#cab2d6','#6a3d9a','#fb9a99','#fdbf6f']#'#ffff99','#ff7f00'

approach_list=app_list
pp_list=list(range(2,30))

num_alg=len(approach_list)
num_runs=len(pp_list)

score_name=['Temporal Stability']

friedmann_df=pd.DataFrame(data=None, index=approach_list, columns=pp_list)
i=0
for app in approach_list:
    friedmann_df.loc[app,:]=np.around(collect_TS_df.loc[:,app].astype(float), decimals=4)

friedmann_rank=pd.DataFrame(data=None, index=approach_list)



for cl in pp_list:
    friedmann_temp=friedmann_df.loc[:,cl].sort_values( kind='mergesort', ascending=False).reset_index()
    friedmann_temp['rank_score_%s'%cl]=range(1,num_alg+1)
    friedmann_rank=friedmann_rank.join(friedmann_temp[['index','rank_score_%s'%cl]].set_index('index'))
    
    if len(set(friedmann_df.loc[:,cl]))==1:
        friedmann_rank['rank_score_%s'%cl]=5
    elif len(set(friedmann_df.loc[:,cl]))<9:
        temp_df=friedmann_df.loc[:,cl].value_counts().reset_index().sort_values(by='index',ascending=False).reset_index(drop=True)
        pre_count=0
        for i in range(len(temp_df)):
            value=temp_df.iloc[i,0]
            count=temp_df.iloc[i,1]
            rank=np.cumsum(range(count+1))[-1]/count+pre_count
            temp_alg_list=list(friedmann_df[friedmann_df.loc[:,cl]==value].loc[:,cl].reset_index()['index'])
            for alg in temp_alg_list:
                friedmann_rank.loc[alg,'rank_score_%s'%cl]=rank
                
            pre_count+=count


friedmann_rank_average=pd.DataFrame(data=None, index=approach_list, columns=['average_rank'])

for alg in approach_list:
    friedmann_rank_average.loc[alg,'average_rank']=np.mean(friedmann_rank.loc[alg])

nemenyi = np.array([friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5],friedmann_df.iloc[6],friedmann_df.iloc[7],friedmann_df.iloc[8]])

friedmann_stats=stats.friedmanchisquare(friedmann_df.iloc[0],friedmann_df.iloc[1],friedmann_df.iloc[2],friedmann_df.iloc[3],friedmann_df.iloc[4],friedmann_df.iloc[5],friedmann_df.iloc[6],friedmann_df.iloc[7],friedmann_df.iloc[8])
CD=2.728*(((num_alg*(num_alg+1))/(6*num_runs))**(1/2))

print(friedmann_stats)
print(CD)

sns.set(font_scale=1.2,rc={"lines.linewidth": 2, 'figure.figsize':(10,10), 'figure.dpi':100, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":9  })
sns.set_style("white")

sorted_friedmann=friedmann_rank_average.sort_values(by='average_rank').reset_index()

cd = CD


rank_1_value=sorted_friedmann.iloc[0,1]
rank_2_value=sorted_friedmann.iloc[1,1]
rank_3_value=sorted_friedmann.iloc[2,1]
rank_4_value=sorted_friedmann.iloc[3,1]
rank_5_value=sorted_friedmann.iloc[4,1]
rank_6_value=sorted_friedmann.iloc[5,1]
rank_7_value=sorted_friedmann.iloc[6,1]
rank_8_value=sorted_friedmann.iloc[7,1]
rank_9_value=sorted_friedmann.iloc[8,1]



rank_1=sorted_friedmann.iloc[0,0]
rank_2=sorted_friedmann.iloc[1,0]
rank_3=sorted_friedmann.iloc[2,0]
rank_4=sorted_friedmann.iloc[3,0]
rank_5=sorted_friedmann.iloc[4,0]
rank_6=sorted_friedmann.iloc[5,0]
rank_7=sorted_friedmann.iloc[6,0]
rank_8=sorted_friedmann.iloc[7,0]
rank_9=sorted_friedmann.iloc[8,0]


value_list=[rank_1_value,rank_2_value,rank_3_value,rank_4_value,rank_5_value,rank_6_value,rank_7_value,rank_8_value,rank_9_value]
rank_list=[rank_1,rank_2,rank_3,rank_4,rank_5,rank_6,rank_7,rank_8,rank_9]

limits=(1,9)

fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(left=0.2, right=0.8)


ax.set_xlim(limits)
ax.set_ylim(0,1)
ax.spines['top'].set_position(('axes', 0.6))
#ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_visible(False)
for pos in ["bottom", "left", "right"]:
    ax.spines[pos].set_visible(False)

ax.plot([limits[0],limits[0]+cd], [.8,.8], color="k")
ax.plot([limits[0],limits[0]], [.8-0.03,.8+0.03], color="k")
ax.plot([limits[0]+cd,limits[0]+cd], [.8-0.03,.8+0.03], color="k") 
ax.text(limits[0]+cd/2., 0.82, "CD", ha="center", va="bottom") 



bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.0)
arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90", color='black')
kw = dict(xycoords='data',textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, va="center")
ax.annotate(rank_1, xy=(rank_1_value, 0.6), xytext=(0,0.55),ha="right",  **kw)
ax.annotate(rank_2, xy=(rank_2_value, 0.6), xytext=(0,0.45),ha="right",  **kw)
ax.annotate(rank_3, xy=(rank_3_value, 0.6), xytext=(0,0.35),ha="right",  **kw)
ax.annotate(rank_4, xy=(rank_4_value, 0.6), xytext=(0,0.25),ha="right",  **kw)
ax.annotate(rank_5, xy=(rank_5_value, 0.6), xytext=(0,0.15),ha="right",  **kw)
ax.annotate(rank_6, xy=(rank_6_value, 0.6), xytext=(1.,0.15),ha="left",  **kw)
ax.annotate(rank_7, xy=(rank_7_value, 0.6), xytext=(1.,0.25),ha="left",  **kw)
ax.annotate(rank_8, xy=(rank_8_value, 0.6), xytext=(1.,0.35),ha="left",  **kw)
ax.annotate(rank_9, xy=(rank_9_value, 0.6), xytext=(1.,0.45),ha="left",  **kw)



k=0
row=0
for i in range(num_alg):
    value_idx=np.where((np.array(value_list[i:])<(value_list[i]+CD)))
    if ((len(value_idx[0]) > 0) & (k==0)):
        k=value_idx[0][-1]
        alg_1=value_list[i]
        alg_2=value_list[i+k]

        ax.plot([alg_1,alg_2],[0.55-(0.05*row),0.55-(0.05*row)], color="k", lw=3)
        row+=1
    k-=1
    if k<0:
        k=0

#Temporal Stability RF_AAGG
full_df=pd.DataFrame(data=None, columns=['case_length'])
folder='RF_AGG'

for case_len in range (2,30):

    for pred_point in range (1,case_len+1): 
        
        if pred_point==1:
            file_read='Results/Results_Probs/{}/probs_agg_base_{}_{}.csv'.format(folder, pred_point,case_len)
            temp_df=pd.read_csv(file_read, header=None)
            temp_df.columns=['pred_point_{}'.format(pred_point)]
        else:
            file_read='Results/Results_Probs/{}/probs_agg_base_{}_{}.csv'.format(folder, pred_point,case_len)
            temp_df['pred_point_{}'.format(pred_point)]=pd.read_csv(file_read, header=None)
            
    temp_df['case_length']=case_len
    full_df=pd.concat([full_df,temp_df])
        
tot=0
for i in range(2,30):
    tot+=len(full_df[full_df['case_length']==i])

case_length_val=[]
TS_val=[]
for Ti in range(2,30):
    temp_df=full_df[full_df['case_length']==Ti]
    n=len(temp_df)
    summed_diff=0
    for t in range(2,Ti+1):
        summed_diff+=sum(abs(temp_df['pred_point_{}'.format(t)]-temp_df['pred_point_{}'.format(t-1)]))
       
    TS=1-(1/n)*(1/(Ti-1))*summed_diff
    case_length_val.append(Ti)
    TS_val.append(TS)

TS_df=pd.DataFrame(data=TS_val, index=case_length_val, columns=['TS'])
TS_df.reset_index(inplace=True)

s=sns.barplot(data=TS_df, x='index', y='TS', palette='dark:b')
s.set(ylim=(0.5,1))
s.set(xlabel='case length')
plt.savefig('Temporal_stability_{}.jpg'.format(folder), bbox_inches='tight',dpi=300)


## Temporal Stability LSTM Network
full_df=pd.DataFrame(data=None, columns=['case_length'])

for case_len in range (2,25):

    for pred_point in range (1,case_len+1): 
        
        if pred_point==1:
            file_read='Results/Results_Probs_LSTM/probs_lstm_{}_{}.csv'.format(pred_point,case_len)
            temp_df=pd.read_csv(file_read, header=None)
            temp_df.columns=['pred_point_{}'.format(pred_point)]
        else:
            file_read='Results/Results_Probs_LSTM/probs_lstm_{}_{}.csv'.format(pred_point,case_len)
            temp_df['pred_point_{}'.format(pred_point)]=pd.read_csv(file_read, header=None)
            
    temp_df['case_length']=case_len
    full_df=pd.concat([full_df,temp_df])
        
tot=0
for i in range(2,30):
    tot+=len(full_df[full_df['case_length']==i])
    
case_length_val=[]
TS_val=[]
for Ti in range(2,25):
    temp_df=full_df[full_df['case_length']==Ti]
    n=len(temp_df)
    summed_diff=0
    for t in range(2,Ti+1):
        summed_diff+=sum(abs(temp_df['pred_point_{}'.format(t)]-temp_df['pred_point_{}'.format(t-1)]))        
    TS=1-(1/n)*(1/(Ti-1))*summed_diff
    case_length_val.append(Ti)
    TS_val.append(TS)

TS_df=pd.DataFrame(data=TS_val, index=case_length_val, columns=['TS'])
TS_df.reset_index(inplace=True)

s=sns.barplot(data=TS_df, x='index', y='TS', palette='dark:b')
s.set(ylim=(0.5,1))
s.set(xlabel='case length')

file_read='Results/probs_full.csv'.format(pred_point,case_len)
full_df=pd.read_csv(file_read, header=None)

columns=['case_length']
for i in range(1,30):
    columns.append('pred_point_{}'.format(i))
    
full_df.columns=columns

case_length_val=[]
TS_val=[]
for Ti in range(2,30):
    temp_df=full_df[full_df['case_length']==Ti]
    n=len(temp_df)
    summed_diff=0
    for t in range(2,Ti+1):
        summed_diff+=sum(abs(temp_df['pred_point_{}'.format(t)]-temp_df['pred_point_{}'.format(t-1)]))
       
    TS=1-(1/n)*(1/(Ti-1))*summed_diff
    print(TS, Ti)
    case_length_val.append(Ti)
    TS_val.append(TS)

TS_df=pd.DataFrame(data=TS_val, index=case_length_val, columns=['TS'])
TS_df.reset_index(inplace=True)


s=sns.barplot(data=TS_df, x='index', y='TS', palette='dark:b')
s.set(ylim=(0.9,1))
s.set(xlabel='case length')


for Ti in range(2,30):
    temp_df=full_df[full_df['case_length']==Ti]
    n=len(temp_df)
    summed_diff=0
    summed_diff=sum(abs(temp_df['pred_point_{}'.format(Ti)]-temp_df['pred_point_1'.format(t-1)]))
      
    TS=1-(1/n)*(1/(2-1))*summed_diff
    print(TS, Ti)



for Ti in range(2,30):
    temp_df=full_df[full_df['case_length']>=Ti]
    n=len(temp_df)
    summed_diff=0
    summed_diff=sum(abs(temp_df['pred_point_{}'.format(Ti)]-temp_df['pred_point_1'.format(t-1)]))
        
    TS=1-(1/n)*(1/(2-1))*summed_diff
    print(TS, Ti)


full_df[full_df['case_length']==25].reset_index(drop=True)

display_df=full_df[full_df['case_length']==10].reset_index(drop=True)
display_df=display_df.loc[[1,5,10,50,100,150]].drop('case_length', axis=1)
display_df.reset_index(inplace=True,drop=True)
display_df=display_df.loc[:,:'pred_point_10']

sns.set(font_scale=0.5,rc={'figure.figsize':((5),(5)), "lines.linewidth": 2, 'figure.dpi':600, 'xtick.top' : False, 'markers.fillstyle': 'full', "lines.markersize":4,'lines.markeredgewidth':0 })
sns.set_style("whitegrid")

s=sns.lineplot(data=display_df.T,legend='brief', palette='viridis',ci=None)
plt.legend(bbox_to_anchor=(1.5, 1), loc='upper right', borderaxespad=0)
plt.xticks(rotation=90)
s.set(ylabel='Prediction score')
s.set(xlabel='Prediction Point')
s.set(ylim=(0,1))
