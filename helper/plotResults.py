import pandas as pd 
import numpy as np
from glob import glob
from os.path import join 
import os 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import seaborn as sns
import re
import sklearn.metrics as metrics

############################ Common for ML and CNN pipelines #################################

def plot_result(df_full, x="test_score", input_type='',
                colsgroupby='technique', colsgroupby_only=[],
                no_confs=False, sorty_by=None, join=False, beautify_io=True, legend=True):
    
    input_types_cnt = 0
    if isinstance(df_full, (list, tuple)):
        if isinstance(input_type, (list, tuple)) and len(input_type)==len(df_full):
            df_full = [dfi.assign(i_type=i_type) for dfi, i_type in zip(df_full, input_type)]
            input_types_cnt = len(input_type)
        else:
            df_full = [dfi.assign(i_type=input_type) for dfi in df_full]
        df = pd.concat(df_full)
        is_multi = True
    else:
        df = df_full.copy()
        df = df.assign(i_type=input_type)
        is_multi = False
    # drop confound-related predictions if requested
    if "i_is_conf" in df.columns and no_confs: 
        df = df[~(df.i_is_conf) & ~(df.o_is_conf)]
        
    # calculate p_values if permutation tests were run
    if f"permuted_{x}" in df.columns:
        df = calc_p_values(df, x)
    
    # if conf_ctrl is not configured then automatically determine it
    if not colsgroupby_only: 
        colsgroupby_only = df[colsgroupby].unique()
    else:
        df = df[df[colsgroupby].isin(colsgroupby_only)]
    
    # combine input and outputs as 'io'
    df["io"] = df.apply(lambda row: f"{row.inp}-{row.out}", axis=1)
    ios = df["io"].unique() 
    
    # if the x metric is not in the results, then try to compute this metric from sklearn.metrics
    if x not in df.columns:
        df = compute_metric(df, x)
    
    # make io labels latex formated
    if beautify_io:
        df.loc[:,"io"] = df.apply(remap_io, axis=1)
    df[x] = df[x].apply(lambda x: x*100)   
    
    # setup the figure properties
    sns.set(style='whitegrid', context='paper')
    fig, axes = plt.subplots(1, len(colsgroupby_only), 
                             sharex=True, sharey=True, 
                             dpi=120, figsize=(4*len(colsgroupby_only), 1+0.4*(len(ios)+input_types_cnt)))
    if not isinstance(axes, np.ndarray): axes = [axes]
    fig.gca().set_xlim([0,100])    
        
    y="io"     
    if is_multi:
        # if sample_size are different for different io (i.e. they are from different data subsets)
        # then print this info in y_label
        df = combine_io_sample_size(df)
        y="io_n" 
    y_order = get_y_order(df, y, sorty_by)
    for ((t, dfi), ax) in zip(df.groupby(colsgroupby), axes):   
        assert (len(set(y_order) - set(dfi[y].unique())))==0 or ('baseline-cb' in df[colsgroupby].unique())
        all_models = dfi["model"].unique()
        # plotting details
        palette = sns.color_palette()
        ci, dodge, scale, errwidth, capsize = 95, 0.4, 0.4, 0.9, 0.08  
        ML_MODELS = ["LR", "SVM-lin", "SVM-rbf", "GB"]
        hue_order= [m for m in ML_MODELS if m in all_models]
        # if none of the ML_MODELS are in this df then don't differentiate by model
        if len(hue_order)==0: hue_order=None
        
        ax = sns.pointplot(y=y, x=x, order=y_order,
                           hue="model", hue_order=hue_order,
                           join=join, data=dfi, ax=ax,
                           ci=ci, errwidth=errwidth, capsize=capsize,
                           dodge=dodge, scale=scale, palette=palette)# todo bugfix ci is sensible only if each trial score is statistically independant
        ax.legend_.remove()
        ax.set_title(r"{} ($n \approx{:.0f}$)".format(t.upper(), dfi["n_samples"].mean()))
        ax.set_xlabel(x)
        ax.set_ylabel("")

        # Add significance stars if permutation scores are available
        if f"permuted_{x}" in dfi.columns:
            # collect p_values as a dict of form {mean_accuracy: p_value}
            p_dict = {g[x].mean(): g["p_value"].iloc[0] 
                      for i, g in dfi.groupby(["io", "model"])}
            # filter out the error bars from all lines 
            # and store the position of their right edges to print the p_values at 
            err_bars_pos = [(l.get_xdata()[1], l.get_ydata()[0])
                            for l in ax.lines if l.get_ydata()[0]==l.get_ydata()[1]] 
            # also collect the position of the mean 'accuracy' points
            points_pos = [tuple(p_pos)  for c in ax.collections for p_pos in c.get_offsets()]
            
            for i, pos in enumerate(err_bars_pos):
                
                p_pos = points_pos[i]
                # check that the y position of both the point and error bar are the same
                assert p_pos[1] == pos[1], f"y position of mean pt ({p_pos}) and error bar ({pos}) don't match"
                # choose the error bar with the same mean accuracy point
                p =  p_dict[p_pos[0]] 
                ast = return_asterisks(p)
                
                ax.annotate(ast, np.array(pos) + (0.02, 0.06), 
                            color=palette[i//len(df[y].unique())], fontsize=5)


    if legend:
        # draw the chance line in the legend
        chance = (len(dfi[y].unique()))*[50] #todo
        for z, ch in enumerate(chance): 
            ax.axvline(x=ch, ymin=z/len(chance), ymax=(z+1)/(len(chance)), 
                        label="chance", c='gray', ls='--', lw=1.5)
            
        # add legend: add models info and chance label
        handles, legends = ax.get_legend_handles_labels()
        leg1 = fig.legend(handles[len(chance):], legends[len(chance):], 
                          title="Models",
                          bbox_to_anchor=[1.13, 0.89], loc="upper right", fancybox=True, frameon=True)
        # only choose the first 'chance' legend
        leg2 = fig.legend([handles[0]], [legends[0]], loc="lower right")
        fig.add_artist(leg1)

    fig.tight_layout()
    
    return fig
    


def remap_io(r):
    if ("i_is_conf" in r) and r.i_is_conf:
        i = "c_{{{}}}".format(r.inp.replace("_", "\_"))
    elif r.inp == 'X' and ("i_type" in r): 
        i = "X_{{{}}}".format(r.i_type.replace("_", "\_"))
    elif r.inp == 'X': 
        i = "X"
    # if none of the above then assume its a label
    else:
        i = "y_{{{}}}".format(r.inp.replace("_", "\_"))


    if ("o_is_conf" in r) and r.o_is_conf:
        o = "c_{{{}}}".format(r.out.replace("_", "\_"))
    elif r.out == 'y':
        o = "y"
    else: # then its a label
        o = "y_{{{}}}".format(r.out.replace("_", "\_"))

    new_io = r"${} \longmapsto {}$".format(i, o) 
        
    return new_io

################################  CNNpipeline    #######################################

def plot_training_curves(df_results, metrics='all'):
    
    for (model,inp,out), dfi in df_results.groupby(['model','inp','out']):
        
        # if metrics='all' then plot all the learning curve metrics present in the results file
        if isinstance(metrics,(str)) and metrics.lower()=='all':
            metrics = {col.split('_curve_')[-1] for col in dfi.columns if '_curve_' in col}

        f, axes = plt.subplots(1, len(metrics), 
                               figsize=(5*len(metrics),4),
                               sharex=True, constrained_layout=True)
        axes = axes.ravel()
        f.suptitle(r"Model: {}     Task: ${} \rightarrow {}$".format(model, inp, out), fontsize=16)

        # use different colors for metrics and different line styles for different CV trials repeats
        metrics_cmap = cm.get_cmap('tab10')
        trial_linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0,(1,5)), (0,(5,10)), 
                            (0,(3,5,1,5)), (0,(1,10)), (0,(3,5,1,5,1,5)), (0,(3,1,1,1,1,1))]

        for i, metric  in enumerate(metrics):
            ax = axes[i]
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epochs')
            ax.grid(True)    
                    
            # if there are results from multiple CV trials 
            # then show them in a single plot using different line styles            
            leg_handles, leg_names = [], []

            for j, row in dfi.iterrows():
                ls = 'solid'
                if 'trial' in row:
                    ls = trial_linestyles[row['trial']]
                    # show the linestyles used as the legend
                    leg_handles.extend([Line2D([0,1],[0,1], ls=ls, color='gray')])
                    leg_names.extend([f"CV trial {row['trial']}"])
                    
                for k, (colname, vals) in enumerate(row.filter(like='_curve_'+metric).items()):
                    if not pd.isna(vals):
                        ax.plot(eval(vals), 
                                color=metrics_cmap(k),
                                ls=ls,
                                label=colname.split('_curve_')[0])
                    
                
                leg_data = ax.legend(loc='upper left')
                    
                if 'trial' in row: 
                    ax.legend(leg_handles, leg_names, loc='lower right')
                    # re add the legend showing the colors of the data split
                    ax.add_artist(leg_data)
                    
        plt.show()


################################   MLpipeline    ######################################

# Calculate p-value
def calc_p_values(df, x="test_score", viz=False):
    
    df["p_value"] = np.nan
    grp_order = ["io", "technique", "model"]
    if 'i_type' in df and len(df['i_type'].unique())>1:
        grp_order.insert(0, 'i_type')
    groups = df.dropna(subset=[x, f"permuted_{x}"]).groupby(grp_order)   
    n_models = len(df["model"].unique())
    
    if viz:
        sns.reset_orig()
        n_rows = len(groups)//n_models
#         fig, axes = plt.subplots(n_rows, n_models, 
#                                  sharex=True, sharey=False,
#                                  figsize=(20, n_models*n_rows))
        ## paper plot jugaad
        fig, axes = plt.subplots(1, 3, 
                                 sharex=True, sharey=True,
                                 figsize=(12, 4))
        axes = np.ravel(axes)
        plt.xlim([0,1])
        
    for i, (g, rows) in enumerate(groups):
        
        p_vals = [] 
        rows = rows.filter(like=x)
        if viz:
            permuted_scores = []    
            true_scores = rows[x]
            
        for _, r in rows.iterrows():
            p_scores = np.array(eval(r[f'permuted_{x}']))
            # calc p_value = (C + 1)/(n_permutations + 1) 
            # where C is permutations whose score >= true_score 
            true_score = r[x]
            C = np.sum(p_scores >= r[x])
            pi = (C+1)/(len(p_scores)+1)
            p_vals.extend([pi])
            
            if viz: permuted_scores.extend([*p_scores])        
#         if np.std(permuted_scores)>=1:
#             print("[WARN] the p-values for {} have high variance across each test-set (trial). \
# Simply averaging the p-values across trials in such a case is not recommended.".format(g))  

        df.loc[(df[grp_order]==g).all(axis=1), ["p_value", ""]] = np.mean(p_vals)  
        
        if viz:
            ax = axes[i]
#             ax.set_title("Model={}".format(g[-1]))
#             if i%n_models == 0:
#                 ax.set_ylabel("{} with {}".format(*g[-3:-1]))
            ax.hist(permuted_scores, bins='auto', alpha=0.8)
            for true_score in true_scores:
                ax.axvline(true_score, color='r')
            # draw chance lines 
            if x in ["test_score"]:
                # chance is 14% for site prediction and 50% for y and sex predictions
                chance = 0.5 if g[0][-1]!='c' else (1/7)
                ax.axvline(chance, color='grey', linestyle="--", lw=1.5)
                ax.set_xlim(0.,1.)
            ## paper plot jugaad
            ax.set_xticklabels([str(item) for item in range(0,120, 120//len(ax.get_xticklabels()))])
            inp = "X_{{{}}}".format(["14yr", "19yr", "22yr"][i])
            out = "y_{{{binge}}}"
            ax.set_title(r"${} \longmapsto {}$".format(inp,out))
#             if i==0: ax.set_ylabel("distribution / counts")
            if i==1: ax.set_xlabel("Balanced accuracy (%)")
            if i==0:
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color="tab:blue", markerfacecolor="tab:blue", marker='o', markersize=5, lw=0),
                                Line2D([0], [0], color="tab:red", lw=1, linestyle="--")]
                ax.legend(custom_lines, ['permuted score', 'model score'], loc='lower left')
            
    sns.set(style='whitegrid')
    return df


def return_asterisks(pval):
    if pd.isna(pval):
        ast = ""
    elif pval <= 0.001:
        ast = '***'
    elif pval <= 0.01:
        ast = '**'
    elif pval <= 0.05:
        ast = '*'
    else:
        ast = 'n.s.'
    return ast


def combine_io_sample_size(df):
    df["io_n"] = ''
    # only estimate sample_size for the first technique to be plotted
    dfi = df.query("technique == '{}'".format(df.technique.unique()[0]))
    for io, dfii in dfi.groupby("io"):
        # set the same sample_size in io across all techniques to avoid different y_labels in subplots
        df.loc[(df["io"]==io), "io_n"] = "{}\n(n={:.0f})".format(io, dfii["n_samples"].mean())
    return df


def get_y_order(df, y="io", sorty_by=None):     
    if sorty_by is not None:
        dfi = df.query("technique == '{}'".format(df.technique.unique()[0]))[[y, sorty_by]] 
        y_order = dfi.groupby(y).mean().sort_values(by=sorty_by, ascending=True).index.values        
        return y_order
    else:
        return df[y].sort_values().unique()


## METRICS ##
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fp)

def compute_metric(df, metric_name):
    
    metrics_map = {"recall_score":    metrics.recall_score,
                   "precision_score": metrics.precision_score,
                   "sensitivity":     metrics.recall_score,
                   "specificity":     specificity_score,
                   "accuracy_score":  metrics.accuracy_score,
                   "f1_score":        metrics.f1_score,
                  }
    metric=None
    for k in metrics_map.keys():
        if metric_name.lower() in k:
            metric = metrics_map[k] 
            break
    if metric is None:
        raise ValueError(
            "ERROR: Invalid 'x' metric requested. Allowed metric_names are {}".format(
                metrics_map.keys()))
    
    df['y_true'] = df["test_lbls"].apply(lambda lbls: np.array(eval(lbls), dtype=int))
    df['y_pred'] = df["test_probs"].apply(lambda probs: np.argmax(np.array(eval(probs)), axis=1))
    
    df[metric_name] = df.apply(lambda x: metric(y_true=x.y_true, y_pred=x.y_pred), axis=1)
    
    return df