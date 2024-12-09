import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# **********************************************
#
#                VISUALISATION
#
# **********************************************

def corrplot(df, target_var):
    """
    > Plot correlation matrix on dataframe df with target_var
    """
    X = df.drop(target_var, axis=1)
    y = df[target_var]
    size = len(X.columns)

    num_cols, mt_width = 7, 2
    mt_heigh = 1 + size//(mt_width * num_cols)
    
    fig, ax = plt.subplots(mt_heigh, mt_width, figsize=(8, 4*mt_heigh))
    plt.subplots_adjust(wspace=1, hspace=1)
    fig.suptitle("Correlation matrix", size=15)

    for i in range(mt_heigh):
        for j in range(mt_width):
            k = mt_width * i + j
            size_k = min(num_cols, (size - k*num_cols))
            ax[i, j].plot([size_k, size_k], [0, size_k], color='black')
            ax[i, j].plot([0, size_k], [size_k, size_k], color='black')
            sns.heatmap(
                pd.concat( (X.iloc[:,k*num_cols:(k+1)*num_cols], y), axis=1 ).corr(), 
                annot=False, 
                cmap='coolwarm', 
                linewidths=0.25, 
                vmin=-1, 
                vmax=1, 
                ax=ax[i, j]
            )
    
def biScatterPlot(var_name, target_var, df):
    """
    > BiScatter plot for target_var depending on var_name
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Correlation {target_var} - {var_name} ", size=15)

    ax = axs[0]
    ax.grid()
    ax.scatter(df[var_name], df[target_var], marker='.', color='orange')
    ax.set_xlabel(var_name)
    ax.set_ylabel(target_var)

    ax = axs[1]
    ax.grid()
    ax.scatter(np.log(df[var_name] + 1), df[target_var], marker='.', color='green')
    ax.set_xlabel(f'log({var_name} + 1)')
    ax.set_ylabel(target_var)

def colorScatterPlot(var_name1, var_name2, target_var, df):
    """
    > Plot for var_name1 and var_name2 with gradient color depending on target_var
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle(f"Color plot on {target_var}", size=15)

    ax.grid()
    ax.scatter(df[var_name1], df[var_name2], c=df[target_var], marker='.', cmap='viridis', label=target_var)
    ax.set_xlabel(var_name1)
    ax.set_ylabel(var_name2)
    ax.legend()

def mapNYC(train, test, img):

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    origin = (20, 32)
    size = 300
    extent = (min(test['DO_location_lon']),
              max(test['DO_location_lon']),
              min(test['DO_location_lat']),
              max(test['DO_location_lat']))

    ax[0, 0].imshow(img[origin[0]:origin[0] + size, origin[1]:origin[1] + size], extent=extent, aspect='equal')
    ax[0, 0].scatter(train['PU_location_lon'], train['PU_location_lat'], color='black', marker='.')
    ax[0, 0].title.set_text("Train : Points de départ")

    ax[0, 1].imshow(img[origin[0]:origin[0] + size, origin[1]:origin[1] + size], extent=extent, aspect='equal')
    ax[0, 1].scatter(train['DO_location_lon'], train['DO_location_lat'], color='black', marker='.')
    ax[0, 1].title.set_text("Train : Points d'arrivée")

    ax[1, 0].imshow(img[origin[0]:origin[0] + size, origin[1]:origin[1] + size], extent=extent, aspect='equal')
    ax[1, 0].scatter(test['PU_location_lon'], test['PU_location_lat'], color='black', marker='.')
    ax[1, 0].title.set_text("Test : Points de départ")

    ax[1, 1].imshow(img[origin[0]:origin[0] + size, origin[1]:origin[1] + size], extent=extent, aspect='equal')
    ax[1, 1].scatter(test['DO_location_lon'], test['DO_location_lat'], color='black', marker='.')
    ax[1, 1].title.set_text("Test : Points d'arrivée")

# **********************************************
#
#               CROSS VALIDATION
#
# **********************************************
class GridSearchAnalysis():

    def __init__(self, model, Xtrain, Ytrain, scoring, n_folds=5, param_grid={}, refit=None):

        # metrique de réévaluation du meilleur modèle
        if refit == None : refit = list(scoring.keys())[0]

        # grid search
        grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, refit=refit, cv=n_folds).fit(Xtrain, Ytrain)
    
        self.__params = list(param_grid.keys())
        self.__metrics = list(scoring.keys())
        self.__n_folds = n_folds

        # resultats détaillés
        results = grid_search.cv_results_
    
        # affichage dans un dataframe
        self.__results = pd.DataFrame({})
        self.__results['mean_fit_time'] = results['mean_fit_time']
        self.__results['std_fit_time'] = results['std_fit_time']

        for key_param in self.__params :
            self.__results[key_param] = [results['params'][i][key_param] for i in range(len(results['params']))]
        
        for metric in self.__metrics:
            self.__results['mean_' + metric] = np.abs(results['mean_test_' + metric])
            self.__results['std_' + metric] = np.abs(results['std_test_' + metric])

        self.__results
        
    # retour des résultats
    def score_table(self, groupBy=None, synthetic=True):
        res = self.__results.copy()
        if groupBy in self.__params : 
            res = res.groupby(groupBy).mean() 
        if synthetic : 
            res['fit_time'] = [f"{res['mean_fit_time'].iloc[i]:.3f} ± {res['std_fit_time'].iloc[i]:.3f}" for i in range(len(res['mean_fit_time']))]
            res = res.drop('mean_fit_time', axis=1)
            res = res.drop('std_fit_time', axis=1)
            for metric in self.__metrics :
                res[metric] = [f"{abs(res['mean_' + metric].iloc[i]):.3f} ± {res['std_' + metric].iloc[i]:.3f}" for i in range(len(res['mean_' + metric]))]
                res = res.drop('mean_' + metric, axis=1)
                res = res.drop('std_'  + metric, axis=1)
        return res
    
    def plot_score(self, metric=None, groupBy=None):

        if metric == None : metric = self.__metrics[0]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.grid()
        res = self.score_table(groupBy=groupBy, synthetic=False)

        if groupBy == None :  
            ax.plot(res['mean_' + metric], color='red', marker='o', label='mean')
            ax.plot(res['mean_' + metric] + res['std_' + metric], linestyle='--', color='black', label='std')
            ax.plot(res['mean_' + metric] - res['std_' + metric], linestyle='--', color='black')
            ax.set_xlabel('param config id')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} depending on complexity, CV={self.__n_folds}')

        else :
            ax.plot(res.index, res['mean_' + metric], color='red', marker='o', label=f'mean-{metric}, groupByMean={groupBy}')
            ax.plot(res.index, res['mean_' + metric] + res['std_' + metric], linestyle='--', color='black', label=f'std-{metric}, groupByMean={groupBy}')
            ax.plot(res.index, res['mean_' + metric] - res['std_' + metric], linestyle='--', color='black')
            ax.set_xlabel(groupBy)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} depending on {groupBy} (mean groupBy), CV={self.__n_folds}')

        ax.legend()

    def save_score(self, filename):
        self.score_table().to_csv('GSResults/' + filename + '.csv', index=False)

# **********************************************
#
#                  SUBMISSION
#
# **********************************************

def submit_model(filename, Ypred, test):
    
    pd.DataFrame({
        'row_ID':test.index,
        'tip_amount':Ypred
    }).to_parquet('predictions/' + filename + '.parquet', index=False)

