import os
import pandas as pd
import numpy as np
import arviz as az
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, Any

def poly_model(time:np.ndarray, n:int=1) -> np.ndarray:
    return np.power(time, 1/n)

def log_model(time:np.ndarray) -> np.ndarray:
    return np.log(time)

def inv_log_model(time:np.ndarray) -> np.ndarray:
    return 1/time

def calculate_BIC(model:pm.Model, n)->Tuple[np.float64, np.float64]:
    MAP = pm.find_MAP(model = model)
    logP = -model.logp(MAP)
    k = 0.5 * len(list(MAP.keys()))
    return k*np.log(n) + 2*logP, logP

def create_dirs(data_path:str):
    dirs_to_build = ['summaries', 'tracePlots', 'ppcPlots', 'selection_metrics']
    
    try:
        data_path = data_path.replace('.csv', '')
        data_path = os.path.join('output_info', data_path)
        os.mkdir(data_path)
    except FileExistsError:
        pass
    
    try:
        for dir_ in dirs_to_build:
            os.mkdir(os.path.join(data_path, dir_))
    except FileExistsError:
        pass

class dataAnalizer():
    def __init__(self, folder_path:str, filename:str) -> None:
        
        self.model_mapping: Dict[str, Tuple[Callable, Callable]] = {
                            'linear_model': (lambda x: poly_model(x, n=1), lambda x: poly_model(x, n=1)),
                            'cuadratic_model': (lambda x: poly_model(x, n=2),lambda x: poly_model(x, n=1)),
                            'cubic_model': (lambda x: poly_model(x, n=3),lambda x: poly_model(x, n=1)),
                            'cuartic_model': (lambda x: poly_model(x, n=4),lambda x: poly_model(x, n=1)),
                            'log_model': (log_model, poly_model),
                            'inv_log_model': (log_model, inv_log_model)
                        }
        filePATH = os.path.join(folder_path,filename)
        print('--//--'*20)
        print(f'Starting analysis for file... {filePATH}')
        
        self.data:pd.DataFrame = pd.read_csv(filePATH)
        self.keys = list(self.data.keys())
        
        self.path = filename.replace('.csv', '')
    
    def sanitize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        time = np.array(self.data[self.keys[0]].values)
        time = time[~np.isnan(time)]
        
        mass_gain = np.array(self.data[self.keys[1]].values)
        mass_gain = mass_gain[~np.isnan(mass_gain)]
        
        # TODO implement something to normalize/sanitize data other than dropping NaNs
        
        return time, mass_gain
                
    def build_model(self, time_funtion:Callable, 
        observed_values_transformation:Callable) -> Tuple[pm.Model, np.ndarray, np.ndarray]:

        time, mass_gain = self.sanitize_data()
        time = time_funtion(time)
        mass_gain = observed_values_transformation(mass_gain)

        with pm.Model() as model:

            nu = pm.HalfNormal('Nu', sigma=1)

            k = pm.Normal('k', mu=0, sigma = 10)
            C = pm.HalfCauchy('C', beta = 8)

            noise = pm.HalfCauchy('noise',beta=5)
            mean = k*time + C

            likelihood = pm.StudentT('likelihood', nu=nu, mu = mean, sigma = noise, observed = mass_gain)

        return model, time, mass_gain
    
    def perform_inference_on_models(self) -> Tuple[Dict[str, az.InferenceData], Dict[str, pm.Model]]:
        # Initial_inference process->
        trace_dict: Dict[str, az.InferenceData] = {}
        model_dict: Dict[str, pm.Model] = {}

        for model, args in self.model_mapping.items():
            print('-'*30)
            print(f'Performing Inference for: {model}')

            Inference_model, time, mass_gain = self.build_model(*args)

            with Inference_model:

                trace = pm.sample(return_inferencedata=True)
                trace_dict[model] = trace
                model_dict[model] = Inference_model
                sum = az.summary(trace)
                sum.to_csv(f'output_info/{self.path}/summaries/{model}.csv')

                # print(sum)
                plt.figure()
                az.plot_trace(trace)
                plt.savefig(f'output_info/{self.path}/tracePlots/{model}.png')
        
        return trace_dict, model_dict
    
    def posterior_predictive_check(self, trace_dict, model_dict)->None:
        time, mass_gain = self.sanitize_data()
        for model, args in self.model_mapping.items():
            
            plt.figure(figsize=(7, 7))
            plt.xlabel(f'Time transformed according to: {model}')
            plt.ylabel(f'Mass Gain transformed according to: {model}')        
            
            pm.plot_posterior_predictive_glm(trace_dict[model],
                                            eval = time,
                                            lm = lambda x,sample: args[1](sample['C'] + sample['k'] * args[0](x)), 
                                            samples=100, 
                                            label="Posterior Predictive regression lines")
            
            plt.scatter(time, mass_gain, label = 'Experimental Data')
        
            plt.legend()
            plt.savefig(f'output_info/{self.path}/ppcPlots/{model}.png')
            
    def model_selection_metric_plots(self, trace_dict, model_dict):
        time, _ = self.sanitize_data()
        
        index = ['linear_model',
                'cuadratic_model',
                'cubic_model',
                'cuartic_model',
                'log_model',
                'inv_log_model']
        
        dfLogP = pd.DataFrame(index = index, columns=['Experimental Data'])
        dfBIC = pd.DataFrame(index = index, columns=['Experimental Data'])
        dfWAIC = pd.DataFrame(index = index, columns=['Experimental Data'])
        
        dfLogP.index.name = 'model'
        dfBIC.index.name = 'model'
        dfWAIC.index.name = 'model'

        for model_name in index:
            BIC, logP = calculate_BIC(model=model_dict[model_name], n = len(time))
            dfLogP.loc[model_name,'Experimental Data'] =logP
            dfBIC.loc[model_name,'Experimental Data'] =BIC
            dfWAIC.loc[model_name,'Experimental Data'] = az.waic(trace_dict[model_name], pointwise=False).values[0]
                
        dfLogP = pd.melt(dfLogP.reset_index(), id_vars=['model'], var_name='Data', value_name='log_likelihood')
        dfBIC = pd.melt(dfBIC.reset_index(), id_vars=['model'], var_name='Data', value_name='BIC')
        dfWAIC = pd.melt(dfWAIC.reset_index(), id_vars=['model'], var_name='Data', value_name='WAIC')

        g_1 = sns.catplot(x='model', y='log_likelihood' ,data=dfLogP, kind='bar', size=6, height=6, aspect=1.5)
        g_2 = sns.catplot(x='model', y='BIC' ,data=dfBIC, kind='bar', size=6, height=6, aspect=1.5)
        g_3 = sns.catplot(x='model', y='WAIC' ,data=dfWAIC, kind='bar', size=6, height=6, aspect=1.5)

        g_1.savefig(f'output_info/{self.path}/selection_metrics/log_likelihood.png')
        g_2.savefig(f'output_info/{self.path}/selection_metrics/BIC.png')
        g_3.savefig(f'output_info/{self.path}/selection_metrics/WAIC.png')

        dfLogP.to_csv(f'output_info/{self.path}/selection_metrics/LogP.csv')
        dfBIC.to_csv(f'output_info/{self.path}/selection_metrics/BIC.csv')
        dfWAIC.to_csv(f'output_info/{self.path}/selection_metrics/WAIC.csv')
        
        plt.figure()
        compare_data = az.compare(trace_dict, ic = 'WAIC')
        az.plot_compare(compare_data)
        plt.savefig(f'output_info/{self.path}/selection_metrics/WAIC_Compare.png')