import os
from src.data_analizer import dataAnalizer
from src.helpers import create_dirs

root, dirs, files = next(os.walk('test_data'))

for file in files[:4]:
    create_dirs(data_path=file)
    analyzer = dataAnalizer(folder_path='test_data', filename=file)
    trace_dict, model_dict = analyzer.perform_inference_on_models()
    analyzer.posterior_predictive_check(trace_dict, model_dict)
    analyzer.model_selection_metric_plots(trace_dict, model_dict)
