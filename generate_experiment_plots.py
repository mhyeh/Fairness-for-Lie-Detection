import numpy as np
import os
from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score
import sys

constraint_name = sys.argv[1]
epsilon = sys.argv[2]

specfile = f'data/spec/lie_detection_{constraint_name}_{epsilon}.pkl'
spec = load_pickle(specfile)

performance_metric = 'accuracy'
n_trials = 50
data_fracs = np.logspace(-3,0,15)
n_workers = 8
verbose=False
results_dir = f'results/lie_detection_{constraint_name}_{epsilon}_{performance_metric}'
os.makedirs(results_dir,exist_ok=True)

plot_savename = os.path.join(results_dir,f'{constraint_name}_{epsilon}_{performance_metric}.png')

dataset = spec.dataset
test_features = dataset.features
test_labels = dataset.labels 

def perf_eval_fn(y_pred,y,**kwargs):
    if performance_metric == 'log_loss':
        return log_loss(y,y_pred)
    elif performance_metric == 'accuracy':
        return accuracy_score(y,y_pred > 0.5)

perf_eval_kwargs = {
    'X':test_features,
    'y':test_labels,
    }

plot_generator = SupervisedPlotGenerator(
    spec=spec,
    n_trials=n_trials,
    data_fracs=data_fracs,
    n_workers=n_workers,
    datagen_method='resample',
    perf_eval_fn=perf_eval_fn,
    constraint_eval_fns=[],
    results_dir=results_dir,
    perf_eval_kwargs=perf_eval_kwargs,
    )

plot_generator.run_baseline_experiment(
    model_name='random_classifier',verbose=verbose)

plot_generator.run_baseline_experiment(
    model_name='logistic_regression',verbose=verbose)

plot_generator.run_seldonian_experiment(verbose=verbose)

plot_generator.make_plots(fontsize=12,legend_fontsize=8,
    performance_label=performance_metric,
    # performance_yscale='log',
    savename=plot_savename)