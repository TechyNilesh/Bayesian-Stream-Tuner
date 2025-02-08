import os
from capymoa.classifier import HoeffdingTree, KNN
from capymoa.regressor import KNNRegressor, ORTO
from StreamTunner.BayesianStreamTuner import BayesianStreamTunerClassifier, BayesianStreamTunerRegressor
from capymoa.evaluation import ClassificationEvaluator, RegressionEvaluator
from capymoa.stream import ARFFStream, Stream
from moa.streams.filters import StandardisationFilter
from moa.streams import FilteredStream
import psutil
import time
import json
import warnings
warnings.filterwarnings("ignore")

# Fixed parameters
FIXED_PARAMS = {
    'population_size': 5,
    'window_size': 1000,
    'acquisition_function': 'pi',
}

# Hyperparameter ranges
ht_params_range = {
    'grace_period': (int, (50, 500)),
    'confidence': (float, (0.001, 0.1)),
    'tie_threshold': (float, (0.01, 0.2)),
    'leaf_prediction': (str, ['MajorityClass', 'NaiveBayesAdaptive']),
    'nb_threshold': (int, (0, 50)),
    'split_criterion': (str, ['InfoGainSplitCriterion', 'GiniSplitCriterion']),
}

knn_params_range = {
    'k': (int, (2, 15)),
    'window_size': (int, (100, 1000)),
}

knn_reg_params_range = {
    'k': (int, (2, 15)),
    'window_size': (int, (100, 1000)),
    'median': (bool, (True, False)),
}

orto_params_range = {
    'max_trees': (int, (5, 50)),
    'grace_period': (int, (50, 500)),
    'split_confidence': (float, (0.001, 0.1)),
    'tie_threshold': (float, (0.01, 0.2)),
    'learning_ratio': (float, (0.001, 0.1)),
    'regression_tree': (bool, (True, True))
}

def evaluate_classification(dataset, output_dir, random_seed=42):
    stream = ARFFStream(path=f"data_cls/{dataset}.arff")
    stream = Stream(CLI=f"-s ({stream.moa_stream.getCLICreationString(stream.moa_stream.__class__)}) -f StandardisationFilter ",
                   moa_stream=FilteredStream())

    # Different component configurations to test
    # Keep only the model parameters in the config dict
    configs = [
        {'config': {'drift_detection': False, 'statistical_features': False}, 'suffix': 'DD_F_ST_F'},
        {'config': {'drift_detection': True, 'statistical_features': False}, 'suffix': 'DD_T_ST_F'},
        {'config': {'drift_detection': False, 'statistical_features': True}, 'suffix': 'DD_F_ST_T'},
        {'config': {'drift_detection': True, 'statistical_features': True}, 'suffix': 'DD_T_ST_T'}
    ]

    for model_class, params_range, model_prefix in [
        (HoeffdingTree, ht_params_range, "HT"),
        (KNN, knn_params_range, "KNN")
    ]:
        for config_dict in configs:
            model_name = f"Bayesian_{model_prefix}_{config_dict['suffix']}"
            print(f"Running {model_name} on {dataset}")

            model = BayesianStreamTunerClassifier(
                model=model_class,
                schema=stream.get_schema(),
                params_range=params_range,
                n_models=FIXED_PARAMS['population_size'],
                window_size=FIXED_PARAMS['window_size'],
                metric='accuracy',
                acquisition_function=FIXED_PARAMS['acquisition_function'],
                random_seed=random_seed,
                **config_dict['config']  # Only pass the actual config parameters
            )

            evaluator = ClassificationEvaluator(
                schema=stream.get_schema(), 
                window_size=FIXED_PARAMS['window_size']
            )
            stream.restart()

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            instances_processed = 0
            while stream.has_more_instances():
                try:
                    instance = stream.next_instance()
                    prediction = model.predict(instance)
                    evaluator.update(instance.y_index, prediction)
                    model.train(instance)
                    instances_processed += 1
                except Exception as e:
                    print(f"Error processing instance {instances_processed} for {model_name}: {e}")
                    continue

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            result = {
                'model': model_name,
                'dataset': dataset,
                'instances_processed': instances_processed,
                'metrics': evaluator.metrics_dict(),
                'metrics_per_window': evaluator.metrics_per_window().to_dict(orient='list'),
                'total_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'configuration': {
                    **FIXED_PARAMS,
                    **config_dict['config']
                }
            }

            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/{model_name}_{dataset}.json', 'w') as f:
                json.dump(result, f, indent=2)

def evaluate_regression(dataset, output_dir, random_seed=42):
    stream = ARFFStream(path=f"data_reg/{dataset}.arff")
    stream = Stream(CLI=f"-s ({stream.moa_stream.getCLICreationString(stream.moa_stream.__class__)}) -f StandardisationFilter ",
                   moa_stream=FilteredStream())

    # Different component configurations to test
    configs = [
        {'config': {'drift_detection': False, 'statistical_features': False}, 'suffix': 'DD_F_ST_F'},
        {'config': {'drift_detection': True, 'statistical_features': False}, 'suffix': 'DD_T_ST_F'},
        {'config': {'drift_detection': False, 'statistical_features': True}, 'suffix': 'DD_F_ST_T'},
        {'config': {'drift_detection': True, 'statistical_features': True}, 'suffix': 'DD_T_ST_T'}
    ]
    
    for model_class, params_range, model_prefix in [
        (KNNRegressor, knn_reg_params_range, "KNN_REG"),
        (ORTO, orto_params_range, "ORTO")
    ]:
        for config_dict in configs:
            model_name = f"Bayesian_{model_prefix}_{config_dict['suffix']}"
            print(f"Running {model_name} on {dataset}")

            model = BayesianStreamTunerRegressor(
                model=model_class,
                schema=stream.get_schema(),
                params_range=params_range,
                n_models=FIXED_PARAMS['population_size'],
                window_size=FIXED_PARAMS['window_size'],
                metric='rmse',
                acquisition_function=FIXED_PARAMS['acquisition_function'],
                random_seed=random_seed,
                **config_dict['config']  # Only pass the actual config parameters
            )

            evaluator = RegressionEvaluator(
                schema=stream.get_schema(), 
                window_size=FIXED_PARAMS['window_size']
            )
            stream.restart()

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            instances_processed = 0
            while stream.has_more_instances():
                try:
                    instance = stream.next_instance()
                    prediction = model.predict(instance)
                    evaluator.update(instance.y_value, prediction)
                    model.train(instance)
                    instances_processed += 1
                except Exception as e:
                    print(f"Error processing instance {instances_processed} for {model_name}: {e}")
                    continue

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            result = {
                'model': model_name,
                'dataset': dataset,
                'instances_processed': instances_processed,
                'metrics': evaluator.metrics_dict(),
                'metrics_per_window': evaluator.metrics_per_window().to_dict(orient='list'),
                'total_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'configuration': {
                    **FIXED_PARAMS,
                    **config_dict['config']
                }
            }

            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/{model_name}_{dataset}.json', 'w') as f:
                json.dump(result, f, indent=2)

if __name__ == "__main__":
    output_dir = "COMPONENT_RESULTS"
    os.makedirs(output_dir, exist_ok=True)

    # Define datasets
    classification_datasets = [
        'RTG_2abrupt',      # Sudden drift
        'SEA_Mixed_5',      # Gradual drift
        'electricity',      # Real-world
        'Forestcover'       # High-dimensional
    ]

    regression_datasets = [
        'FriedmanGra',      # Synthetic
        'bike',             # Real-world streaming
        'MetroTraffic',     # Real-world complex
        'sarcos'            # High-dimensional
    ]

    # Use a single random seed for all experiments
    random_seed = 42
    
    # Process classification datasets
    for dataset in classification_datasets:
        print(f"\nProcessing classification dataset: {dataset}")
        try:
            evaluate_classification(dataset, output_dir, random_seed)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    # Process regression datasets
    for dataset in regression_datasets:
        print(f"\nProcessing regression dataset: {dataset}")
        try:
            evaluate_regression(dataset, output_dir, random_seed)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

print("Component analysis experiments completed!")