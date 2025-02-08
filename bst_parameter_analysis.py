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

# Default parameters
DEFAULT_PARAMS = {
    'population_size': 5, # number of models
    'window_size': 1000,
    'acquisition_function': 'pi',
    'drift_detection': True,
    'statistical_features': True
}

# Parameter variations to test
PARAM_VARIATIONS = {
    'population_size': [3, 5, 7, 9, 12],
    'window_size': [500, 1000, 2000, 5000],
    'acquisition_function': ['pi', 'ei', 'ucb']
}

# Classification hyperparameter ranges
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

# Regression hyperparameter ranges
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

def evaluate_classification(dataset, param_to_vary, output_dir, random_seed=42):
    stream = ARFFStream(path=f"data_cls/{dataset}.arff")
    
    for model_class, params_range, model_prefix in [
        (HoeffdingTree, ht_params_range, "HT"),
        (KNN, knn_params_range, "KNN")
    ]:
        # Use default parameters, only varying one parameter
        for param_value in PARAM_VARIATIONS[param_to_vary]:
            # Create configuration with default params
            config = DEFAULT_PARAMS.copy()
            # Override the parameter we're testing
            config[param_to_vary] = param_value
            
            model_name = f"Bayesian_{model_prefix}_{param_to_vary}_{param_value}"
            print(f"Running {model_name} on {dataset}")

            model = BayesianStreamTunerClassifier(
                model=model_class,
                schema=stream.get_schema(),
                params_range=params_range,
                random_seed=random_seed,
                n_models=config['population_size'],
                window_size=config['window_size'],
                acquisition_function=config['acquisition_function'],
                drift_detection=config['drift_detection'],
                statistical_features=config['statistical_features'],
                metric='accuracy'
            )

            evaluator = ClassificationEvaluator(
                schema=stream.get_schema(), 
                window_size=1000
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
                    print(f"Error processing instance {instances_processed}: {e}")
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
                'configuration': config,
                'varied_parameter': param_to_vary,
                'parameter_value': param_value
            }

            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/{model_name}_{dataset}.json', 'w') as f:
                json.dump(result, f, indent=2)

def evaluate_regression(dataset, param_to_vary, output_dir, random_seed=42):
    stream = ARFFStream(path=f"data_reg/{dataset}.arff")
    # Create a FilterStream and use the StandardisationFilter
    stream = Stream(CLI=f"-s ({stream.moa_stream.getCLICreationString(stream.moa_stream.__class__)}) -f StandardisationFilter ",
                   moa_stream=FilteredStream())
    
    for model_class, params_range, model_prefix in [
        (KNNRegressor, knn_reg_params_range, "KNN_REG"),  # Updated to KNN_REG for clarity
        (ORTO, orto_params_range, "ORTO")
    ]:
        for param_value in PARAM_VARIATIONS[param_to_vary]:
            # Create configuration with default params
            config = DEFAULT_PARAMS.copy()
            # Override the parameter we're testing
            config[param_to_vary] = param_value
            
            model_name = f"Bayesian_{model_prefix}_{param_to_vary}_{param_value}"
            print(f"Running {model_name} on {dataset}")

            model = BayesianStreamTunerRegressor(
                model=model_class,
                schema=stream.get_schema(),
                params_range=params_range,
                random_seed=random_seed,
                n_models=config['population_size'],
                window_size=config['window_size'],
                acquisition_function=config['acquisition_function'],
                drift_detection=config['drift_detection'],
                statistical_features=config['statistical_features'],
                metric='rmse'
            )

            evaluator = RegressionEvaluator(
                schema=stream.get_schema(), 
                window_size=1000
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
                    print(f"Error processing instance {instances_processed}: {e}")
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
                'configuration': config,
                'varied_parameter': param_to_vary,
                'parameter_value': param_value
            }

            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/{model_name}_{dataset}.json', 'w') as f:
                json.dump(result, f, indent=2)

if __name__ == "__main__":
    output_dir = "PARAMETER_RESULTS"
    os.makedirs(output_dir, exist_ok=True)

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

    # Test one parameter at a time
    parameters_to_test = ['population_size', 'window_size', 'acquisition_function']
    
    for param in parameters_to_test:
        print(f"\nTesting parameter: {param}")
        
        for dataset in classification_datasets:
            print(f"\nProcessing classification dataset: {dataset}")
            try:
                evaluate_classification(dataset, param, output_dir, random_seed)
            except Exception as e:
                print(f"Error processing {dataset}: {e}")

        for dataset in regression_datasets:
            print(f"\nProcessing regression dataset: {dataset}")
            try:
                evaluate_regression(dataset, param, output_dir, random_seed)
            except Exception as e:
                print(f"Error processing {dataset}: {e}")

print("Parameter analysis experiments completed!")