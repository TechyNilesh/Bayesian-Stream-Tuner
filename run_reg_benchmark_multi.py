from capymoa.stream import Stream
from moa.streams import FilteredStream  # type: ignore
from moa.streams.filters import StandardisationFilter, NormalisationFilter, RandomProjectionFilter  # type: ignore
from StreamTunner.BayesianStreamTuner import BayesianStreamTunerRegressor
from StreamTunner.SSPT import SSPTRegressor
from StreamTunner.MESSPT import MESSPTRegressor
from StreamTunner.RandomStreamSearch import RandomStreamSearchRegressor
from capymoa.regressor import KNNRegressor, ORTO
from capymoa.stream import ARFFStream
from capymoa.evaluation import RegressionEvaluator
import psutil
import time
import json
import os
import warnings
warnings.filterwarnings("ignore")

# Create  a Output folder
if not os.path.exists('RESULTS_REG_MULTI'):
    os.makedirs('RESULTS_REG_MULTI')


# Define hyperparameter ranges
orto_params_range = {
    'max_trees': (int, (5, 50)),  # Max number of trees in the option tree
    'grace_period': (int, (50, 500)),  # Instances between split attempts
    # Confidence level for split decisions
    'split_confidence': (float, (0.001, 0.1)),
    'tie_threshold': (float, (0.01, 0.2)),  # Threshold for tie-breaking splits
    'learning_ratio': (float, (0.001, 0.1)),  # Learning ratio for training
    'regression_tree': (bool, (True, True))
}

knn_reg_params_range = {
    'k': (int, (2, 15)),
    'window_size': (int, (100, 1000)),
    'median': (bool, (True, False)),
}


def create_learners(stream, random_seed=42):
    return {
        'ORTO': ORTO(stream.get_schema(),
                     random_seed=random_seed, regression_tree=True),
        'KNN_REG': KNNRegressor(stream.get_schema(),
                                random_seed=random_seed),
        "MESSPT_ORTO": MESSPTRegressor(
            model=ORTO,
            schema=stream.get_schema(),
            params_range=orto_params_range,
            metric='rmse',
            population_size=4,
            grace_period=500,
            convergence_sphere=0.001,
            F_ini=0.5,
            CR_ini=0.5,
            aug=0.025,
            random_seed=random_seed,
            verbose=False
        ),
        "MESSPT_KNN_REG": MESSPTRegressor(
            model=KNNRegressor,
            schema=stream.get_schema(),
            params_range=knn_reg_params_range,
            metric='rmse',
            population_size=4,
            grace_period=500,
            convergence_sphere=0.001,
            F_ini=0.5,
            CR_ini=0.5,
            aug=0.025,
            random_seed=random_seed,
            verbose=False
        ),
        "SSPT_ORTO": SSPTRegressor(
            model=ORTO,
            schema=stream.get_schema(),
            params_range=orto_params_range,
            metric='rmse',
            grace_period=500,
            convergence_sphere=0.001,
            random_seed=random_seed,
            verbose=False
        ),
        "SSPT_KNN_REG": SSPTRegressor(
            model=KNNRegressor,
            schema=stream.get_schema(),
            params_range=knn_reg_params_range,
            metric='rmse',
            grace_period=500,
            convergence_sphere=0.001,
            random_seed=random_seed,
            verbose=False
        ),
        "Bayesian_ORTO": BayesianStreamTunerRegressor(
            model=ORTO,
            schema=stream.get_schema(),
            params_range=orto_params_range,
            n_models=5,
            window_size=1000,
            metric='rmse',
            acquisition_function='pi',
            random_seed=random_seed,
            verbose=False,
        ),
        "Bayesian_KNN_REG": BayesianStreamTunerRegressor(
            model=KNNRegressor,
            schema=stream.get_schema(),
            params_range=knn_reg_params_range,
            n_models=5,
            metric='rmse',
            window_size=1000,
            acquisition_function='pi',
            random_seed=random_seed,
            verbose=False,
        ),
        "RandomSearch_ORTO": RandomStreamSearchRegressor(
            model=ORTO,
            schema=stream.get_schema(),
            param_range=orto_params_range,
            random_seed=random_seed,
            window_size=1000,
            metric='rmse',
            n_models=5,
            verbose=False
        ),
        "RandomSearch_KNN_REG": RandomStreamSearchRegressor(
            model=KNNRegressor,
            schema=stream.get_schema(),
            param_range=knn_reg_params_range,
            random_seed=random_seed,
            window_size=1000,
            metric='rmse',
            n_models=5,
            verbose=False
        )
    }


datasets = [
    'MetroTraffic',
    'diamonds',
    'sarcos',
    'fried',
    'FriedmanGra',
    'FriedmanLea',
    'FriedmanGsg',
    'hyperA',
    'bike',
    'health_insurance',
]


def evaluate_models(dataset, window_size=1000, run_count=None, seed=None):
    results = {}
    stream = ARFFStream(path=f"data_reg/{dataset}.arff")

    # Create a FilterStream and use the NormalisationFilter
    stream_std = Stream(CLI=f"-s ({stream.moa_stream.getCLICreationString(stream.moa_stream.__class__)}) \
    -f StandardisationFilter ", moa_stream=FilteredStream())

    models = create_learners(stream_std, random_seed=seed)

    for model_name, model in models.items():
        evaluator = RegressionEvaluator(
            schema=stream_std.get_schema(), window_size=window_size)
        stream_std.restart()

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB

        instances_processed = 0
        print(f"Processing {model_name}...")
        while stream_std.has_more_instances():
            try:
                instance = stream_std.next_instance()
                prediction = model.predict(instance)
                evaluator.update(instance.y_value, prediction)
                model.train(instance)
                instances_processed += 1
                if instances_processed % 1000 == 0:
                    print(f"  {instances_processed} instances processed")
            except Exception as e:
                print(f"Error processing instance {instances_processed}: {e}")
                continue

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB

        results[model_name] = {
            'evaluator': evaluator,
            'total_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'instances_processed': instances_processed
        }
        print(
            f"Finished processing {model_name}. Total instances: {instances_processed}")

    print(f"\nResults for {dataset}")
    best_r2 = 0
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Instances processed: {result['instances_processed']}")
        print(f"  R2: {result['evaluator'].r2()}")
        print(f"  RMSE: {result['evaluator'].rmse()}")
        print(f"  Total Time: {result['total_time']:.2f} seconds")
        print(f"  Memory Used: {result['memory_used']:.2f} MB")
        if result['evaluator'].r2() > best_r2:
            best_r2 = result['evaluator'].r2()
            best_model = name

    print(f"\nBest model: {best_model} with R2: {best_r2}")

    for name, result in results.items():
        result_json = {
            'run_counter': run_count if run_count else None,
            'random_seed': seed if seed else None,
            'model': name,
            'dataset': dataset,
            'instances_processed': result['instances_processed'],
            'metrics': result['evaluator'].metrics_dict(),
            'metrics_per_window': result['evaluator'].metrics_per_window().to_dict(orient='list'),
            'total_time': result['total_time'],
            'memory_used': result['memory_used']
        }

        # if run_count and seed:
        with open(f'RESULTS_REG_MULTI/{name}_{dataset}_{run_count}_{seed}.json', 'w', encoding='utf8') as json_file:
            json.dump(result_json, json_file)
        # else:
        #     with open(f'RESULTS_MULTI/{name}_{dataset}.json', 'w', encoding='utf8') as json_file:
        #         json.dump(result_json, json_file)


if __name__ == '__main__':
    import random
    num_runs = 10

    for dataset in datasets:
        print(f"Running benchmark for {dataset}")
        for run_count in range(num_runs):
            seed = random.randint(1, 100)
            print(f"  Run {run_count} with seed {seed}")
            try:
                evaluate_models(dataset, run_count=run_count, seed=seed)
            except Exception as e:
                print(f"Error processing {dataset}: {e}")
                continue
