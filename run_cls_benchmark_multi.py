from StreamTunner.BayesianStreamTuner import BayesianStreamTunerClassifier
from StreamTunner.BayesianStreamTunerSimple import BayesianStreamTunerClassifierSimple
from StreamTunner.SSPT import SSPTClassifier
from StreamTunner.MESSPT import MESSPTClassifier
from StreamTunner.RandomStreamSearch import RandomStreamSearchClassifier
from capymoa.classifier import HoeffdingTree, KNN, SAMkNN
from capymoa.stream import ARFFStream
from capymoa.evaluation import ClassificationEvaluator
import psutil
import time
import json
import os
import warnings
warnings.filterwarnings("ignore")

# Create  a Output folder
if not os.path.exists('RESULTS_MULTI'):
    os.makedirs('RESULTS_MULTI')

# Define hyperparameter ranges
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


def create_learners(stream, random_seed=42):
    return {
        'HT': HoeffdingTree(stream.get_schema(),
                            random_seed=random_seed),
        'KNN': KNN(stream.get_schema(),
                   random_seed=random_seed),
        "MESSPT_HT": MESSPTClassifier(
            model=HoeffdingTree,
            schema=stream.get_schema(),
            params_range=ht_params_range,
            metric='accuracy',
            population_size=4,
            grace_period=500,
            convergence_sphere=0.001,
            F_ini=0.5,
            CR_ini=0.5,
            aug=0.025,
            random_seed=random_seed,
            verbose=False
        ),
        "MESSPT_KNN": MESSPTClassifier(
            model=KNN,
            schema=stream.get_schema(),
            params_range=knn_params_range,
            metric='accuracy',
            population_size=4,
            grace_period=500,
            convergence_sphere=0.001,
            F_ini=0.5,
            CR_ini=0.5,
            aug=0.025,
            random_seed=random_seed,
            verbose=False
        ),
        "SSPT_HT": SSPTClassifier(
            model=HoeffdingTree,
            schema=stream.get_schema(),
            params_range=ht_params_range,
            metric='accuracy',
            grace_period=500,
            convergence_sphere=0.001,
            random_seed=random_seed,
            verbose=False
        ),
        "SSPT_KNN": SSPTClassifier(
            model=KNN,
            schema=stream.get_schema(),
            params_range=knn_params_range,
            metric='accuracy',
            grace_period=500,
            convergence_sphere=0.001,
            random_seed=random_seed,
            verbose=False
        ),
        "Bayesian_HT": BayesianStreamTunerClassifier(
            model=HoeffdingTree,
            schema=stream.get_schema(),
            params_range=ht_params_range,
            n_models=5,
            metric='accuracy',
            acquisition_function='pi',
            random_seed=random_seed,
            verbose=False
        ),
        "Bayesian_KNN": BayesianStreamTunerClassifier(
            model=KNN,
            schema=stream.get_schema(),
            params_range=knn_params_range,
            n_models=5,
            metric='accuracy',
            acquisition_function='pi',
            random_seed=random_seed,
            verbose=False
        ),
        "RandomSearch_HT": RandomStreamSearchClassifier(
            model=HoeffdingTree,
            schema=stream.get_schema(),
            param_range=ht_params_range,
            random_seed=random_seed,
            window_size=1000,
            metric='accuracy',
            n_models=5,
            verbose=False
        ),
        "RandomSearch_KNN": RandomStreamSearchClassifier(
            model=KNN,
            schema=stream.get_schema(),
            param_range=knn_params_range,
            random_seed=random_seed,
            window_size=1000,
            metric='accuracy',
            n_models=5,
            verbose=False
        ),
    }


datasets = [
    'electricity',
    'RTG_2abrupt',
    'RBFm_100k',
    'HYPERPLANE_01',
    'SEA_Mixed_5',
    'SEA_Abrubt_5',
    'sine_stream_with_drift',
    'nomao',
    'Forestcover',
    'new_airlines',
]


def evaluate_models(dataset, window_size=1000, run_count=None, seed=None):
    results = {}
    stream = ARFFStream(path=f"data_cls/{dataset}.arff")

    models = create_learners(stream, random_seed=seed)

    for model_name, model in models.items():
        evaluator = ClassificationEvaluator(
            schema=stream.get_schema(), window_size=window_size)
        stream.restart()

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB

        instances_processed = 0
        print(f"Processing {model_name}...")
        while stream.has_more_instances():
            try:
                instance = stream.next_instance()
                prediction = model.predict(instance)
                evaluator.update(instance.y_index, prediction)
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
    best_accuracy = 0
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Instances processed: {result['instances_processed']}")
        print(f"  Accuracy: {result['evaluator'].accuracy()}")
        print(f"  Kappa: {result['evaluator'].kappa()}")
        print(f"  F1 Score: {result['evaluator'].f1_score()}")
        print(f"  Total Time: {result['total_time']:.2f} seconds")
        print(f"  Memory Used: {result['memory_used']:.2f} MB")
        if result['evaluator'].accuracy() > best_accuracy:
            best_accuracy = result['evaluator'].accuracy()
            best_model = name

    print(f"\nBest model: {best_model} with accuracy: {best_accuracy}")

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

        try:
            output_path = f'RESULTS_MULTI/{name}_{dataset}_{run_count}_{seed}.json'
            with open(output_path, 'w', encoding='utf8') as json_file:
                json.dump(result_json, json_file)
        except IOError as e:
            print(f"Error saving results to {output_path}: {e}")


if __name__ == '__main__':
    import random
    num_runs = 10

    for dataset in datasets:
        print(f"Running benchmark for {dataset}")
        for run_count in range(num_runs):
            seed = random.randint(1, 1000)
            print(f"  Run {run_count} with seed {seed}")
            try:
                evaluate_models(dataset, run_count=run_count, seed=seed)
            except Exception as e:
                print(f"Error processing {dataset}: {e}")
                continue
