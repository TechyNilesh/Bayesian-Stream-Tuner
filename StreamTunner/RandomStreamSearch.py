import numpy as np
from capymoa.base import Classifier
from capymoa.instance import Instance, LabeledInstance
from capymoa.type_alias import LabelProbabilities, LabelIndex
from typing import Dict, Tuple, Optional, Any
import random
from capymoa.base import Classifier,Regressor
from capymoa.evaluation import ClassificationEvaluator, RegressionEvaluator

class RandomStreamSearchClassifier(Classifier):
    def __init__(self, model, schema, param_range: Dict[str, Tuple[Any, Any]],random_seed: Optional[int] = None, window_size=1000, metric='accuracy', n_models=5, verbose=False):
        super().__init__(schema=schema,random_seed=random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.base_model = model
        self.param_range = param_range
        self.window_size = window_size
        self.metric = metric
        self.n_models = n_models
        self.models = []
        self.evaluators = []
        self.configurations = []
        self.best_model_idx = 0
        self.sample_count = 0
        self._initialize_models()
        self.verbose = verbose

    def __str__(self):
        return f"RandomStreamSearchClassifier({str(self.base_model.__name__)})"

    def _initialize_models(self):
        self._select_random_configs()
        
        self.models = []
        self.evaluators = []
        for config in self.configurations:
            model = self.base_model(schema=self.schema, **config)
            self.models.append(model)
            evaluator = ClassificationEvaluator(schema=self.schema, window_size=self.window_size)
            self.evaluators.append(evaluator)

    def _select_random_configs(self):
        self.configurations = []
        for _ in range(self.n_models):
            config = {}
            for param, (param_type, param_range) in self.param_range.items():
                if param_type == int:
                    config[param] = np.random.randint(param_range[0], param_range[1] + 1)
                elif param_type == float:
                    config[param] = np.random.uniform(param_range[0], param_range[1])
                else:
                    config[param] = np.random.choice(param_range)
            self.configurations.append(config)

    def train(self, instance):
        for idx, model in enumerate(self.models):
            try:
                prediction = model.predict(instance)
                self.evaluators[idx].update(instance.y_index, prediction)
                model.train(instance)
            except:
                pass

        self.sample_count += 1

        if self.sample_count % self.window_size == 0:
            performances = [evaluator[self.metric] for evaluator in self.evaluators]
            self.best_model_idx = np.argmax(performances)
            
            if self.verbose:
                print(f"OnlineRandomSearch - Samples Processed: {self.sample_count}")
                print(f"Best model configuration: {self.configurations[self.best_model_idx]}")
                print(f"Best model performance: {performances[self.best_model_idx]}")

            # Select new random configurations for the next window
            self._select_random_configs()
            
            # Keep the best model from the previous window
            new_models = [self.models[self.best_model_idx]]
            new_evaluators = [self.evaluators[self.best_model_idx]]
            
            # Create new models for the other configurations
            for config in self.configurations[1:]:  # Skip the first config as it's for the best model
                model = self.base_model(schema=self.schema, **config)
                new_models.append(model)
                evaluator = ClassificationEvaluator(schema=self.schema, window_size=self.window_size)
                new_evaluators.append(evaluator)
            
            self.models = new_models
            self.evaluators = new_evaluators
            self.best_model_idx = 0  # The best model from the previous window is now at index 0

    def predict(self, instance):
        return self.models[self.best_model_idx].predict(instance)

    def predict_proba(self, instance):
        return self.models[self.best_model_idx].predict_proba(instance)

class RandomStreamSearchRegressor(Regressor):
    def __init__(self, model, schema, param_range: Dict[str, Tuple[Any, Any]],random_seed: Optional[int] = None, window_size=1000, metric='rmse', n_models=5, verbose=False):
        super().__init__(schema=schema,random_seed=random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.base_model = model
        self.param_range = param_range
        self.window_size = window_size
        self.metric = metric
        self.n_models = n_models
        self.models = []
        self.evaluators = []
        self.configurations = []
        self.best_model_idx = 0
        self.sample_count = 0
        self._initialize_models()
        self.verbose = verbose

    def __str__(self):
        return f"RandomStreamSearchRegressor({str(self.base_model.__name__)})"

    def _initialize_models(self):
        self._select_random_configs()
        
        self.models = []
        self.evaluators = []
        for config in self.configurations:
            model = self.base_model(schema=self.schema, **config)
            self.models.append(model)
            evaluator = RegressionEvaluator(schema=self.schema, window_size=self.window_size)
            self.evaluators.append(evaluator)

    def _select_random_configs(self):
        self.configurations = []
        for _ in range(self.n_models):
            config = {}
            for param, (param_type, param_range) in self.param_range.items():
                if param_type == int:
                    config[param] = np.random.randint(param_range[0], param_range[1] + 1)
                elif param_type == float:
                    config[param] = np.random.uniform(param_range[0], param_range[1])
                else:
                    config[param] = np.random.choice(param_range)
            self.configurations.append(config)

    def train(self, instance):
        for idx, model in enumerate(self.models):
            try:
                prediction = model.predict(instance)
                self.evaluators[idx].update(instance.y_value, prediction)
                model.train(instance)
            except:
                pass

        self.sample_count += 1

        if self.sample_count % self.window_size == 0:
            performances = [evaluator.metrics_dict()[self.metric] for evaluator in self.evaluators]
            self.best_model_idx = np.argmin(performances)
            
            if self.verbose:
                print(f"OnlineRandomSearch - Samples Processed: {self.sample_count}")
                print(f"Best model configuration: {self.configurations[self.best_model_idx]}")
                print(f"Best model performance: {performances[self.best_model_idx]}")

            # Select new random configurations for the next window
            self._select_random_configs()
            
            # Keep the best model from the previous window
            new_models = [self.models[self.best_model_idx]]
            new_evaluators = [self.evaluators[self.best_model_idx]]
            
            # Create new models for the other configurations
            for config in self.configurations[1:]:  # Skip the first config as it's for the best model
                model = self.base_model(schema=self.schema, **config)
                new_models.append(model)
                evaluator = RegressionEvaluator(schema=self.schema, window_size=self.window_size)
                new_evaluators.append(evaluator)
            
            self.models = new_models
            self.evaluators = new_evaluators
            self.best_model_idx = 0  # The best model from the previous window is now at index 0

    def predict(self, instance):
        return self.models[self.best_model_idx].predict(instance)

    def predict_proba(self, instance):
        return self.models[self.best_model_idx].predict_proba(instance)
    
    
class RandomSearchStreamClassifierPE(Classifier):
    def __init__(self,
                 classifier_class: Classifier,
                 schema,
                 params_range: Dict[str, Tuple[Any, Any]],
                 n_combinations: int = 10,
                 random_seed: Optional[int] = None,
                 verbose: bool = False):
        super().__init__(schema=schema, random_seed=random_seed)
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.classifier_class = classifier_class
        self.params_range = params_range
        self.n_combinations = n_combinations
        self.rng = np.random.default_rng(random_seed)
        self.verbose = verbose
        
        self.models = []
        self.model_accuracy = []
        self.hyperparameters = []
        self.seen_instances = 0
        
        self._initialize_models()

    def __str__(self):
        return f"RandomSearchStreamClassifierPE({str(self.classifier_class.__name__)})"

    def _generate_random_config(self):
        config = {}
        for param, (p_type, p_range) in self.params_range.items():
            if p_type == int:
                config[param] = self.rng.integers(p_range[0], p_range[1] + 1)
            elif p_type == float:
                config[param] = self.rng.uniform(p_range[0], p_range[1])
            else:
                config[param] = self.rng.choice(p_range)
        return config

    def _initialize_models(self):
        self.hyperparameters = [self._generate_random_config() for _ in range(self.n_combinations)]
        self.models = [self.classifier_class(schema=self.schema, **hp_kwargs) for hp_kwargs in self.hyperparameters]
        self.model_accuracy = [0.0 for _ in range(len(self.models))]
        
        if self.verbose:
            print(f"Initialized {self.n_combinations} models with the following configurations:")
            for idx, config in enumerate(self.hyperparameters):
                print(f"Model {idx}: {config}")

    def train(self, instance: LabeledInstance):
        if self.schema is None:
            self.schema = instance.schema
            for model in self.models:
                model.schema = self.schema

        for model_idx, model in enumerate(self.models):
            y_hat = model.predict(instance)
            correct = int(y_hat == instance.y_index)
            old_acc = self.model_accuracy[model_idx]
            new_acc = (old_acc * self.seen_instances + correct) / (self.seen_instances + 1)
            self.model_accuracy[model_idx] = new_acc
            model.train(instance)
        
        self.seen_instances += 1

        if self.verbose and self.seen_instances % 1000 == 0:
            best_model_idx = np.argmax(self.model_accuracy)
            print(f"Instances processed: {self.seen_instances}")
            print(f"Best model index: {best_model_idx}")
            print(f"Best model accuracy: {self.model_accuracy[best_model_idx]:.4f}")
            print(f"Best model configuration: {self.hyperparameters[best_model_idx]}")

        return self

    def predict(self, instance: Instance) -> Optional[LabelIndex]:
        best_model_idx = np.argmax(self.model_accuracy)
        best_model = self.models[best_model_idx]
        return best_model.predict(instance)

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        best_model_idx = np.argmax(self.model_accuracy)
        best_model = self.models[best_model_idx]
        return best_model.predict_proba(instance)