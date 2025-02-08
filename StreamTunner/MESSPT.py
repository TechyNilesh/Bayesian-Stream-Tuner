from capymoa.base import Classifier,Regressor
from capymoa.evaluation import ClassificationEvaluator,RegressionEvaluator
from capymoa.stream import Schema
from capymoa.drift.detectors import ADWIN
import numpy as np
import copy
import random
from typing import Dict, Tuple

class ModelWrapper:
    def __init__(self, model, evaluator, params):
        self.model = model
        self.evaluator = evaluator
        self.params = params

class MESSPTClassifier(Classifier):
    def __init__(self, model=None, schema=None, random_seed=1, params_range=None,
                 metric='accuracy', population_size=4, grace_period=500,
                 convergence_sphere=0.001, F_ini=0.5, CR_ini=0.5, aug=0.025, verbose=False):
        super().__init__(schema=schema, random_seed=random_seed)
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        if model is None:
            raise ValueError("A valid model must be provided.")
        self.base_model = model
        self.params_range = params_range if params_range is not None else {}
        self.metric = metric
        self.population_size = population_size
        self.population = []
        self.grace_period = grace_period
        self.convergence_sphere = convergence_sphere
        self.F = F_ini
        self.CR = CR_ini
        self.aug = aug
        self.drift_detector = ADWIN()
        self.converged = False
        self.verbose = verbose
        self._rng = random.Random(random_seed)
        self._n = 0
        self._old_best_params = None
        self._initialize_population()

    def __str__(self):
        return f'MESSPTClassifier({str(self.base_model.__name__)})'

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            params = self._random_config()
            model = self.base_model(schema=self.schema, **params)
            evaluator = ClassificationEvaluator(schema=self.schema)
            self.population.append(ModelWrapper(model, evaluator, params))
        if self.verbose:
            print(f"Initialized population with {self.population_size} models")

    def _random_config(self):
        return {param: self._generate(p_data) for param, p_data in self.params_range.items()}

    def _generate(self, p_data):
        p_type, p_range = p_data
        if p_type == int:
            return self._rng.randint(p_range[0], p_range[1])
        elif p_type == float:
            return self._rng.uniform(p_range[0], p_range[1])
        else:
            return self._rng.choice(p_range)

    def _de_cross_best_1(self):
        best = self.population[0]
        r1, r2 = self._rng.sample(self.population[1:], 2)
        new_params = {}
        for param, p_data in self.params_range.items():
            p_type, p_range = p_data
            if p_type in (int, float):
                new_value = best.params[param] + self.F * (r1.params[param] - r2.params[param])
                new_value = max(p_range[0], min(new_value, p_range[1]))
                new_params[param] = int(new_value) if p_type == int else new_value
            else:
                new_params[param] = self._rng.choice(p_range)
        return new_params

    def _crossover(self, target_params, trial_params):
        new_params = {}
        for param in self.params_range:
            if self._rng.random() < self.CR:
                new_params[param] = trial_params[param]
            else:
                new_params[param] = target_params[param]
        return new_params

    def _models_converged(self) -> bool:
        if self._old_best_params is None:
            return False
        
        distance = sum((self._old_best_params[param] - self.population[0].params[param])**2 
                       for param in self.params_range 
                       if isinstance(self.params_range[param][0], (int, float)))
        return distance < self.convergence_sphere**2

    def train(self, instance):
        self._n += 1

        if self.converged:
            self._train_converged(instance)
        else:
            self._train_not_converged(instance)

    def _train_converged(self, instance):
        best_model = self.population[0].model
        try:
            prediction = best_model.predict(instance)
            self.population[0].evaluator.update(instance.y_index, prediction)
            best_model.train(instance)
        except Exception as e:
            prediction = None
            if self.verbose:
                print(f"Error training model: {e}")

        drift_input = float(instance.y_index != prediction)
        self.drift_detector.add_element(drift_input)

        if self.drift_detector.detected_change():
            if self.verbose:
                print("Drift detected, reinitializing population")
            self._initialize_population()
            self.converged = False
            self._n = 0
            self.F = 0.5
            self.CR = 0.5
            self._old_best_params = None

    def _train_not_converged(self, instance):
        for wrapper in self.population:
            try:
                prediction = wrapper.model.predict(instance)
                wrapper.evaluator.update(instance.y_index, prediction)
                wrapper.model.train(instance)
            except Exception as e:
                if self.verbose:
                    print(f"Error training model: {e}")

        if self._n % self.grace_period == 0:
            self._update_population()

            if self._models_converged():
                if self.verbose:
                    print("Models converged")
                self.converged = True
            else:
                self.F = max(0.0, self.F - self.aug)
                self.CR = min(1.0, self.CR + self.aug)

    def _update_population(self):
        self.population.sort(key=lambda w: w.evaluator[self.metric], reverse=True)
        self._old_best_params = copy.deepcopy(self.population[0].params)

        new_population = [self.population[0]]
        for i in range(1, self.population_size):
            trial_params = self._de_cross_best_1()
            target_params = self.population[i].params
            new_params = self._crossover(target_params, trial_params)
            
            new_model = self.base_model(schema=self.schema, **new_params)
            new_evaluator = ClassificationEvaluator(schema=self.schema)
            new_population.append(ModelWrapper(new_model, new_evaluator, new_params))

        self.population = new_population

    def predict(self, instance):
        return self.population[0].model.predict(instance)
    
    def predict_proba(self, instance):
        return self.population[0].model.predict_proba(instance)

class MESSPTRegressor(Regressor):
    def __init__(self, model=None, schema=None, random_seed=1, params_range=None,
                 metric='rmse', population_size=4, grace_period=500,
                 convergence_sphere=0.001, F_ini=0.5, CR_ini=0.5, aug=0.025, verbose=False):
        super().__init__(schema=schema, random_seed=random_seed)
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        if model is None:
            raise ValueError("A valid model must be provided.")
        self.base_model = model
        self.params_range = params_range if params_range is not None else {}
        self.metric = metric
        self.population_size = population_size
        self.population = []
        self.grace_period = grace_period
        self.convergence_sphere = convergence_sphere
        self.F = F_ini
        self.CR = CR_ini
        self.aug = aug
        self.drift_detector = ADWIN()
        self.converged = False
        self.verbose = verbose
        self._rng = random.Random(random_seed)
        self._n = 0
        self._old_best_params = None
        self._initialize_population()

    def __str__(self):
        return f'MESSPTRegressor({str(self.base_model.__name__)})'

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            params = self._random_config()
            model = self.base_model(schema=self.schema, **params)
            evaluator = RegressionEvaluator(schema=self.schema)
            self.population.append(ModelWrapper(model, evaluator, params))
        if self.verbose:
            print(f"Initialized population with {self.population_size} models")

    def _random_config(self):
        return {param: self._generate(p_data) for param, p_data in self.params_range.items()}

    def _generate(self, p_data):
        p_type, p_range = p_data
        if p_type == int:
            return self._rng.randint(p_range[0], p_range[1])
        elif p_type == float:
            return self._rng.uniform(p_range[0], p_range[1])
        else:
            return self._rng.choice(p_range)

    def _de_cross_best_1(self):
        best = self.population[0]
        r1, r2 = self._rng.sample(self.population[1:], 2)
        new_params = {}
        for param, p_data in self.params_range.items():
            p_type, p_range = p_data
            if p_type in (int, float):
                new_value = best.params[param] + self.F * (r1.params[param] - r2.params[param])
                new_value = max(p_range[0], min(new_value, p_range[1]))
                new_params[param] = int(new_value) if p_type == int else new_value
            else:
                new_params[param] = self._rng.choice(p_range)
        return new_params

    def _crossover(self, target_params, trial_params):
        new_params = {}
        for param in self.params_range:
            if self._rng.random() < self.CR:
                new_params[param] = trial_params[param]
            else:
                new_params[param] = target_params[param]
        return new_params

    def _models_converged(self) -> bool:
        if self._old_best_params is None:
            return False
        
        distance = sum((self._old_best_params[param] - self.population[0].params[param])**2 
                       for param in self.params_range 
                       if isinstance(self.params_range[param][0], (int, float)))
        return distance < self.convergence_sphere**2

    def train(self, instance):
        self._n += 1

        if self.converged:
            self._train_converged(instance)
        else:
            self._train_not_converged(instance)

    def _train_converged(self, instance):
        best_model = self.population[0].model
        try:
            prediction = best_model.predict(instance)
            self.population[0].evaluator.update(instance.y_value, prediction)
            best_model.train(instance)
        except Exception as e:
            prediction = None
            if self.verbose:
                print(f"Error training model: {e}")

        drift_input = float(instance.y_value != prediction)
        self.drift_detector.add_element(drift_input)

        if self.drift_detector.detected_change():
            if self.verbose:
                print("Drift detected, reinitializing population")
            self._initialize_population()
            self.converged = False
            self._n = 0
            self.F = 0.5
            self.CR = 0.5
            self._old_best_params = None

    def _train_not_converged(self, instance):
        for wrapper in self.population:
            try:
                prediction = wrapper.model.predict(instance)
                wrapper.evaluator.update(instance.y_value, prediction)
                wrapper.model.train(instance)
            except Exception as e:
                if self.verbose:
                    print(f"Error training model: {e}")

        if self._n % self.grace_period == 0:
            self._update_population()

            if self._models_converged():
                if self.verbose:
                    print("Models converged")
                self.converged = True
            else:
                self.F = max(0.0, self.F - self.aug)
                self.CR = min(1.0, self.CR + self.aug)

    def _update_population(self):
        self.population.sort(key=lambda w: w.evaluator.metrics_dict()[self.metric])
        self._old_best_params = copy.deepcopy(self.population[0].params)

        new_population = [self.population[0]]
        for i in range(1, self.population_size):
            trial_params = self._de_cross_best_1()
            target_params = self.population[i].params
            new_params = self._crossover(target_params, trial_params)
            
            new_model = self.base_model(schema=self.schema, **new_params)
            new_evaluator = RegressionEvaluator(schema=self.schema)
            new_population.append(ModelWrapper(new_model, new_evaluator, new_params))

        self.population = new_population

    def predict(self, instance):
        return self.population[0].model.predict(instance)
    
    def predict_proba(self, instance):
        return self.population[0].model.predict_proba(instance)