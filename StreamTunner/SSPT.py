from capymoa.base import Classifier,Regressor
from capymoa.evaluation import ClassificationEvaluator,RegressionEvaluator
from capymoa.drift.detectors import ADWIN
import numpy as np
import random
from typing import Dict, Tuple, List, Any

class ModelWrapper:
    def __init__(self, model, evaluator, params):
        self.model = model
        self.evaluator = evaluator
        self.params = params
        self.instances_seen = 0

    def update(self, y_true, y_pred):
        self.evaluator.update(y_true, y_pred)
        self.instances_seen += 1

    def get_metric(self, metric):
        if self.instances_seen == 0:
            return float('-inf')  # or some other default value
        try:
            if isinstance(self.evaluator, ClassificationEvaluator):
                return self.evaluator[metric]
            return self.evaluator.metrics_dict()[metric]
        except:
            return float('-inf')  # or some other default value

class SSPTClassifier(Classifier):
    def __init__(self, model=None, schema=None, random_seed=1, params_range=None,
                 metric='accuracy', grace_period=500, convergence_sphere=0.001, verbose=False):
        super().__init__(schema=schema, random_seed=random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if model is None:
            raise ValueError("A valid model must be provided.")
        self.base_model = model
        self.params_range = params_range if params_range is not None else {}
        self.metric = metric
        self.grace_period = grace_period
        self.convergence_sphere = convergence_sphere
        self.drift_detector = ADWIN()
        self.converged = False
        self.verbose = verbose
        self._rng = random.Random(random_seed)
        self._n = 0
        self._old_centroid = None
        self._simplex: List[ModelWrapper] = []
        self._expanded: Dict[str, ModelWrapper] = {}
        self._initialize_simplex()

    def __str__(self):
        return f"SSPTClassifier({str(self.base_model.__name__)})"

    def _initialize_simplex(self):
        self._simplex = []
        for _ in range(3):  # Simplex has 3 points
            params = self._random_config()
            model = self.base_model(schema=self.schema, **params)
            evaluator = ClassificationEvaluator(schema=self.schema)
            self._simplex.append(ModelWrapper(model, evaluator, params))
        if self.verbose:
            print(f"Initialized simplex with 3 models")

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

    def _sort_simplex(self):
        self._simplex.sort(key=lambda w: w.get_metric(self.metric), reverse=True)

    def _nelder_mead_expansion(self):
        expanded = {}
        best, good, worst = self._simplex

        def combine(p1: Dict[str, Any], p2: Dict[str, Any], func) -> Dict[str, Any]:
            result = {}
            for param, value in self.params_range.items():
                p_type, p_range = value
                if p_type in (int, float):
                    combined_value = func(p1[param], p2[param])
                    if p_type == int:
                        combined_value = int(round(combined_value))
                    result[param] = max(p_range[0], min(combined_value, p_range[1]))
                else:
                    result[param] = self._rng.choice([p1[param], p2[param]])
            return result

        midpoint = combine(best.params, good.params, lambda h1, h2: (h1 + h2) / 2)
        reflection = combine(midpoint, worst.params, lambda h1, h2: 2 * h1 - h2)
        expansion = combine(reflection, midpoint, lambda h1, h2: 2 * h1 - h2)
        shrink = combine(best.params, worst.params, lambda h1, h2: (h1 + h2) / 2)
        contraction1 = combine(midpoint, worst.params, lambda h1, h2: (h1 + h2) / 2)
        contraction2 = combine(midpoint, reflection, lambda h1, h2: (h1 + h2) / 2)

        for name, params in [('midpoint', midpoint), ('reflection', reflection),
                             ('expansion', expansion), ('shrink', shrink),
                             ('contraction1', contraction1), ('contraction2', contraction2)]:
            model = self.base_model(schema=self.schema, **params)
            evaluator = ClassificationEvaluator(schema=self.schema)
            expanded[name] = ModelWrapper(model, evaluator, params)

        return expanded

    def _nelder_mead_operators(self):
        b, g, w = self._simplex
        r = self._expanded['reflection']
        c1, c2 = self._expanded['contraction1'], self._expanded['contraction2']
        e = self._expanded['expansion']
        s = self._expanded['shrink']

        if c1.get_metric(self.metric) > c2.get_metric(self.metric):
            contraction = c1
        else:
            contraction = c2

        if r.get_metric(self.metric) > g.get_metric(self.metric):
            if b.get_metric(self.metric) > r.get_metric(self.metric):
                self._simplex[2] = r
            else:
                if e.get_metric(self.metric) > b.get_metric(self.metric):
                    self._simplex[2] = e
                else:
                    self._simplex[2] = r
        else:
            if r.get_metric(self.metric) > w.get_metric(self.metric):
                self._simplex[2] = r
            else:
                if contraction.get_metric(self.metric) > w.get_metric(self.metric):
                    self._simplex[2] = contraction
                else:
                    self._simplex[2] = s
                    self._simplex[1] = self._expanded['midpoint']

        self._sort_simplex()

    def _models_converged(self) -> bool:
        if self._old_centroid is None:
            return False
        
        current_centroid = self._calculate_centroid()
        distance = sum((self._old_centroid[param] - current_centroid[param])**2 
                       for param, (p_type, _) in self.params_range.items()
                       if p_type in (int, float))
        return distance < self.convergence_sphere**2

    def _calculate_centroid(self):
        centroid = {}
        for param, (p_type, p_range) in self.params_range.items():
            if p_type in (int, float):
                value = np.mean([w.params[param] for w in self._simplex])
                if p_type == int:
                    value = int(round(value))
                centroid[param] = max(p_range[0], min(value, p_range[1]))
            else:
                # For categorical parameters, use the mode
                values = [w.params[param] for w in self._simplex]
                centroid[param] = max(set(values), key=values.count)
        return centroid

    def train(self, instance):
        self._n += 1

        if self.converged:
            self._train_converged(instance)
        else:
            self._train_not_converged(instance)

    def _train_converged(self, instance):
        best_model = self._simplex[0].model
        prediction = best_model.predict(instance)
        self._simplex[0].update(instance.y_index, prediction)
        best_model.train(instance)

        drift_input = float(instance.y_index != prediction)
        self.drift_detector.add_element(drift_input)

        if self.drift_detector.detected_change():
            if self.verbose:
                print("Drift detected, reinitializing simplex")
            self._initialize_simplex()
            self.converged = False
            self._n = 0
            self._old_centroid = None

    def _train_not_converged(self, instance):
        for wrapper in self._simplex:
            prediction = wrapper.model.predict(instance)
            wrapper.update(instance.y_index, prediction)
            wrapper.model.train(instance)

        if self._expanded:
            for wrapper in self._expanded.values():
                prediction = wrapper.model.predict(instance)
                wrapper.update(instance.y_index, prediction)
                wrapper.model.train(instance)

        if self._n % self.grace_period == 0:
            self._update_simplex()

            if self._models_converged():
                if self.verbose:
                    print("Models converged")
                self.converged = True

    def _update_simplex(self):
        self._sort_simplex()
        self._old_centroid = self._calculate_centroid()

        if not self._expanded:
            self._expanded = self._nelder_mead_expansion()

        self._nelder_mead_operators()
        self._expanded = None

    def predict(self, instance):
        return self._simplex[0].model.predict(instance)
    
    def predict_proba(self, instance):
        return self._simplex[0].model.predict_proba(instance)

class SSPTRegressor(Regressor):
    def __init__(self, model=None, schema=None, random_seed=1, params_range=None,
                 metric='rmse', grace_period=500, convergence_sphere=0.001, verbose=False):
        super().__init__(schema=schema, random_seed=random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if model is None:
            raise ValueError("A valid model must be provided.")
        self.base_model = model
        self.params_range = params_range if params_range is not None else {}
        self.metric = metric
        self.grace_period = grace_period
        self.convergence_sphere = convergence_sphere
        self.drift_detector = ADWIN()
        self.converged = False
        self.verbose = verbose
        self._rng = random.Random(random_seed)
        self._n = 0
        self._old_centroid = None
        self._simplex: List[ModelWrapper] = []
        self._expanded: Dict[str, ModelWrapper] = {}
        self._initialize_simplex()

    def __str__(self):
        return f"SSPTRegressor({str(self.base_model.__name__)})"

    def _initialize_simplex(self):
        self._simplex = []
        for _ in range(3):  # Simplex has 3 points
            params = self._random_config()
            model = self.base_model(schema=self.schema, **params)
            evaluator = RegressionEvaluator(schema=self.schema)
            self._simplex.append(ModelWrapper(model, evaluator, params))
        if self.verbose:
            print(f"Initialized simplex with 3 models")

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

    def _sort_simplex(self):
        self._simplex.sort(key=lambda w: w.get_metric(self.metric))

    def _nelder_mead_expansion(self):
        expanded = {}
        best, good, worst = self._simplex

        def combine(p1: Dict[str, Any], p2: Dict[str, Any], func) -> Dict[str, Any]:
            result = {}
            for param, value in self.params_range.items():
                p_type, p_range = value
                if p_type in (int, float):
                    combined_value = func(p1[param], p2[param])
                    if p_type == int:
                        combined_value = int(round(combined_value))
                    result[param] = max(p_range[0], min(combined_value, p_range[1]))
                else:
                    result[param] = self._rng.choice([p1[param], p2[param]])
            return result

        midpoint = combine(best.params, good.params, lambda h1, h2: (h1 + h2) / 2)
        reflection = combine(midpoint, worst.params, lambda h1, h2: 2 * h1 - h2)
        expansion = combine(reflection, midpoint, lambda h1, h2: 2 * h1 - h2)
        shrink = combine(best.params, worst.params, lambda h1, h2: (h1 + h2) / 2)
        contraction1 = combine(midpoint, worst.params, lambda h1, h2: (h1 + h2) / 2)
        contraction2 = combine(midpoint, reflection, lambda h1, h2: (h1 + h2) / 2)

        for name, params in [('midpoint', midpoint), ('reflection', reflection),
                             ('expansion', expansion), ('shrink', shrink),
                             ('contraction1', contraction1), ('contraction2', contraction2)]:
            model = self.base_model(schema=self.schema, **params)
            evaluator = RegressionEvaluator(schema=self.schema)
            expanded[name] = ModelWrapper(model, evaluator, params)

        return expanded

    def _nelder_mead_operators(self):
        b, g, w = self._simplex
        r = self._expanded['reflection']
        c1, c2 = self._expanded['contraction1'], self._expanded['contraction2']
        e = self._expanded['expansion']
        s = self._expanded['shrink']

        if c1.get_metric(self.metric) < c2.get_metric(self.metric):
            contraction = c1
        else:
            contraction = c2

        if r.get_metric(self.metric) < g.get_metric(self.metric):
            if b.get_metric(self.metric) < r.get_metric(self.metric):
                self._simplex[2] = r
            else:
                if e.get_metric(self.metric) < b.get_metric(self.metric):
                    self._simplex[2] = e
                else:
                    self._simplex[2] = r
        else:
            if r.get_metric(self.metric) < w.get_metric(self.metric):
                self._simplex[2] = r
            else:
                if contraction.get_metric(self.metric) < w.get_metric(self.metric):
                    self._simplex[2] = contraction
                else:
                    self._simplex[2] = s
                    self._simplex[1] = self._expanded['midpoint']

        self._sort_simplex()

    def _models_converged(self) -> bool:
        if self._old_centroid is None:
            return False
        
        current_centroid = self._calculate_centroid()
        distance = sum((self._old_centroid[param] - current_centroid[param])**2 
                       for param, (p_type, _) in self.params_range.items()
                       if p_type in (int, float))
        return distance < self.convergence_sphere**2

    def _calculate_centroid(self):
        centroid = {}
        for param, (p_type, p_range) in self.params_range.items():
            if p_type in (int, float):
                value = np.mean([w.params[param] for w in self._simplex])
                if p_type == int:
                    value = int(round(value))
                centroid[param] = max(p_range[0], min(value, p_range[1]))
            else:
                # For categorical parameters, use the mode
                values = [w.params[param] for w in self._simplex]
                centroid[param] = max(set(values), key=values.count)
        return centroid

    def train(self, instance):
        self._n += 1

        if self.converged:
            self._train_converged(instance)
        else:
            self._train_not_converged(instance)

    def _train_converged(self, instance):
        best_model = self._simplex[0].model
        prediction = best_model.predict(instance)
        self._simplex[0].update(instance.y_value, prediction)
        best_model.train(instance)

        drift_input = float(instance.y_value != prediction)
        self.drift_detector.add_element(drift_input)

        if self.drift_detector.detected_change():
            if self.verbose:
                print("Drift detected, reinitializing simplex")
            self._initialize_simplex()
            self.converged = False
            self._n = 0
            self._old_centroid = None

    def _train_not_converged(self, instance):
        for wrapper in self._simplex:
            prediction = wrapper.model.predict(instance)
            wrapper.update(instance.y_value, prediction)
            wrapper.model.train(instance)

        if self._expanded:
            for wrapper in self._expanded.values():
                prediction = wrapper.model.predict(instance)
                wrapper.update(instance.y_value, prediction)
                wrapper.model.train(instance)

        if self._n % self.grace_period == 0:
            self._update_simplex()

            if self._models_converged():
                if self.verbose:
                    print("Models converged")
                self.converged = True

    def _update_simplex(self):
        self._sort_simplex()
        self._old_centroid = self._calculate_centroid()

        if not self._expanded:
            self._expanded = self._nelder_mead_expansion()

        self._nelder_mead_operators()
        self._expanded = None

    def predict(self, instance):
        return self._simplex[0].model.predict(instance)
    
    def predict_proba(self, instance):
        return self._simplex[0].model.predict_proba(instance)