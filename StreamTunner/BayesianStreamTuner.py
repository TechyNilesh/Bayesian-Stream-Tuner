from capymoa.base import Regressor, Classifier
from scipy.stats import norm
from capymoa.evaluation import ClassificationEvaluator, RegressionEvaluator
from capymoa.instance import RegressionInstance
from capymoa.stream import Schema
import numpy as np
from scipy.stats import skew, kurtosis
import random
from capymoa.drift.detectors import ADWIN
from .BLR import BayesianLinearRegression


class BayesianStreamTunerClassifier(Classifier):
    """A Bayesian optimizer for online hyperparameter tuning of classification models.

    This class implements an online Bayesian optimization approach for tuning hyperparameters
    of streaming classification models. It maintains multiple model instances with different
    hyperparameter configurations and uses Bayesian optimization to propose new configurations.

    Parameters
    ----------
    model : class
        The base model class to be tuned.
    schema : Schema, optional
        The schema defining the data structure.
    random_seed : int, default=1
        Random seed for reproducibility.
    params_range : dict, optional
        Dictionary defining the hyperparameter search space.
        Format: {param_name: (type, range)}
    window_size : int, default=1000
        Size of the sliding window for performance evaluation.
    metric : str, default='accuracy'
        Metric used for model evaluation.
    n_models : int, default=5
        Number of concurrent models to maintain.
    acquisition_function : {'pi', 'ei', 'ucb'}, default='pi'
        The acquisition function for Bayesian optimization.
    verbose : bool, default=False
        If True, prints detailed information during optimization.
    delta : float, default=0.002
        Delta parameter for the ADWIN drift detector.
    Attributes
    ----------
    models : list
        List of active model instances.
    model_params : list
        List of hyperparameter configurations for each model.
    _best_model_idx : int
        Index of the currently best performing model.
    """

    def __init__(self, model=None,
                 schema=None,
                 random_seed=1,
                 params_range=None,
                 window_size=1000,
                 metric='accuracy',
                 n_models=5,
                 acquisition_function='pi',
                 verbose=False,
                 delta=0.002):
        super().__init__(schema=schema, random_seed=random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if model is None:
            raise ValueError("A valid model must be provided.")
        self._rng = random.Random(random_seed)
        self.model = model
        self.window_size = window_size
        self.metric = metric
        self.params_range = params_range if params_range is not None else {}
        self.n_models = n_models
        self.models = []
        self.model_params = []
        self.evaluators = []
        self._best_model_idx = 0
        self.sample_count = 0
        self.data_window = []
        self._create_blr_schema()
        self.blr = BayesianLinearRegression(
            schema=self.blr_schema, random_seed=random_seed)
        self._initialize_models()
        self.acquisition_function = acquisition_function
        self.verbose = verbose
        # Initialize drift detector
        self.drift = ADWIN(delta=delta)
        
        

    def __str__(self):
        return f'BayesianStreamTunerClassifier({str(self.model.__name__)})'

    def _create_blr_schema(self):
        feature_names = []

        # Add features for hyperparameters
        for param in self.params_range.keys():
            feature_names.append(f"param_{param}")

        # Add features for statistical features
        stat_features = ["mean", "std", "median", "range",
                         "q1", "q3", "min", "max", "skewness", "kurtosis"]
        feature_names.extend([f"stat_{feat}" for feat in stat_features])

        # Create the schema
        self.blr_schema = Schema.from_custom(
            feature_names=feature_names,
            dataset_name="BLR_Optimization",
            target_attribute_name="performance",
            target_type="numeric"
        )

    def _initialize_models(self):
        # Create default model
        default_model = self.model(schema=self.schema)
        self.models.append(default_model)
        self.model_params.append({})  # Empty dict for default params
        self.evaluators.append(ClassificationEvaluator(
            schema=self.schema, window_size=self.window_size))

        # Create additional models with random hyperparameters
        for _ in range(self.n_models - 1):
            params = self._random_config()
            model = self.model(schema=self.schema, **params)
            self.models.append(model)
            self.model_params.append(params)
            self.evaluators.append(ClassificationEvaluator(
                schema=self.schema, window_size=self.window_size))

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

    def _params_to_vector(self, params):
        vector = []
        for param, p_data in self.params_range.items():
            p_type, p_range = p_data
            if p_type in (int, float):
                vector.append(float(params.get(param, p_range[0])))
            else:
                vector.append(
                    float(p_range.index(params.get(param, p_range[0]))))
        return np.array(vector)

    def _vector_to_params(self, vector):
        params = {}
        idx = 0
        for param, p_data in self.params_range.items():
            p_type, p_range = p_data
            if p_type == int:
                params[param] = int(round(vector[idx]))
            elif p_type == float:
                params[param] = float(vector[idx])
            else:
                params[param] = p_range[int(round(vector[idx]))]
            idx += 1
        return params

    def _extract_statistical_features(self):
        if len(self.data_window) == 0:
            # Corrected the size to match the number of features
            return np.zeros(10)  # Updated to 10 features
        window_data = np.vstack(self.data_window)
        # Compute statistics over all data points
        flat_data = window_data.flatten()
        return np.array([
            np.mean(flat_data),
            np.std(flat_data),
            np.median(flat_data),
            np.max(flat_data) - np.min(flat_data),  # Range
            np.percentile(flat_data, 25),  # 1st quartile
            np.percentile(flat_data, 75),  # 3rd quartile
            np.min(flat_data),  # Minimum
            np.max(flat_data),   # Maximum
            skew(flat_data),  # Skewness
            kurtosis(flat_data)  # Kurtosis
        ])

    def _combine_params_and_stats(self, params):
        param_vector = self._params_to_vector(params)
        stats = self._extract_statistical_features()
        return np.concatenate([param_vector, stats])

    def _get_evaluator_metric(self, evaluator, metric):
        try:
            return evaluator[metric]
        except Exception:
            # Return a default value if the metric is not available
            return 0.0

    def _acquisition_function(self, X):
        instances = [RegressionInstance.from_array(
            self.blr_schema, x, 0.0) for x in X]  # Use 0.0 as a dummy y_value
        mu, sigma = zip(*[self.blr.predict(instance, with_dist=True)
                        for instance in instances])
        mu, sigma = np.array(mu), np.array(sigma)

        performances = [self._get_evaluator_metric(
            evaluator, self.metric) for evaluator in self.evaluators]
        best_f = np.max(performances) if performances else 0.0

        if self.acquisition_function == 'pi':
            # Probability of Improvement
            z = (mu - best_f) / (sigma + 1e-9)
            return norm.cdf(z)
        elif self.acquisition_function == 'ei':
            # Expected Improvement
            z = (mu - best_f) / (sigma + 1e-9)
            return (mu - best_f) * norm.cdf(z) + sigma * norm.pdf(z)
        elif self.acquisition_function == 'ucb':
            # Upper Confidence Bound
            # kappa = 2.0  # Exploration-exploitation trade-off parameter
            kappa = max(
                0.1, 2.0 * (1 - self.sample_count / (10 * self.window_size)))
            return mu + kappa * sigma
        else:
            raise ValueError(
                "Invalid acquisition function. Choose 'pi', 'ei', or 'ucb'.")

    def _update_models(self):
        performances = [self._get_evaluator_metric(
            evaluator, self.metric) for evaluator in self.evaluators]
        if performances:
            self._best_model_idx = np.argmax(performances)

        # Update BLR with both hyperparameters and statistical features
        X_sample = np.array([self._combine_params_and_stats(params)
                            for params in self.model_params[1:]])  # Exclude default model
        # Exclude default model performance
        y_sample = np.array(performances[1:])

        for x, y in zip(X_sample, y_sample):
            instance = RegressionInstance.from_array(self.blr_schema, x, y)
            self.blr.train(instance)

        # Always keep the default model (index 0)
        models_to_replace = list(range(1, len(self.models)))
        models_to_replace.sort(key=lambda i: performances[i] if i < len(
            performances) else float('-inf'))
        # Replace half of the non-default models
        models_to_replace = models_to_replace[:self.n_models // 2]

        for idx in models_to_replace:
            new_params = self._propose_next_params()
            self.models[idx] = self.model(schema=self.schema, **new_params)
            self.model_params[idx] = new_params
            self.evaluators[idx] = ClassificationEvaluator(
                schema=self.schema, window_size=self.window_size)

        if self.verbose:
            print(f"Sample Processed: {self.sample_count}")
            print(
                f"Best model index: {self._best_model_idx}, Score: {performances[self._best_model_idx] if performances else 'N/A'}")
            print(
                f"Default model Score: {performances[0] if performances else 'N/A'}")
            if self._best_model_idx != 0 and performances:
                print(
                    f"Best model hyperparameters: {self.model_params[self._best_model_idx]}")
            print(
                f"Replaced {len(models_to_replace)} models with new hyperparameters from BLR optimization")

    def _reset_models(self):
        self.models = []
        self.model_params = []
        self.evaluators = []
        self._best_model_idx = 0
        # self.sample_count = 0
        self.data_window = []
        self.blr = BayesianLinearRegression(
            schema=self.blr_schema, random_seed=self.random_seed)
        self._initialize_models()
        self.drift.reset()  # Reset the drift detector
        if self.verbose:
            print("**Drift detected. Resetting all models and Drift detector.**")

    def train(self, instance):
        """Train the model on a new instance.

        Parameters
        ----------
        instance : Instance
            The training instance containing features and target.
        """
        self.data_window.append(instance.x)
        if len(self.data_window) > self.window_size:
            self.data_window.pop(0)

        best_prediction = self.models[self._best_model_idx].predict(instance)

        # Update Drift with the prediction error
        self.drift.add_element(float(best_prediction != instance.y_index))

        # Check if drift is detected
        if self.drift.detected_change():
            self._reset_models()
        else:
            for idx, model in enumerate(self.models):
                try:
                    prediction = model.predict(instance)
                    self.evaluators[idx].update(instance.y_index, prediction)
                    model.train(instance)
                except Exception as e:
                    if self.verbose:
                        print(f"Error training model {idx}: {str(e)}")

            self.sample_count += 1

            if self.sample_count % self.window_size == 0:
                self._update_models()

    def _propose_next_params(self):
        X_candidates = self._generate_random_candidates(
            100)  # Generate 100 random candidates
        X_candidates_full = np.array([self._combine_params_and_stats(
            self._vector_to_params(c)) for c in X_candidates])
        acquisition_values = self._acquisition_function(X_candidates_full)
        best_candidate_idx = np.argmax(acquisition_values)
        return self._vector_to_params(X_candidates[best_candidate_idx])

    def _generate_random_candidates(self, n):
        candidates = []
        for _ in range(n):
            params = self._random_config()
            candidates.append(self._params_to_vector(params))
        return np.array(candidates)

    def predict(self, instance):
        """Predict the class for an instance using the best model.

        Parameters
        ----------
        instance : Instance
            The instance to make predictions for.

        Returns
        -------
        int
            Predicted class label.
        """
        return self.models[self._best_model_idx].predict(instance)

    def predict_proba(self, instance):
        return self.models[self._best_model_idx].predict_proba(instance)

    def get_number_of_models(self):
        return len(self.models)


class BayesianStreamTunerRegressor(Regressor):
    """A Bayesian optimizer for online hyperparameter tuning of regression models.

    This class implements an online Bayesian optimization approach for tuning hyperparameters
    of streaming regression models. It maintains multiple model instances with different
    hyperparameter configurations and uses Bayesian optimization to propose new configurations.

    Parameters
    ----------
    model : class
        The base model class to be tuned.
    schema : Schema, optional
        The schema defining the data structure.
    random_seed : int, default=1
        Random seed for reproducibility.
    params_range : dict, optional
        Dictionary defining the hyperparameter search space.
        Format: {param_name: (type, range)}
    window_size : int, default=1000
        Size of the sliding window for performance evaluation.
    metric : str, default='rmse'
        Metric used for model evaluation.
    n_models : int, default=5
        Number of concurrent models to maintain.
    acquisition_function : {'pi', 'ei', 'ucb'}, default='pi'
        The acquisition function for Bayesian optimization.
    verbose : bool, default=False
        If True, prints detailed information during optimization.
    delta : float, default=0.002
        Delta parameter for the ADWIN drift detector.
    Attributes
    ----------
    models : list
        List of active model instances.
    model_params : list
        List of hyperparameter configurations for each model.
    _best_model_idx : int
        Index of the currently best performing model.
    """

    def __init__(self, model=None,
                 schema=None,
                 random_seed=1,
                 params_range=None,
                 window_size=1000,
                 metric='rmse',  # Changed default to 'mae'
                 n_models=5,
                 acquisition_function='pi',
                 verbose=False,
                 delta=0.002):
        super().__init__(schema=schema, random_seed=random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if model is None:
            raise ValueError("A valid model must be provided.")
        self._rng = random.Random(random_seed)
        self.model = model
        self.window_size = window_size
        self.metric = metric
        self.params_range = params_range if params_range is not None else {}
        self.n_models = n_models
        self.models = []
        self.model_params = []
        self.evaluators = []
        self._best_model_idx = 0#np.random.randint(n_models)
        self.sample_count = 0
        self.data_window = []
        self._create_blr_schema()
        self.blr = BayesianLinearRegression(
            schema=self.blr_schema, random_seed=random_seed)
        self._initialize_models()
        self.acquisition_function = acquisition_function
        self.verbose = verbose
        self.drift = ADWIN(delta=delta)

    def __str__(self):
        return f'BayesianStreamTunerRegressor({str(self.model.__name__)})'

    def _create_blr_schema(self):
        feature_names = []

        # Add features for hyperparameters
        for param in self.params_range.keys():
            feature_names.append(f"param_{param}")

        # Add features for statistical features
        stat_features = ["mean", "std", "median", "range",
                         "q1", "q3", "min", "max", "skewness", "kurtosis"]
        feature_names.extend([f"stat_{feat}" for feat in stat_features])

        # Create the schema
        self.blr_schema = Schema.from_custom(
            feature_names=feature_names,
            dataset_name="BLR_Optimization",
            target_attribute_name="performance",
            target_type="numeric"
        )

    def _initialize_models(self):
        # Create default model
        default_model = self.model(schema=self.schema)
        self.models.append(default_model)
        self.model_params.append({})  # Empty dict for default params
        self.evaluators.append(RegressionEvaluator(
            schema=self.schema, window_size=self.window_size))

        # Create additional models with random hyperparameters
        for _ in range(self.n_models - 1):
            params = self._random_config()
            model = self.model(schema=self.schema, **params)
            self.models.append(model)
            self.model_params.append(params)
            self.evaluators.append(RegressionEvaluator(
                schema=self.schema, window_size=self.window_size))

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

    def _params_to_vector(self, params):
        vector = []
        for param, p_data in self.params_range.items():
            p_type, p_range = p_data
            if p_type in (int, float):
                vector.append(float(params.get(param, p_range[0])))
            else:
                vector.append(
                    float(p_range.index(params.get(param, p_range[0]))))
        return np.array(vector)

    def _vector_to_params(self, vector):
        params = {}
        idx = 0
        for param, p_data in self.params_range.items():
            p_type, p_range = p_data
            if p_type == int:
                params[param] = int(round(vector[idx]))
            elif p_type == float:
                params[param] = float(vector[idx])
            else:
                params[param] = p_range[int(round(vector[idx]))]
            idx += 1
        return params

    def _extract_statistical_features(self):
        if len(self.data_window) == 0:
            # Corrected the size to match the number of features
            return np.zeros(10)  # Updated to 10 features
        window_data = np.vstack(self.data_window)
        # Compute statistics over all data points
        flat_data = window_data.flatten()
        return np.array([
            np.mean(flat_data),
            np.std(flat_data),
            np.median(flat_data),
            np.max(flat_data) - np.min(flat_data),  # Range
            np.percentile(flat_data, 25),  # 1st quartile
            np.percentile(flat_data, 75),  # 3rd quartile
            np.min(flat_data),  # Minimum
            np.max(flat_data),   # Maximum
            skew(flat_data),  # Skewness
            kurtosis(flat_data)  # Kurtosis
        ])

    def _combine_params_and_stats(self, params):
        param_vector = self._params_to_vector(params)
        stats = self._extract_statistical_features()
        return np.concatenate([param_vector, stats])

    def _get_evaluator_metric(self, evaluator, metric):
        try:
            # Simply return the score for the provided metric
            return evaluator.metrics_dict()[metric]
        except Exception:
            # Return a high value (inf) if the metric is not available to ensure poor performance for comparison
            return float('inf')

    def _acquisition_function(self, X):
        instances = [RegressionInstance.from_array(
            self.blr_schema, x, 0.0) for x in X]
        mu, sigma = zip(*[self.blr.predict(instance, with_dist=True)
                        for instance in instances])
        mu, sigma = np.array(mu), np.array(sigma)

        performances = [self._get_evaluator_metric(
            evaluator, self.metric) for evaluator in self.evaluators]
        best_f = np.min(performances) if performances else float(
            'inf')  # Changed to min

        if self.acquisition_function == 'pi':
            # Probability of Improvement (modified for minimization)
            z = (best_f - mu) / (sigma + 1e-9)
            return norm.cdf(z)
        elif self.acquisition_function == 'ei':
            # Expected Improvement (modified for minimization)
            z = (best_f - mu) / (sigma + 1e-9)
            return (best_f - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        elif self.acquisition_function == 'ucb':
            # Lower Confidence Bound (for minimization)
            kappa = max(
                0.1, 2.0 * (1 - self.sample_count / (10 * self.window_size)))
            return mu - kappa * sigma
        else:
            raise ValueError(
                "Invalid acquisition function. Choose 'pi', 'ei', or 'ucb'.")

    def _update_models(self):
        # Retrieve performances from all models
        performances = [self._get_evaluator_metric(
            evaluator, self.metric) for evaluator in self.evaluators]
        if performances:
            # Use np.argmin to select the best model (lower is better for MAE/RMSE)
            self._best_model_idx = np.argmin(performances)

        # Update Bayesian Linear Regression (BLR) model
        X_sample = np.array([self._combine_params_and_stats(params)
                            for params in self.model_params[1:]])
        y_sample = np.array(performances[1:])

        for x, y in zip(X_sample, y_sample):
            instance = RegressionInstance.from_array(self.blr_schema, x, y)
            self.blr.train(instance)

        # Replace models with poor performance
        models_to_replace = list(range(1, len(self.models)))
        models_to_replace.sort(key=lambda i: performances[i] if i < len(
            performances) else float('inf'), reverse=True)
        models_to_replace = models_to_replace[:self.n_models // 2]

        for idx in models_to_replace:
            new_params = self._propose_next_params(X_sample)
            self.models[idx] = self.model(schema=self.schema, **new_params)
            self.model_params[idx] = new_params
            self.evaluators[idx] = RegressionEvaluator(
                schema=self.schema, window_size=self.window_size)

        if self.verbose:
            print(f"Sample Processed: {self.sample_count}")
            print(
                f"Best model index: {self._best_model_idx}, Score: {performances[self._best_model_idx] if performances else 'N/A'}")
            print(
                f"Default model Score: {performances[0] if performances else 'N/A'}")
            if self._best_model_idx != 0 and performances:
                print(
                    f"Best model hyperparameters: {self.model_params[self._best_model_idx]}")
            print(
                f"Replaced {len(models_to_replace)} models with new hyperparameters from BLR optimization")

    def _reset_models(self):
        self.models = []
        self.model_params = []
        self.evaluators = []
        self._best_model_idx = 0
        # self.sample_count = 0
        self.data_window = []
        self.blr = BayesianLinearRegression(
            schema=self.blr_schema, random_seed=self.random_seed)
        self._initialize_models()
        self.drift.reset()  # Reset the ADWIN detector
        if self.verbose:
            print("**Drift detected. Resetting all models and Drift detector.**")

    def train(self, instance):
        """Train the model on a new instance.

        Parameters
        ----------
        instance : Instance
            The training instance containing features and target.
        """
        self.data_window.append(instance.x)
        
        if len(self.data_window) > self.window_size:
            self.data_window.pop(0)

        best_prediction = self.models[self._best_model_idx].predict(instance)

        # Update Drift with the prediction error
        self.drift.add_element(abs(best_prediction - instance.y_value))

        # Check if drift is detected
        if self.drift.detected_change():
            
            self._reset_models()
        
        else:
            for idx, model in enumerate(self.models):
                try:
                    prediction = model.predict(instance)
                    self.evaluators[idx].update(instance.y_value, prediction)
                    model.train(instance)
                except Exception as e:
                    if self.verbose:
                        print(f"Error training model {idx}: {str(e)}")

            self.sample_count += 1

            if self.sample_count % self.window_size == 0:
                self._update_models()

    def _propose_next_params(self, X_sample):
        X_candidates = self._generate_random_candidates(
            100)  # Generate 100 random candidates
        X_candidates_full = np.array([self._combine_params_and_stats(
            self._vector_to_params(c)) for c in X_candidates])
        acquisition_values = self._acquisition_function(X_candidates_full)

        if self.acquisition_function in ['pi', 'ei']:
            # For PI and EI, we still want to maximize the acquisition function
            best_candidate_idx = np.argmax(acquisition_values)
        elif self.acquisition_function == 'ucb':
            # For LCB (previously UCB), we want to minimize
            best_candidate_idx = np.argmin(acquisition_values)
        else:
            raise ValueError(
                "Invalid acquisition function. Choose 'pi', 'ei', or 'ucb'.")

        return self._vector_to_params(X_candidates[best_candidate_idx])

    def _generate_random_candidates(self, n):
        candidates = []
        for _ in range(n):
            params = self._random_config()
            candidates.append(self._params_to_vector(params))
        return np.array(candidates)

    def predict(self, instance):
        """Predict the value for an instance using the best model.

        Parameters
        ----------
        instance : Instance
            The instance to make predictions for.

        Returns
        -------
        float
            Predicted value.
        """
        return self.models[self._best_model_idx].predict(instance)

    def predict_proba(self, instance):
        return self.models[self._best_model_idx].predict_proba(instance)

    def get_number_of_models(self):
        return len(self.models)