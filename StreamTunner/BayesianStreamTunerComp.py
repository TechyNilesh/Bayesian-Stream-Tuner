from capymoa.base import Regressor, Classifier
from scipy.stats import norm
from capymoa.evaluation import ClassificationEvaluator, RegressionEvaluator
from capymoa.instance import RegressionInstance
from capymoa.stream import Schema
import numpy as np
import random
from capymoa.drift.detectors import ADWIN
from .BLR import BayesianLinearRegression
from scipy.stats import skew, kurtosis


class BayesianStreamTunerClassifier(Classifier):
    def __init__(self, model=None,            
                 schema=None,
                 random_seed=1,
                 params_range=None,
                 window_size=1000,
                 metric='accuracy',
                 n_models=5,
                 acquisition_function='pi',
                 verbose=False,
                 drift_detection=True,
                 statistical_features=True,
                 delta=0.002):
        super().__init__(schema=schema, random_seed=random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.random_seed = random_seed
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
        self.drift_detection = drift_detection
        self.statistical_features = statistical_features
        self._create_blr_schema()
        self.blr = BayesianLinearRegression(
            schema=self.blr_schema, random_seed=random_seed)
        self._initialize_models()
        self.acquisition_function = acquisition_function
        self.verbose = verbose
        self.drift = ADWIN(delta=delta) if drift_detection else None

    def __str__(self):
        return f'BayesianStreamTunerClassifier({str(self.model.__name__)})'
    
    def _create_blr_schema(self):
        """
        Creates the schema for the Bayesian Linear Regression model with all statistical features.
        """
        feature_names = []

        # Add features for hyperparameters
        for param in self.params_range.keys():
            feature_names.append(f"param_{param}")

        # Add all statistical features if enabled
        if self.statistical_features:
            stat_features = [
                "mean", "std", "range", "min", "max",
                "skewness", "kurtosis",
                "q1", "median", "q3",
            ]
            feature_names.extend([f"stat_{feat}" for feat in stat_features])

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
        """
        Extracts comprehensive statistical features from the data window.
        Returns a fixed-size array of features regardless of the data window state.
        """
        if not self.statistical_features:
            return np.array([])

        n_features = 10  # Total number of statistical features
        
        if len(self.data_window) == 0:
            return np.zeros(n_features)

        window_data = np.vstack(self.data_window)
        flat_data = window_data.flatten()
        
        try:
            # Basic Statistics
            basic_stats = [
                np.mean(flat_data),
                np.std(flat_data),
                np.max(flat_data) - np.min(flat_data),
                np.min(flat_data),
                np.max(flat_data)
            ]

            # Distributional Statistics
            dist_stats = [
                skew(flat_data),
                kurtosis(flat_data),
            ]

            # Quartile Statistics
            q1, med, q3 = np.percentile(flat_data, [25, 50, 75])
            quartile_stats = [q1, med, q3]

            return np.concatenate([basic_stats, dist_stats, quartile_stats])

        except Exception as e:
            if self.verbose:
                print(f"Error in feature extraction: {str(e)}")
            return np.zeros(n_features)

    def _combine_params_and_stats(self, params):
        param_vector = self._params_to_vector(params)
        if self.statistical_features:
            stats = self._extract_statistical_features()
            return np.concatenate([param_vector, stats])
        return param_vector

    def _get_evaluator_metric(self, evaluator, metric):
        try:
            return evaluator.metrics_dict()[metric]
        except Exception:
            return 0.0

    def _acquisition_function(self, X):
        instances = [RegressionInstance.from_array(
            self.blr_schema, x, 0.0) for x in X]
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

        X_sample = np.array([self._combine_params_and_stats(params)
                            for params in self.model_params[1:]])
        y_sample = np.array(performances[1:])

        for x, y in zip(X_sample, y_sample):
            instance = RegressionInstance.from_array(self.blr_schema, x, y)
            self.blr.train(instance)

        models_to_replace = list(range(1, len(self.models)))
        models_to_replace.sort(key=lambda i: performances[i] if i < len(
            performances) else float('-inf'))
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
        if self.statistical_features:
            self.data_window = []
        self.blr = BayesianLinearRegression(
            schema=self.blr_schema, random_seed=self.random_seed)
        self._initialize_models()
        if self.drift_detection:
            self.drift.reset()
        if self.verbose:
            print("**Drift detected. Resetting all models and Drift detector.**")

    def train(self, instance):
        if self.statistical_features:
            self.data_window.append(instance.x)
            if len(self.data_window) > self.window_size:
                self.data_window.pop(0)

        best_prediction = self.models[self._best_model_idx].predict(instance)

        # Handle drift detection if enabled
        if self.drift_detection:
            self.drift.add_element(float(best_prediction != instance.y_index))
            if self.drift.detected_change():
                self._reset_models()
                return

        # Regular training process
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
        X_candidates = self._generate_random_candidates(100)
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
        return self.models[self._best_model_idx].predict(instance)

    def predict_proba(self, instance):
        return self.models[self._best_model_idx].predict_proba(instance)

    def get_number_of_models(self):
        return len(self.models)

class BayesianStreamTunerRegressor(Regressor):
    def __init__(self, model=None,
                 schema=None,
                 random_seed=1,
                 params_range=None,
                 window_size=1000,
                 metric='rmse',  # Default metric for regression
                 n_models=5,
                 acquisition_function='pi',
                 verbose=False,
                 drift_detection=True,
                 statistical_features=True,
                 delta=0.002):
        """
        Initializes the BayesianStreamTunerRegressor.

        Parameters:
            model: The base regression model to be tuned.
            schema: The schema of the data stream.
            random_seed (int): Seed for random number generators.
            params_range (dict): Dictionary defining the range of hyperparameters.
            window_size (int): Size of the data window for statistical feature extraction.
            metric (str): Performance metric to optimize (e.g., 'rmse', 'mae').
            n_models (int): Number of models to maintain.
            acquisition_function (str): Acquisition function to use ('pi', 'ei', 'ucb').
            verbose (bool): If True, prints detailed logs.
            drift_detection (bool): If True, enables drift detection.
            statistical_features (bool): If True, extracts statistical features.
            drift_min_n_instances (int): Minimum number of instances before drift detection starts.
        """
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
        self.drift_detection = drift_detection
        self.statistical_features = statistical_features

        # Create BLR schema based on whether statistical features are extracted
        self._create_blr_schema()
        self.blr = BayesianLinearRegression(
            schema=self.blr_schema, random_seed=random_seed)

        self._initialize_models()
        self.acquisition_function = acquisition_function
        self.verbose = verbose
        # Initialize drift detector if enabled
        self.drift = ADWIN(delta=delta) if drift_detection else None

    def __str__(self):
        return f'BayesianStreamTunerRegressor({str(self.model.__name__)})'

    def _create_blr_schema(self):
        """
        Creates the schema for the Bayesian Linear Regression model with all statistical features.
        """
        feature_names = []

        # Add features for hyperparameters
        for param in self.params_range.keys():
            feature_names.append(f"param_{param}")

        # Add all statistical features if enabled
        if self.statistical_features:
            stat_features = [
                "mean", "std", "range", "min", "max",
                "skewness", "kurtosis",
                "q1", "median", "q3",
            ]
            feature_names.extend([f"stat_{feat}" for feat in stat_features])

        self.blr_schema = Schema.from_custom(
            feature_names=feature_names,
            dataset_name="BLR_Optimization",
            target_attribute_name="performance",
            target_type="numeric"
        )

    def _initialize_models(self):
        """
        Initializes the ensemble of models with random hyperparameters.
        The first model is the default model with no hyperparameter tuning.
        """
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
        """
        Generates a random hyperparameter configuration based on the defined ranges.
        """
        return {param: self._generate(p_data) for param, p_data in self.params_range.items()}

    def _generate(self, p_data):
        """
        Generates a random value for a hyperparameter based on its type and range.
        """
        p_type, p_range = p_data
        if p_type == int:
            return self._rng.randint(p_range[0], p_range[1])
        elif p_type == float:
            return self._rng.uniform(p_range[0], p_range[1])
        else:
            return self._rng.choice(p_range)

    def _params_to_vector(self, params):
        """
        Converts a hyperparameter dictionary to a numerical vector.
        """
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
        """
        Converts a numerical vector back to a hyperparameter dictionary.
        """
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
        """
        Extracts comprehensive statistical features from the data window.
        Returns a fixed-size array of features regardless of the data window state.
        """
        if not self.statistical_features:
            return np.array([])

        n_features = 10  # Total number of statistical features
        
        if len(self.data_window) == 0:
            return np.zeros(n_features)

        window_data = np.vstack(self.data_window)
        flat_data = window_data.flatten()
        
        try:
            # Basic Statistics
            basic_stats = [
                np.mean(flat_data),
                np.std(flat_data),
                np.max(flat_data) - np.min(flat_data),
                np.min(flat_data),
                np.max(flat_data)
            ]

            # Distributional Statistics
            dist_stats = [
                skew(flat_data),
                kurtosis(flat_data),
            ]

            # Quartile Statistics
            q1, med, q3 = np.percentile(flat_data, [25, 50, 75])
            quartile_stats = [q1, med, q3]

            return np.concatenate([basic_stats, dist_stats, quartile_stats])

        except Exception as e:
            if self.verbose:
                print(f"Error in feature extraction: {str(e)}")
            return np.zeros(n_features)

    def _combine_params_and_stats(self, params):
        """
        Combines hyperparameter vectors with statistical features if enabled.
        """
        param_vector = self._params_to_vector(params)
        if self.statistical_features:
            stats = self._extract_statistical_features()
            return np.concatenate([param_vector, stats])
        return param_vector

    def _get_evaluator_metric(self, evaluator, metric):
        """
        Retrieves the specified metric from the evaluator.
        Returns infinity if the metric is unavailable to ensure poor performance.
        """
        try:
            return evaluator.metrics_dict()[metric]
        except Exception:
            return float('inf')

    def _acquisition_function_calc(self, X):
        """
        Calculates the acquisition function values for a set of candidate hyperparameters.
        """
        if not self.blr:
            # If BLR is not initialized, return random acquisition values
            return np.random.rand(X.shape[0])

        instances = [RegressionInstance.from_array(
            self.blr_schema, x, 0.0) for x in X]
        mu, sigma = zip(*[self.blr.predict(instance, with_dist=True)
                         for instance in instances])
        mu, sigma = np.array(mu), np.array(sigma)

        performances = [self._get_evaluator_metric(
            evaluator, self.metric) for evaluator in self.evaluators]
        best_f = np.min(performances) if self.metric in ['rmse', 'mae'] else 0.0  # Assuming minimization for regression

        if self.acquisition_function == 'pi':
            # Probability of Improvement (for minimization)
            z = (best_f - mu) / (sigma + 1e-9)
            return norm.cdf(z)
        elif self.acquisition_function == 'ei':
            # Expected Improvement (for minimization)
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
        """
        Updates the ensemble of models by training the BLR model and replacing underperforming models.
        """
        performances = [self._get_evaluator_metric(
            evaluator, self.metric) for evaluator in self.evaluators]
        if performances:
            self._best_model_idx = np.argmin(performances)  # Best model has the lowest metric

        # Update BLR with hyperparameters and statistical features if enabled
        X_sample = np.array([self._combine_params_and_stats(params)
                             for params in self.model_params[1:]])  # Exclude default model
        y_sample = np.array(performances[1:])  # Exclude default model performance

        if len(X_sample) > 0 and len(y_sample) > 0:
            for x, y in zip(X_sample, y_sample):
                instance = RegressionInstance.from_array(self.blr_schema, x, y)
                self.blr.train(instance)

        # Identify models to replace (e.g., half of the non-default models)
        models_to_replace = list(range(1, len(self.models)))
        models_to_replace.sort(key=lambda i: performances[i] if i < len(
            performances) else float('inf'), reverse=True)
        models_to_replace = models_to_replace[:self.n_models // 2]

        for idx in models_to_replace:
            new_params = self._propose_next_params()
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
        """
        Resets all models, hyperparameters, evaluators, and the BLR optimizer.
        """
        self.models = []
        self.model_params = []
        self.evaluators = []
        self._best_model_idx = 0
        self.data_window = []
        self.blr = BayesianLinearRegression(
            schema=self.blr_schema, random_seed=self.random_seed)
        self._initialize_models()
        if self.drift:
            self.drift.reset()  # Reset the drift detector
        if self.verbose:
            print("**Drift detected. Resetting all models and Drift detector.**")

    def train(self, instance):
        """
        Trains the regressor on a new instance from the data stream.
        """
        if self.statistical_features:
            self.data_window.append(instance.x)
            if len(self.data_window) > self.window_size:
                self.data_window.pop(0)

        best_prediction = self.models[self._best_model_idx].predict(instance)

        # Handle drift detection if enabled
        if self.drift_detection and self.drift:
            error = abs(best_prediction - instance.y_value)
            self.drift.add_element(error)
            if self.drift.detected_change():
                self._reset_models()
                return  # Early return after reset

        # Regular training process
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

    def _propose_next_params(self):
        """
        Proposes the next set of hyperparameters by optimizing the acquisition function.
        """
        X_candidates = self._generate_random_candidates(100)
        X_candidates_full = np.array([self._combine_params_and_stats(
            self._vector_to_params(c)) for c in X_candidates])
        acquisition_values = self._acquisition_function_calc(X_candidates_full)

        if self.acquisition_function in ['pi', 'ei']:
            # For PI and EI, we want to maximize the acquisition function
            best_candidate_idx = np.argmax(acquisition_values)
        elif self.acquisition_function == 'ucb':
            # For Lower Confidence Bound (LCB), we want to minimize
            best_candidate_idx = np.argmin(acquisition_values)
        else:
            raise ValueError(
                "Invalid acquisition function. Choose 'pi', 'ei', or 'ucb'.")

        return self._vector_to_params(X_candidates[best_candidate_idx])

    def _generate_random_candidates(self, n):
        """
        Generates a specified number of random hyperparameter candidates.
        """
        candidates = []
        for _ in range(n):
            params = self._random_config()
            candidates.append(self._params_to_vector(params))
        return np.array(candidates)

    def predict(self, instance):
        """
        Makes a prediction using the best-performing model.
        """
        return self.models[self._best_model_idx].predict(instance)

    def predict_proba(self, instance):
        """
        Returns prediction probabilities using the best-performing model.
        Note: Not typically used in regression. Included for consistency.
        """
        return self.models[self._best_model_idx].predict_proba(instance)

    def get_number_of_models(self):
        """
        Returns the number of models in the ensemble.
        """
        return len(self.models)