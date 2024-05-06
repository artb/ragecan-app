import logging
import optuna
import time
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from .utils import return_metric_dict, FSOptimizationResult
from .constants import N_JOBS_FS, CV_FOLDS, SVM_KERNEL_LIST, PRUNER, SVM_C_LIST, RF_ESTIMATORS_FLOOR,RF_ESTIMATORS_CELING,RF_DEPTH_FLOOR , RF_DEPTH_CELING
from mrmr import mrmr_classif
from icecream import ic


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseFeatureSelection:
	def __init__(self, metric, selection_method, execution_classifier='SVM'):
		self.metric = metric
		self.selection_method = selection_method
		self.execution_classifier = execution_classifier
		self.desired_dimension = None
		self.scaler = MinMaxScaler()
		self.best_classifier = None  
		self.best_features = None
		self.name = None

	def objective(self, trial, X, y):
		logging.info('Starting Base Feature Selection trial objective evaluation')
		k = trial.suggest_int('n_features_to_select', self.desired_dimension[0], self.desired_dimension[1])
		X_selected = self.select_features(X, y, k)
		classifier = self.get_classifier(trial)
		logging.info(f"Starting cross-validation evaluation on the train")
		score = cross_val_score(classifier, X_selected, y, cv=CV_FOLDS, scoring=self.metric, n_jobs=1).mean()
		ic(f'Completed trial with score: {score}')
		return score

	def get_classifier(self, trial):
		if self.execution_classifier == 'SVM':
			C = trial.suggest_categorical('svc_c', SVM_C_LIST)
			kernel = trial.suggest_categorical('svc_kernel', SVM_KERNEL_LIST)
			classifier = SVC(C=C, kernel=kernel)
			ic(f'SVM Classifier with C={C}, kernel={kernel}')
		elif self.execution_classifier == 'RandomForest':
			n_estimators = trial.suggest_int('rf_n_estimators', RF_ESTIMATORS_FLOOR, RF_ESTIMATORS_CELING)
			max_depth = trial.suggest_int('rf_max_depth', RF_DEPTH_FLOOR , RF_DEPTH_CELING)
			classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
			ic(f'RandomForest Classifier with n_estimators={n_estimators}, max_depth={max_depth}')
		elif self.execution_classifier == 'NaiveBayes':
			classifier = GaussianNB()
			ic('NaiveBayes Classifier Selected!')
		else:
			logging.error('Unsupported classifier type')
			raise ValueError("Unsupported classifier type")
		return classifier

	def select_features(self, X, y, n_features):
		selector = SelectKBest(self.selection_method, k=n_features)
		return selector.fit_transform(X, y)

	def optimize(self, X, y, desired_dimension, n_trials):
		self.desired_dimension = desired_dimension
		study_name = f"{self.__class__.__name__.upper()}_{self.execution_classifier}"
		logging.info(f'Starting {self.__class__.__name__} optimization process')
		study = optuna.create_study(study_name=study_name, direction="maximize", pruner=PRUNER)
		study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials, n_jobs=N_JOBS_FS)
		ic(f"Optimization completed for {self.execution_classifier}. Best parameters: {study.best_params}")
		self.best_classifier = self.get_classifier(study.best_trial)
		ic(f"Best classifier is: {self.best_classifier}\n")
		self.best_features = study.best_params['n_features_to_select']
		ic(f"Optimize returning this study: {study}")
		return study.best_params

	def evaluate_on_test(self, X_train, y_train, X_test, y_test):
		logging.info(f'Evaluating {self.__class__.__name__} on test dataset')
		selector = SelectKBest(self.selection_method, k=self.best_features)
		X_train_selected = selector.fit_transform(X_train, y_train)
		X_test_selected = selector.transform(X_test)
		self.best_classifier.fit(X_train_selected, y_train)
		y_train_pred = self.best_classifier.predict(X_train_selected)
		y_pred = self.best_classifier.predict(X_test_selected)
		logging.info(f"On evaluation the y has been predicted")
		train_metrics = return_metric_dict(y_train, y_train_pred)
		test_metrics = return_metric_dict(y_test, y_pred)
		ic(f"Test dataset evaluation score: {test_metrics}")
		return (X_train_selected, X_test_selected), (train_metrics, test_metrics)

	def execute_selection(self, X_train, y_train, X_test, y_test, desired_dimension, n_trials):
		logging.info(f'Starting  {self.__class__.__name__} feature selection execution')
		ic(f"Starting execution of selection with {desired_dimension}\n")
		start_time = time.time()
		best_params = self.optimize(X_train, y_train, desired_dimension, n_trials)
		end_time = time.time()
		logging.warning(f"Optimization time: {(end_time - start_time) / 60:.2f} minutes.")
		compressed_xs, fs_metrics = self.evaluate_on_test(X_train, y_train, X_test, y_test)
		opt_res = FSOptimizationResult(self.name, best_params, fs_metrics[0], fs_metrics[1], compressed_xs[0], compressed_xs[1])
		ic(f"Basic Filter selection ended, returning: {opt_res}")
		return opt_res

class AnovaFeatureSelection(BaseFeatureSelection):
	def __init__(self, metric, execution_classifier='SVM'):
		super().__init__(metric, f_classif, execution_classifier=execution_classifier)
		self.name = 'AnovaFeatureSelectionMethod'
		logging.info('AnovaFeatureSelection Instanced')

class ChiSquareFeatureSelection(BaseFeatureSelection):
	def __init__(self, metric, execution_classifier='SVM'):
		super().__init__(metric, chi2, execution_classifier=execution_classifier)
		self.name = 'ChiSquareFeatureSelection'
		logging.info('ChiSquareFeatureSelection Instanced')

	def execute_selection(self, X_train, y_train, X_test, y_test, desired_dimension, n_trials):
		logging.info('Starting ChiSquare feature selection execution')
		ic(f"Starting execution of selection with {desired_dimension}\n")
		X_train_scaled = self.scaler.fit_transform(X_train)
		X_test_scaled = self.scaler.transform(X_test)
		start_time = time.time()
		best_params = self.optimize(X_train_scaled, y_train, desired_dimension, n_trials)
		end_time = time.time()
		ic(f"Optimization time: {(end_time - start_time) / 60:.2f} minutes.")
		compressed_xs, fs_metrics = self.evaluate_on_test(X_train_scaled, y_train, X_test_scaled, y_test)
		opt_res = FSOptimizationResult(self.name, best_params, fs_metrics[0], fs_metrics[1], compressed_xs[0], compressed_xs[1])
		return opt_res

class MutualInformationFeatureSelection(BaseFeatureSelection):
	def __init__(self, metric, execution_classifier='SVM'):
		super().__init__(metric, mutual_info_classif, execution_classifier=execution_classifier)
		self.name = 'MutualInformationFeatureSelection'
		logging.info('MutualInformationFeatureSelection Instanced')



class MRMRFeatureSelection(BaseFeatureSelection):
	def __init__(self, metric, execution_classifier='SVM'):
		super().__init__(metric, None, execution_classifier=execution_classifier)
		self.name = 'MRMRFeatureSelectionSelection'
		self.label_encoder = LabelEncoder()
		logging.info('MRMRFeatureSelection Instanced')

	def select_features(self, X, y, n_features):
		X = pd.DataFrame(X)
		y_encoded = self.label_encoder.fit_transform(y)
		y = pd.Series(y_encoded)
		X = X.reset_index(drop=True)
		y = y.reset_index(drop=True)
		selected_features = mrmr_classif(X=X, y=y, K=n_features)
		column_names = X.columns[selected_features]
		return X[column_names]

	def evaluate_on_test(self, X_train, y_train, X_test, y_test):
		logging.info(f'Evaluating {self.__class__.__name__} on test dataset')
		X_train_selected = self.select_features(self.selection_method, k=self.best_features)
		#selector.fit_transform(X_train, y_train)
		#THIS IS WRONG!!
		X_test_selected = self.select_features(self.selection_method, k=self.best_features)
		self.best_classifier.fit(X_train_selected, y_train)
		y_train_pred = self.best_classifier.predict(X_train_selected)
		y_pred = self.best_classifier.predict(X_test_selected)
		logging.info(f"On evaluation the y has been predicted")
		train_metrics = return_metric_dict(y_train, y_train_pred)
		test_metrics = return_metric_dict(y_test, y_pred)
		logging.info(f"Test dataset evaluation score: {test_metrics}")
		return (X_train_selected, X_test_selected), (train_metrics, test_metrics)

	def execute_selection(self, X_train, y_train, X_test, y_test, desired_dimension, n_trials):
		logging.info(f'Starting  {self.__class__.__name__} feature selection execution')
		start_time = time.time()
		best_params = self.optimize(X_train, y_train, desired_dimension, n_trials)
		end_time = time.time()
		logging.warning(f"Optimization time: {(end_time - start_time) / 60:.2f} minutes.")
		compressed_xs, fs_metrics = self.evaluate_on_test(X_train, y_train, X_test, y_test)
		opt_res = FSOptimizationResult(self.name, best_params, fs_metrics[0], fs_metrics[1], compressed_xs[0], compressed_xs[1])
		return opt_res