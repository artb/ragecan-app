import optuna
import logging
import time
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from .utils import return_metric_dict, FSOptimizationResult
from .constants import N_JOBS_WP, SVM_KERNEL_LIST, PRUNER, SVM_C_LIST, RF_ESTIMATORS_FLOOR,RF_ESTIMATORS_CELING,RF_DEPTH_FLOOR , RF_DEPTH_CELING


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RecursiveFeatureEliminationBase:
	def __init__(self, metric='accuracy', execution_estimator='SVM', execution_classifier='SVM'):
		self.metric = metric
		self.desired_dimension = None
		self.execution_estimator = execution_estimator
		self.execution_classifier = execution_classifier
		self.scaler = MinMaxScaler()
		self.best_estimator = None
		self.best_classifier = None
		self.best_optimization_score = -1
		self.best_k = 0

	def _get_classifier(self, trial):
		if self.execution_classifier == 'SVM':
			C = trial.suggest_categorical('svc_c', SVM_C_LIST)
			kernel = trial.suggest_categorical('svc_kernel', SVM_KERNEL_LIST)
			classifier = SVC(C=C, kernel=kernel)
			logging.info(f'SVM Classifier with C={C}, kernel={kernel}')
		elif self.execution_classifier == 'RandomForest':
			n_estimators = trial.suggest_int('rf_n_estimators', RF_ESTIMATORS_FLOOR, RF_ESTIMATORS_CELING)
			max_depth = trial.suggest_int('rf_max_depth', RF_DEPTH_FLOOR , RF_DEPTH_CELING)
			classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
			logging.info(f'RandomForest Classifier with n_estimators={n_estimators}, max_depth={max_depth}')
		elif self.execution_classifier == 'NaiveBayes':
			classifier = GaussianNB()
			logging.info('NaiveBayes Classifier')
		else:
			logging.error('Unsupported classifier type')
			raise ValueError("Unsupported classifier type")
		return classifier
	
	def _get_estimator(self, trial):
		if self.execution_estimator == 'SVM':
			C = trial.suggest_categorical('svc_c', SVM_C_LIST)
			kernel = 'linear'
			classifier = SVC(C=C, kernel=kernel)
		elif self.execution_estimator == 'RandomForest':
			n_estimators = trial.suggest_int('rf_n_estimators', RF_ESTIMATORS_FLOOR, RF_ESTIMATORS_CELING)
			max_depth = trial.suggest_int('rf_max_depth', RF_DEPTH_FLOOR , RF_DEPTH_CELING)
			classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
		else:
			raise ValueError("Unsupported classifier type")
		return classifier

	def objective(self, trial, X, y):
		estimator = self._get_estimator(trial)
		logging.info('Starting RFE Base trial objective evaluation')
		k = trial.suggest_int('n_features_to_select', self.desired_dimension[0], self.desired_dimension[1])
		logging.info(f'Creating RFE search with dimension={k}')
		rfe = RFE(estimator=estimator, n_features_to_select=k)

		X_compressed = rfe.fit_transform(X, y)
		classifier = self._get_classifier(trial)
		score = cross_val_score(classifier, X_compressed, y, cv=5, scoring=self.metric).mean()

		if score > self.best_optimization_score:
			self.best_optimization_score = score
			self.best_estimator = estimator
			self.best_k = k
		logging.info(f'Completed trial with score: {score}')
		return score

	def optimize(self, X, y, n_trials):
		logging.info('Starting RFE Base optimization process')
		study_name = f"{self.__class__.__name__.upper()}_{self.execution_classifier}"
		study = optuna.create_study(study_name=study_name, direction="maximize", pruner=PRUNER)
		study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials, n_jobs=N_JOBS_WP)
		logging.info(f"Optimization completed. Best score: {study.best_value}")
		self.best_classifier = self._get_estimator(study.best_trial)
		
		return study.best_params

	def evaluate_on_test(self, X_train, y_train, X_test, y_test):
		logging.info('Evaluating Best RFE Base on test dataset')
		rfe = RFE(estimator=self.best_estimator, n_features_to_select=self.best_k)
		X_train_selected = rfe.fit_transform(X_train, y_train)
		X_test_selected = rfe.transform(X_test)

		self.best_classifier.fit(X_train_selected, y_train)
		y_train_pred = self.best_classifier.predict(X_train_selected)
		# Predict on the selected test features
		y_pred = self.best_classifier.predict(X_test_selected)
		logging.info(f"On evaluation the y has been predicted")
		train_metrics = return_metric_dict(y_train, y_train_pred)
		test_metrics = return_metric_dict(y_test, y_pred)
		logging.info(f"Test dataset evaluation score: {test_metrics}")
		return (X_train_selected, X_test_selected), (train_metrics, test_metrics)


	def execute_selection(self, X_train, y_train, X_test, y_test, desired_dimension, n_trials):
		self.desired_dimension = desired_dimension
		start_time = time.time()
		best_params = self.optimize(X_train, y_train, n_trials)
		end_time = time.time()
		logging.warning(f"Optimization time: {(end_time - start_time) / 60:.2f} minutes.")
		compressed_xs, fs_metrics = self.evaluate_on_test(X_train, y_train, X_test, y_test)
		opt_res = FSOptimizationResult(self.name, best_params, fs_metrics[0], fs_metrics[1], compressed_xs[0], compressed_xs[1])
		return opt_res


class SVMRecursiveFeatureElimination(RecursiveFeatureEliminationBase):
	def __init__(self, metric, execution_classifier='SVM'):
		super().__init__(metric=metric, execution_estimator='SVM', execution_classifier=execution_classifier)
		self.name = 'SVMRecursiveFeatureElimination'
		logging.info('SVM RFE Instanced')

class RFRecursiveFeatureElimination(RecursiveFeatureEliminationBase):
	def __init__(self, metric, execution_classifier='SVM'):
		super().__init__(metric=metric, execution_estimator='RandomForest', execution_classifier=execution_classifier)
		self.name = 'RFRecursiveFeatureElimination'
		logging.info('RFRFE Instanced')
