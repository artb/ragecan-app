import logging
import numpy as np
import optuna
import tensorflow as tf
import gc 
import time
import tensorflow_addons as tfa
import random
from icecream import ic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from optuna.pruners import MedianPruner
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, accuracy_score, classification_report
from random import randint
from sklearn.decomposition import PCA
from keras.models import load_model, save_model
from .utils import return_metric_dict, FSOptimizationResult
from .constants import AE_LR_LIST, N_JOBS_FE, CV_FOLDS, SVM_KERNEL_LIST, AE_LAYER_MIN_DIST, AE_LIMIT, AE_N_LAYERS_FLOOR, AE_N_LAYERS_CELING, SVM_C_LIST, RF_ESTIMATORS_FLOOR,RF_ESTIMATORS_CELING,RF_DEPTH_FLOOR , RF_DEPTH_CELING


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=13)

AE_COMP_ACT_L = ['relu', 'selu', 'tanh']
AE_FINAL_ACT_L = ['sigmoid', 'softmax']
AE_EPOCHS = 50
AE_BATCH_SIZE = 16

class AutoencoderFeatureSelector:
	def __init__(self, metric, execution_classifier='SVM'):
		self.best_model_file_path =  './executions/best_autoencoder.h5'
		self.metric = metric
		self.input_shape = None
		self.desired_dimension = None
		self.execution_classifier = execution_classifier
		self.best_autoencoder = None
		self.best_classifier = None
		self.best_optimization_score = -1
		self.name = 'AutoencoderFeatureSelector'

	def create_autoencoder(self, trial):
		k = trial.suggest_int('bottleneck_units', self.desired_dimension[0], self.desired_dimension[1])
		n_layers = trial.suggest_int('n_layers', AE_N_LAYERS_FLOOR, AE_N_LAYERS_CELING)
		logging.info(f'Creating autoencoder with bottleneck dimension={k} and {n_layers} layers')
		unit_distribution = []
		ic(f"Selected {n_layers}\n")
		for i in range(n_layers):
			if i == 0:
				upper_bound = min(self.input_shape, AE_LIMIT)
			else:
				upper_bound = unit_distribution[-1] 

			lower_bound = int(k+((upper_bound - k) // 2))
			ic(f"BOUNDS: {lower_bound}, {upper_bound}\n")
			n_units = random.randint(lower_bound, upper_bound)
			unit_distribution.append(n_units)
		ic(f'Autoencoder beeing created with: {n_units}\n')
		activation_function = trial.suggest_categorical('activation_function', AE_COMP_ACT_L )
		final_activation = trial.suggest_categorical('final_activation', AE_FINAL_ACT_L)
		learning_rate = trial.suggest_categorical('learning_rate', AE_LR_LIST)

		#Creating the model
		input_layer = Input(shape=(self.input_shape,))
		encoded = input_layer
		for units in unit_distribution:
			encoded = Dense(units, activation=activation_function)(encoded)
		bottleneck = Dense(k, activation=activation_function, name='bottleneck')(encoded)
		decoded = bottleneck
		for units in reversed(unit_distribution):
			decoded = Dense(units, activation=activation_function)(decoded)
		
		output_layer = Dense(self.input_shape, activation=final_activation, name='output')(decoded)
		autoencoder = Model(input_layer, output_layer)
		autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
		ic(autoencoder.summary())
		return autoencoder

	def _get_classifier(self, trial):
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

	def objective(self, trial, X, y):
		logging.info('Starting Autoencoder trial objective evaluation')
		autoencoder = self.create_autoencoder(trial)
		#model_summary = autoencoder.summary()
		#ic(f'On Objective Autoencoder model summary:\n{model_summary}')
		autoencoder.fit(X, X, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, verbose=2, callbacks=[callback])
		encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('bottleneck').output)
		del autoencoder
		X_compressed = encoder.predict(X)

		gc.collect()
		classifier = self._get_classifier(trial)
		logging.info(f"Starting cross-validation evaluation on the train")
		score = cross_val_score(classifier, X_compressed, y, cv=CV_FOLDS, scoring=self.metric, n_jobs=1).mean()
		ic(f'Objective score: {score}')
		if score > self.best_optimization_score:
			self.best_optimization_score = score
			encoder.save(self.best_model_file_path)
			logging.info('New Model saved')
		del X_compressed
		del encoder
		gc.collect()
		tf.keras.backend.clear_session()
		ic(f'Completed trial with score: {score}')
		return score

	def optimize(self, X, y, n_trials):
		logging.info('Starting Autoencoder optimization process')
		self.input_shape = X.shape[1]
		ic(f"Autoencoder inputed n_features: {self.input_shape}")
		study_name = f"{self.__class__.__name__.upper()}_{self.execution_classifier}"
		study = optuna.create_study(study_name=study_name, direction="maximize", pruner=MedianPruner())
		study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials)
		ic(f"Optimization completed. Best score: {study.best_value}, best parameters: {study.best_params}")
		self.best_classifier = self._get_classifier(study.best_trial)
		ic(f"Best classifier of optimization={self.best_classifier}")
		return 1

	def evaluate_on_test(self, X_train, y_train, X_test, y_test):
		logging.info('Evaluating Best Autoencoder on test dataset')
		selector = load_model(self.best_model_file_path)
		X_train_selected = selector.predict(X_train)
		X_test_selected = selector.predict(X_test)

		self.best_classifier.fit(X_train_selected, y_train)
		y_train_pred = self.best_classifier.predict(X_train_selected)
		# Predict on the selected test features
		y_pred = self.best_classifier.predict(X_test_selected)
		logging.info(f"On evaluation the y has been predicted")
		train_metrics = return_metric_dict(y_train, y_train_pred)
		test_metrics = return_metric_dict(y_test, y_pred)
		ic(f"Test dataset evaluation score: {test_metrics}\n\n")
		return (X_train_selected, X_test_selected), (train_metrics, test_metrics)

	def execute_selection(self, X_train, y_train, X_test, y_test, desired_dimension, n_trials):
		logging.info('Starting Autoencoder feature selection execution')
		self.desired_dimension = desired_dimension
		ic(f"Starting execution of selection with {desired_dimension}")
		start_time = time.time()
		best_params = self.optimize(X_train, y_train, n_trials)
		end_time = time.time()
		ic(f"Optimization time: {(end_time - start_time) / 60:.2f} minutes.")
		compressed_xs, fs_metrics = self.evaluate_on_test(X_train, y_train, X_test, y_test)
		opt_res = FSOptimizationResult(self.name, best_params, fs_metrics[0], fs_metrics[1], compressed_xs[0], compressed_xs[1])
		ic(f"Autoencoder selection ended, returning: {opt_res}")
		return opt_res
  
  
class DenoisingAutoencoderFeatureSelector(AutoencoderFeatureSelector):
	def __init__(self, metric, execution_classifier='SVM'):
		super().__init__(metric, execution_classifier)
		self.name = 'DenoisingAutoencoderFeatureSelector'

	def add_noise(self, X, noise_factor):
		noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
		X_noised = X + noise
		return X_noised

	def objective(self, trial, X, y):
		logging.info('Starting DenoisingAutoencoder trial objective evaluation')
		noise_factor = trial.suggest_float('noise_factor', 0.1, 0.8, log=True)
		logging.info(f'Adding noise factor: {noise_factor}')
		X_noised = self.add_noise(X, noise_factor)
		autoencoder = self.create_autoencoder(trial)
		model_summary = autoencoder.summary()
		logging.info(f'On Objective Autoencoder model summary:\n{model_summary}')
		autoencoder.fit(X_noised, X, epochs=50, batch_size=64, verbose=2, callbacks=[callback])
		encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('bottleneck').output)
		X_compressed = encoder.predict(X_noised)
		del autoencoder, X_noised
		gc.collect()
		classifier = self._get_classifier(trial)
		logging.info(f"Starting cross-validation evaluation on the train")
		score = cross_val_score(classifier, X_compressed, y, cv=CV_FOLDS, scoring=self.metric, n_jobs=1).mean()
		if score > self.best_optimization_score:
			self.best_optimization_score = score
			encoder.save(self.best_model_file_path)
			logging.info('Model saved')
		del X_compressed
		del encoder
		gc.collect()
		tf.keras.backend.clear_session()
		logging.info(f'Completed trial with score: {score}')
		return score

	def execute_selection(self, X_train, y_train, X_test, y_test, desired_dimension, n_trials):
		logging.info('Starting Denoising Autoencoder feature selection execution')
		return super().execute_selection(X_train, y_train, X_test, y_test, desired_dimension, n_trials)


class SiameseNetworksFeatureSelector:
	def __init__(self, metric, execution_classifier='SVM'):
		ic('SIAMESE NETWORK INITIATED!')
		self.best_model_file_path = './executions/best_siamese.h5'
		self.metric = metric
		self.input_shape = None
		self.desired_dimension = None
		self.execution_classifier = execution_classifier
		self.best_siamese_model = None
		self.best_classifier = None
		self.best_optimization_score = -1
		self.label_encoder = LabelEncoder()
		self.name = 'SiameseNetworksFeatureSelector'

	def create_siamese_model(self, trial):
		k = trial.suggest_int('bottleneck_units', self.desired_dimension[0], self.desired_dimension[1])
		n_layers = trial.suggest_int('n_layers', AE_N_LAYERS_FLOOR, AE_N_LAYERS_CELING)
		ic(f'Creating Siamese Networks with bottleneck dimension={k} and {n_layers} layers')
		initial_units =  min((self.input_shape-1500), 8000) + AE_LAYER_MIN_DIST * (n_layers - 1)
		unit_distribution = np.linspace(initial_units, k + AE_LAYER_MIN_DIST, num=n_layers, dtype=int) - AE_LAYER_MIN_DIST
		
		input_layer = Input(shape=(self.input_shape,))
		encoded = input_layer

		for n_units in unit_distribution:
			encoded = Dense(n_units, activation='relu')(encoded)

		bottleneck = Dense(k, activation='relu', name='bottleneck')(encoded)
		siamese_model = Model(input_layer, bottleneck)
		learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
		siamese_model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tfa.losses.TripletSemiHardLoss())
		return siamese_model

	def _get_classifier(self, trial):
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

	def objective(self, trial, X, y):
		logging.info('Starting Siamese Networks trial objective evaluation')
		siamese_model = self.create_siamese_model(trial)
		model_summary = siamese_model.summary()
		logging.info(f'On Objective Autoencoder model summary:\n{model_summary}')
		siamese_model.fit(X, y, epochs=50, batch_size=64, verbose=2, callbacks=[callback])
		X_compressed = siamese_model.predict(X)
		# Train classifier
		classifier = self._get_classifier(trial)
		logging.info(f"Starting cross-validation evaluation on the train")
		score = cross_val_score(classifier, X_compressed, y, cv=CV_FOLDS, scoring=self.metric, n_jobs=1).mean()
		if score > self.best_optimization_score:
			self.best_optimization_score = score
			siamese_model.save(self.best_model_file_path)
			logging.info('Siamese Model saved')
		del X_compressed
		del siamese_model
		gc.collect()
		tf.keras.backend.clear_session()
		logging.info(f'Completed trial with score: {score}')
		return score

	def optimize(self, X, y, n_trials):
		logging.info('Starting Siamese optimization process')
		self.input_shape = X.shape[1]
		study_name = f"{self.__class__.__name__.upper()}_{self.execution_classifier}"
		study = optuna.create_study(study_name=study_name, direction="maximize", pruner=MedianPruner())
		study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials)
		ic(f"Optimization completed. Best score: {study.best_value}")
		# After optimization, instantiate the best classifier and feature selector with the best parameters
		self.best_classifier = self._get_classifier(study.best_trial)
		return 1
	
	def evaluate_on_test(self, X_train, y_train, X_test, y_test):
		logging.info('Evaluating Best Siamese on test dataset')
		# Preprocess the data using the saved best features
		selector = load_model(self.best_model_file_path)
		X_train_selected = selector.predict(X_train)
		X_test_selected = selector.predict(X_test)

		# Fit the best classifier on the selected training features
		self.best_classifier.fit(X_train_selected, y_train)
		y_train_pred = self.best_classifier.predict(X_train_selected)
		# Predict on the selected test features
		y_pred = self.best_classifier.predict(X_test_selected)
		logging.info(f"On evaluation the y has been predicted")
		train_metrics = return_metric_dict(y_train, y_train_pred)
		test_metrics = return_metric_dict(y_test, y_pred)
		ic(f"Test dataset evaluation score: {test_metrics}")
		return (X_train_selected, X_test_selected), (train_metrics, test_metrics)

	def execute_selection(self, X_train, y_train, X_test, y_test, desired_dimension, n_trials):
		logging.info('Starting Siamese Networks feature selection execution')
		y_train_encoded = self.label_encoder.fit_transform(y_train)
		y_test_encoded = self.label_encoder.transform(y_test)
		self.desired_dimension = desired_dimension
		ic(f"Starting execution of selection with {desired_dimension}")
		start_time = time.time()
		best_params = self.optimize(X_train, y_train_encoded, n_trials)
		end_time = time.time()
		ic(f"Optimization time: {(end_time - start_time) / 60:.2f} minutes.")
		compressed_xs, fs_metrics = self.evaluate_on_test(X_train, y_train_encoded, X_test, y_test_encoded)
		opt_res = FSOptimizationResult(self.name, best_params, fs_metrics[0], fs_metrics[1], compressed_xs[0], compressed_xs[1])
		ic(f"Siamese Networks selection ended, returning: {opt_res}")
		return opt_res
	

class PCAFeatureExtraction:
	def __init__(self, metric='accuracy', execution_classifier='SVM'):
		self.metric = metric
		self.input_shape = None
		self.desired_dimension = None
		self.execution_classifier = execution_classifier
		self.scaler = MinMaxScaler()
		self.best_pca = None
		self.best_classifier = None
		self.best_optimization_score = -1
		self.name = 'PCAFeatureExtraction'

	def _get_classifier(self, trial):
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

	def objective(self, trial, X, y):
		logging.info('Starting PCA trial objective evaluation')
		ic(f'Execution desired dimension: {self.desired_dimension}')
		k = trial.suggest_int('n_components', self.desired_dimension[0], self.desired_dimension[1])
		ic(f'Creating PCA space with dimension={k}')
		pca = PCA(n_components=k)
		X_compressed = pca.fit_transform(X)

		# Train classifier on the transformed dataset
		classifier = self._get_classifier(trial)
		score = cross_val_score(classifier, X_compressed, y, cv=CV_FOLDS, scoring=self.metric, n_jobs=1).mean()
		if score > self.best_optimization_score:
			self.best_optimization_score = score
			self.best_pca = pca
		ic(f'Completed trial with score: {score}')
		return score

	def optimize(self, X, y, n_trials):
		logging.info('Starting PCA optimization process')
		self.input_shape = X.shape[0]
		study_name = f"{self.__class__.__name__.upper()}_{self.execution_classifier}"
		study = optuna.create_study(study_name=study_name, direction="maximize", pruner=MedianPruner())
		study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials, n_jobs=N_JOBS_FE)
		ic(f"Optimization completed. Best score: {study.best_value}")
		# After optimization, instantiate the best classifier and feature selector with the best parameters
		self.best_classifier = self._get_classifier(study.best_trial)
		return 1

	def evaluate_on_test(self, X_train, y_train, X_test, y_test):
		logging.info('Evaluating Best PCA on test dataset')
		X_train_selected = self.best_pca.fit_transform(X_train)
		X_test_selected = self.best_pca.transform(X_test)

		# Fit the best classifier on the selected training features
		self.best_classifier.fit(X_train_selected, y_train)
		y_train_pred = self.best_classifier.predict(X_train_selected)
		# Predict on the selected test features
		y_pred = self.best_classifier.predict(X_test_selected)
		logging.info(f"On evaluation the y has been predicted")
		train_metrics = return_metric_dict(y_train, y_train_pred)
		test_metrics = return_metric_dict(y_test, y_pred)
		ic(f"Test dataset evaluation score: {test_metrics}")
		return (X_train_selected, X_test_selected), (train_metrics, test_metrics)

	def execute_selection(self, X_train, y_train, X_test, y_test, desired_dimension, n_trials):
		logging.info('Starting PCA feature selection execution')
		ic(f"\n**************\nStarting execution of selection with {desired_dimension}\n")
		X_train_scaled = self.scaler.fit_transform(X_train)
		X_test_scaled = self.scaler.transform(X_test)
		upper_limiar_pca = min(X_train.shape[0], X_train.shape[1])
		if desired_dimension[1] > upper_limiar_pca:
			if desired_dimension[0] < upper_limiar_pca:
				desired_dimension = (desired_dimension[0], min(X_train.shape[0], X_train.shape[1]))
			else:
				desired_dimension = (upper_limiar_pca-1, upper_limiar_pca)
		self.desired_dimension = desired_dimension
		start_time = time.time()
		best_params = self.optimize(X_train, y_train, n_trials)
		end_time = time.time()
		ic(f"Optimization time: {(end_time - start_time) / 60:.2f} minutes.")
		compressed_xs, fs_metrics = self.evaluate_on_test(X_train_scaled, y_train, X_test_scaled, y_test)
		opt_res = FSOptimizationResult(self.name, best_params, fs_metrics[0], fs_metrics[1], compressed_xs[0], compressed_xs[1])
		ic(f"PCA extraction ended, returning: {opt_res}")
		return opt_res