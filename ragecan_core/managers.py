import pandas as pd
import mlflow
from . import feature_selection, feature_extraction, wrapper_selection
import logging

from icecream import ic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlflow import start_run
from .automatic_reduction_pipelines import BinarySearchFilterSelection
from .constants import GROUP_MAPPING, MIN_VALUE

ALPHA_DIFF = 0.03
TRAIN_ALPHA_DIFF = 0.02
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class SageCanJudge:
	def __init__(self, metric, experiment_goal):
		logging.info('Judger Instanced')
		self.metric = metric
		self.best_experiment_metric = 0.0
		self._current_experiment = []
		self.experiment_goal = experiment_goal
		self.ragecan_logger = None
  
	def is_criteria_reached(self, current_best_metric):
		if current_best_metric > self.experiment_goal:
			return True
		else:
			return False

	def add_logger(self, logger):
		self.ragecan_logger = logger

	def save_rejected_result(self, result, diff):
		self.ragecan_logger.add_rejected_reduction_result(reductor_method_name=result.method_name ,train_metrics=result.train_metrics, test_metrics=result.test_metrics,optimized_parameters=result.optimized_parameters, difference=diff)
		return 1


	def get_metrics_and_reduced(self, experiment_metrics):
		"""
		Ordena todos os resultados em uma única lista e retorna o melhor 
		FSOptimizationResult com base na métrica especificada.

		:param all_results: Lista de listas contendo objetos FSOptimizationResult.
		:return: O melhor FSOptimizationResult com base na métrica escolhida e se o resultado do experimento anterior foi superado
		"""
		best_result = None
		best_metric_value = 0.0
		underscored = True
  
		for result in experiment_metrics:
			experiment_metric = "mcc" if self.metric == "matthews_corrcoef" else self.metric
			current_metric_value = result.test_metrics[experiment_metric]

			# Verifica se o resultado atual é melhor que o melhor encontrado até agora
			if current_metric_value > (best_metric_value+TRAIN_ALPHA_DIFF):
				best_metric_value = current_metric_value
				best_result = result
	
		if best_result.test_metrics[experiment_metric] > (self.best_experiment_metric+ALPHA_DIFF):
			underscored = False
			diff = (self.best_experiment_metric+ALPHA_DIFF) - best_result.test_metrics[experiment_metric]
			logging.info('Saving Rejected Optimation Group Execution!\n')
			self.save_rejected_result(best_result, diff)
   
   
		return best_result, underscored


class SageCanExecuter:
	def __init__(self, execution_methods, metric, classifier_name, optuna_tries):
		self.metric = metric
		self.classifier_name = classifier_name
		self.optuna_tries = optuna_tries
		# Here its created the execution instanced methods that will be grouped as 'FilterMethods', 'WrapperMethods' or 'ExtractionMethods'
		self.instanced_methods = self.instanciate_methods(execution_methods, metric)
		logging.info('Executer is instanced!\n')
  
	#TODO: This function must return if a determined group are instanciated or not
	def check_instanciated_methods(self, methods_group):
		return True
  
	def instanciate_methods(self, execution_methods, metric):
		method_instances = {
			'FilterMethods': [],
			'WrapperMethods': [],
			'ExtractionMethods': []
		}

		method_mapping = {
			'anova': feature_selection.AnovaFeatureSelection,
			'chi-square': feature_selection.ChiSquareFeatureSelection,
			'mutual-info' : feature_selection.MutualInformationFeatureSelection,
			'mrmr': feature_selection.MRMRFeatureSelection,
			'pca': feature_extraction.PCAFeatureExtraction,
			'autoencoder': feature_extraction.AutoencoderFeatureSelector,
			'denoising-autoencoder': feature_extraction.DenoisingAutoencoderFeatureSelector,
			'siamese-networks': feature_extraction.SiameseNetworksFeatureSelector,
			'svm-rfe': wrapper_selection.SVMRecursiveFeatureElimination,
			'rf-rfe': wrapper_selection.RFRecursiveFeatureElimination
		} 

		group_mapping = GROUP_MAPPING
		
		for method in execution_methods:
			if method in method_mapping:
				instance = method_mapping[method](metric, self.classifier_name)
				method_group = group_mapping[method]
				method_instances[method_group].append(instance)

		logging.warning(f'INSTANCED METHODS: {method_instances}')
		return method_instances

 	#Returns the best_model_config, metrics
	def run_fs_method(self, X_train, y_train, X_test, y_test, method, range):
		ic(f'\n***************\nRunning FS with Method: {method.name} and {range}\n')
		fs_optimization_result = method.execute_selection(X_train, y_train, X_test, y_test, range, self.optuna_tries)
		return fs_optimization_result

	def check_pca_execution(self, instanced_methods, n_samples, n_features_to_filter):
		if 'pca' in instanced_methods['ExtractionMethods'].values():
			if n_samples < n_features_to_filter:
				logging.warning('PCA cannot be executed this trial! N_samples < desired_dimension')
				return [method for method in instanced_methods['ExtractionMethods'] if method != 'pca']
		return instanced_methods
  
	#Returns the best_model_config, metrics
	def run_group_executions(self, X_train, y_train, X_test, y_test, method_group_name, range):
		trial_results = []
		logging.info('Running group execution!\n')
		ic(self.instanced_methods)
		ic(method_group_name)
		trial_instanced_methods = self.instanced_methods[method_group_name]
		# if method_group_name == "ExtractionMethods":
		# 	trial_instanced_methods = self.check_pca_execution(trial_instanced_methods, X_train.shape[0], range[1])
   
		for instanced_method in trial_instanced_methods:
			ic(f"executing the method: {instanced_method} with range: {range}")
			fs_optimization_result = self.run_fs_method(X_train, y_train, X_test, y_test , instanced_method, range)
			trial_results.append(fs_optimization_result)
		return trial_results
  

	# trial_methods = [('FilterMethods', (100, 1000)), ('WrapperMethods', (65))]
	def run_groups_and_get_results(self, X_train, y_train, X_test, y_test, trial_methods_groups):
		logging.info('Running multiple group execution!\n')
		groups_results = []
		for execution_config in trial_methods_groups:
			methods_group = execution_config[0]
			trial_range = execution_config[1]
			#Returns the best strategy for the group of tecniques
			execution_results = self.run_group_executions(X_train, y_train, X_test, y_test, methods_group, trial_range)
			groups_results.append(execution_results)
   
		return groups_results


class ExperientOrchestrator:
	def __init__(self, experiment_name, csv_file_path, label_column_name):
		self.raw_dataset = pd.read_csv(csv_file_path)
		self.raw_dataset.rename(columns=lambda x: x.replace('.', '_'), inplace=True)
		self.labels = self.raw_dataset[label_column_name]
		self.raw_dataset.drop(label_column_name, axis=1, inplace=True)
		self.experiment_name = experiment_name
		self.sagecan_logger = SageCanLogger(experiment_name)
		logging.info('ExperimentOrchestrator Instanced\n')	
		
		
	def config(self, selected_methods,classifier_name,patience,optimization_tentatives,target,metric):
		ic(classifier_name)
		opt_tentatives = int(optimization_tentatives)
		self.n_optimization_tentatives = opt_tentatives
		self.sagecan_executer = SageCanExecuter(selected_methods, metric, classifier_name, opt_tentatives)
		self.judger = SageCanJudge(metric, target)
		self.judger.add_logger(self.sagecan_logger)
		self.patience = int(patience)
		self.target = float(target)
		self.metric = metric
		self.classifier_name = classifier_name
		logging.info('[LOG] >> Orchestrator are configured this way:\n')
		self.print_state()


	def run_experiment(self):
		X_train, X_test, y_train, y_test = train_test_split(self.raw_dataset, self.labels, test_size=0.3, random_state=2022)
		logging.warning(f"TRAIN SHAPE: {X_train.shape} AND TEST SHAPE: {X_test.shape}\n")
		bsfs_ins = BinarySearchFilterSelection(self.patience, self.sagecan_logger, self.sagecan_executer, self.judger, self.classifier_name)
		pipeline_result = bsfs_ins.run_pipeline(X_train, y_train, X_test, y_test)
		pip_res_dict = pipeline_result.to_dict()
		self.sagecan_logger.save_metrics(metrics=pip_res_dict['train_metrics'], run_name='FINAL_RESULTS_TRAIN', params=pip_res_dict['optimized_parameters'])
		self.sagecan_logger.save_metrics(metrics=pip_res_dict['test_metrics'], run_name='FINAL_RESULTS_TEST', params=pip_res_dict['optimized_parameters'])
		logging.info('EXPERIMENT IS CONCLUDED SUCCESSFULLY!\n')
		self.sagecan_logger.generate_final_report()
		self.sagecan_logger.generate_final_report_rejected()
		return pipeline_result

	def print_state(self):
		ic(f"Experiment Name: {self.experiment_name}")
		ic(f"Dataset Shape: {self.raw_dataset.shape}")
		ic(f"Labels Shape: {self.labels.shape}")
		ic(f"Classifier Name: {self.classifier_name}")
		ic(f"Metric: {self.metric}")
		ic(f"Patience: {self.patience}")
		ic(f"Optimization Tentatives: {self.n_optimization_tentatives}")
		ic(f"Target: {self.target}")
		
	
class SageCanLogger:
	def __init__(self, experiment_name):
		logging.info('Logger Instanced!\n')
		# must run before mlflow server --host 127.0.0.1 --port 5600
		mlflow.set_tracking_uri(uri="http://127.0.0.1:5600")
		self.experiment_name = experiment_name
		mlflow.set_experiment(experiment_name)
		self.experiments = []
		self.rejected_experiments = []

	def save_metrics(self, metrics, params, run_name='run'):
		with start_run(run_name=run_name):
			mlflow.log_metric("accuracy", metrics["accuracy"])
			mlflow.log_metric("f1_macro", metrics["f1_macro"])
			mlflow.log_metric("MCC", metrics["mcc"])
			mlflow.log_param("Confusion_Matrix", metrics["confusion_matrix"])
			mlflow.log_param("N of Features", params["n_features_to_select"])
			mlflow.log_dict(metrics["clf_report"], "clf_full_report.json")
			mlflow.log_dict(params, "fs_pipeline_params.json")
   
	def add_reduction_stage_result(self, reductor_method_name, train_metrics, test_metrics, optimized_parameters):
		self.experiments.append({
			"reductor_method_name": reductor_method_name,
			"train_metrics": train_metrics,
			"test_metrics": test_metrics,
			"optimized_parameters": optimized_parameters
		})
  
	def add_rejected_reduction_result(self, reductor_method_name, train_metrics, test_metrics, optimized_parameters, difference):
		self.rejected_experiments.append({
			"reductor_method_name": reductor_method_name,
			"train_metrics": train_metrics,
			"test_metrics": test_metrics,
			"optimized_parameters": optimized_parameters,
			"difference": difference
	})
  
	def print_report(self):
		report_lines = ["Experiment Reduction Results Report:"]
		for experiment in self.experiments:
			report_lines.append(f"Method: {experiment['reductor_method_name']}")
			report_lines.append(f"Train Metrics: {experiment['train_metrics']}")
			report_lines.append(f"Test Metrics: {experiment['test_metrics']}")
			report_lines.append(f"Optimized Parameters: {experiment['optimized_parameters']}")
			report_lines.append("-" * 40) 

		report = "\n".join(report_lines)
		print(report)
  
	def generate_final_report(self):
		file_name = './results/' + self.experiment_name + '_report.txt'
		with open(file_name, 'w') as file:
			file.write("Experiment Reduction Results Report:\n")
			for experiment in self.experiments:
				file.write(f"Method: {experiment['reductor_method_name']}\n")
				file.write("Test Metrics:\n")
				for metric, value in experiment['test_metrics'].items():
					file.write(f"\t{metric}: {value}\n")
				file.write("Optimized Parameters:\n")
				for param, value in experiment['optimized_parameters'].items():
					file.write(f"\t{param}: {value}\n")
				file.write('-' * 80 + '\n')
	
	
	def generate_final_report_rejected(self):
		file_name = './rejected_results/' + self.experiment_name + '_report.txt'
		with open(file_name, 'w') as file:
			file.write("Experiment Reduction Rejected Results Report:\n")
			for experiment in self.rejected_experiments:
				file.write(f"Method: {experiment['reductor_method_name']}\n")
				file.write(f"Rejected for: {experiment['difference']} difference\n")
				file.write("Test Metrics:\n")
				for metric, value in experiment['test_metrics'].items():
					file.write(f"\t{metric}: {value}\n")
				file.write("Optimized Parameters:\n")
				for param, value in experiment['optimized_parameters'].items():
					file.write(f"\t{param}: {value}\n")
				file.write('-' * 80 + '\n')