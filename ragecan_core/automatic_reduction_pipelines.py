import logging
from icecream import ic
from .constants import MIN_VALUE, AE_LIMIT
from .feature_selection import AnovaFeatureSelection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class BinarySearchFilterSelection:
	def __init__(self, patience, sagecan_logger, executer, judger, classifier_name):
		self.sagecan_logger = sagecan_logger
		self.executer = executer
		self.judger = judger
		self.patience = patience
		self.classifier_name = classifier_name

	def get_search_ranges(self, X):
		experiment_methods = []
		n_features = X.shape[1]
		# Selection loop: Inspired by binary search, midpoint 
		midpoint = n_features // 2
		lower_range = (MIN_VALUE, midpoint)
		upper_range = (midpoint + 1, (n_features-MIN_VALUE))
		experiment_methods.append(lower_range)
		experiment_methods.append(upper_range)
		return experiment_methods

	def bsfs(self, X_train, y_train, X_test, y_test):
		executions = 0
		underperformances = 0
		above_minimun_dim = True 
		while underperformances < self.patience and above_minimun_dim:
			logging.info('Executing the bsfs loop')
			executions += 1
			experiments = self.get_search_ranges(X_train)
			result_lower = self.executer.run_group_executions(X_train, y_train, X_test, y_test, 'FilterMethods' , experiments[0])
			result_upper = self.executer.run_group_executions(X_train, y_train, X_test, y_test, 'FilterMethods' , experiments[1])
			best_result_lower, _ = self.judger.get_metrics_and_reduced(result_lower)
			best_result_upper, _ = self.judger.get_metrics_and_reduced(result_upper)
			best_result, underperformed = self.judger.get_metrics_and_reduced([best_result_lower, best_result_upper])
			if underperformed:
				underperformances +=1
				ic(underperformances)
			else:
				ic("BSFS current result overpassed previous!\n")
				logging.info(f'Reducing to dimension: {str(best_result.X_train_compressed.shape)}')
				best_result_dict = best_result_upper.to_dict()
				self.sagecan_logger.save_metrics(metrics=best_result_dict['train_metrics'], run_name='BSFS_'+best_result_dict['method_name']+'_train', params=best_result_dict['optimized_parameters'])
				self.sagecan_logger.save_metrics(metrics=best_result_dict['test_metrics'], run_name='BSFS_'+best_result_dict['method_name']+'_test', params=best_result_dict['optimized_parameters'])
				X_train = best_result.X_train_compressed
				X_test = best_result.X_test_compressed
				if (X_train.shape[1]// 2) <= MIN_VALUE:
					logging.warning('BSFS HAS REACHED MIN VALUE')
					above_minimun_dim = False

		return best_result

	def run_pipeline(self, X_train, y_train, X_test, y_test):
		(X_train.shape)
		faction = X_train.shape[1] // 6
		pipeline_result = None
		first_compresser = AnovaFeatureSelection(self.judger.metric, self.classifier_name)
		first_dim = ((X_train.shape[1]-faction), (X_train.shape[1]-MIN_VALUE))
		first_red = self.executer.run_fs_method(X_train, y_train, X_test, y_test, first_compresser, first_dim)
		pipeline_result = first_red
		first_red_dict = first_red.to_dict()
		ic('First ANOVA filter done!\n')
		ic(first_red)
		X_train = first_red.X_train_compressed
		X_test = first_red.X_test_compressed
		self.sagecan_logger.save_metrics(metrics=first_red_dict['train_metrics'], run_name='first_ANOVA_filter_train', params=first_red_dict['optimized_parameters'])
		self.sagecan_logger.save_metrics(metrics=first_red_dict['test_metrics'], run_name='first_ANOVA_filter_test', params=first_red_dict['optimized_parameters'])
		self.sagecan_logger.add_reduction_stage_result(reductor_method_name=first_red_dict['method_name'], train_metrics=first_red_dict['train_metrics'], test_metrics=first_red_dict['test_metrics'], optimized_parameters=first_red_dict['optimized_parameters'])
  
  
		ic(f"EXTRACTION METHODS? {self.executer.check_instanciated_methods('ExtractionMethods')}")
		if False:
			logging.info('Running Extraction methods')
			desired_dimension_ae = (MIN_VALUE, min((X_train.shape[1]-300), AE_LIMIT))
			ic(f"Going for the extraction methods group execution with desired dimension: {desired_dimension_ae}")
			result_extractions_opt = self.executer.run_group_executions(X_train, y_train, X_test, y_test, 'ExtractionMethods' , desired_dimension_ae)
			result_extraction, underperformed = self.judger.get_metrics_and_reduced(result_extractions_opt)
			if underperformed:
				ic('Best Extraction method failed to surpass previous \n')
			else:
				ic('Best Extraction method is chosen for the pipeline! \n')
			logging.info('Extraction methods run concluded')
			pipeline_result = result_extraction
			X_train = result_extraction.X_train_compressed
			X_test = result_extraction.X_test_compressed
			extraction_dict = result_extraction.to_dict()
			self.sagecan_logger.save_metrics(metrics=extraction_dict['train_metrics'], run_name=extraction_dict['method_name']+'_train', params=extraction_dict['optimized_parameters'])
			self.sagecan_logger.save_metrics(metrics=extraction_dict['test_metrics'], run_name=extraction_dict['method_name']+'_test', params=extraction_dict['optimized_parameters'])
			self.sagecan_logger.add_reduction_stage_result(reductor_method_name=extraction_dict['method_name'], train_metrics=extraction_dict['train_metrics'], test_metrics=extraction_dict['test_metrics'], optimized_parameters=extraction_dict['optimized_parameters'])


		logging.info('Starting bsbf from pipeline')
		result_filter_selection = self.bsfs(X_train, y_train, X_test, y_test)
		pipeline_result = result_filter_selection
		bsfs_dict = result_filter_selection.to_dict()
		self.sagecan_logger.save_metrics(metrics=bsfs_dict['train_metrics'], run_name=bsfs_dict['method_name']+'BSFS_train', params=bsfs_dict['optimized_parameters'])
		self.sagecan_logger.save_metrics(metrics=bsfs_dict['test_metrics'], run_name=bsfs_dict['method_name']+'BSFS_test', params=bsfs_dict['optimized_parameters'])
		self.sagecan_logger.add_reduction_stage_result(reductor_method_name=bsfs_dict['method_name'], train_metrics=bsfs_dict['train_metrics'], test_metrics=bsfs_dict['test_metrics'], optimized_parameters=bsfs_dict['optimized_parameters'])
		logging.info('End of bsbf from pipeline')

		if False:
			logging.info('Starting Wrapper methods')
			#TODO: if dim < 1000, pode fazer senao off
			desired_dimension_w = (MIN_VALUE,X_train.shape[1])
			result_wrapper_opt = self.executer.run_group_executions(X_train, y_train, X_test, y_test, 'WrapperMethods' , desired_dimension_w)
			result_wrapper, _ = self.judger.get_metrics_and_reduced(result_wrapper_opt)
			pipeline_result = result_wrapper
			X_train = result_wrapper.X_train_compressed
			X_test = result_wrapper.X_test_compressed
			wrapper_dict = result_wrapper.to_dict()
			self.sagecan_logger.save_metrics(metrics=wrapper_dict['train_metrics'], run_name=wrapper_dict['method_name']+'_train', params=wrapper_dict['optimized_parameters'])
			self.sagecan_logger.save_metrics(metrics=wrapper_dict['test_metrics'], run_name=wrapper_dict['method_name']+'_test', params=wrapper_dict['optimized_parameters'])
			self.sagecan_logger.add_reduction_stage_result(reductor_method_name=wrapper_dict['method_name'], train_metrics=wrapper_dict['train_metrics'], test_metrics=wrapper_dict['test_metrics'], optimized_parameters=wrapper_dict['optimized_parameters'])
		
		logging.info('SUCESSO')
		return pipeline_result

