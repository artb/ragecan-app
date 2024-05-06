import optuna

#RAGECAN CONF
GROUP_MAPPING = {
	'anova': 'FilterMethods',
	'chi-square': 'FilterMethods',
	'mutual-info': 'FilterMethods',
	'mrmr': 'FilterMethods',
	'pca': 'ExtractionMethods',
	'autoencoder': 'ExtractionMethods',
	'denoising-autoencoder': 'ExtractionMethods',
	'siamese-networks': 'ExtractionMethods',
	'svm-rfe': 'WrapperMethods',
	'rf-rfe': 'WrapperMethods'
}
MIN_VALUE = 50
CV_FOLDS = 4

#AUTOENCODER CONF
AE_LAYER_MIN_DIST = 200
AE_N_LAYERS_FLOOR = 1
AE_N_LAYERS_CELING = 2
AE_LIMIT = 4000
AE_LR_LIST = [1e-4, 1e-3, 1e-2, 1e-1]
#OPTUNA CONF
PRUNER = optuna.pruners.MedianPruner()
OPTUNA_EXECUTIONS = 240
N_JOBS_FS = 14
N_JOBS_FE = 1
N_JOBS_WP = 1

#SVM OPT
SVM_KERNEL_LIST = ["linear", "rbf", "poly"]
SVM_C_LIST = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
#RF OPT
RF_ESTIMATORS_FLOOR = 10
RF_ESTIMATORS_CELING = 1000
RF_DEPTH_FLOOR = 2
RF_DEPTH_CELING = 32


#MISC
ACCEPTED_METRICS = ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'completeness_score', 'explained_variance', 'f1', 'f1_macro',
           'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted',
           'matthews_corrcoef', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance',
           'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score',
           'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
           'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score'
        ]
        
        
        
        
        