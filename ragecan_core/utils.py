from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, accuracy_score, classification_report


def return_metric_dict(y_true, y_pred):
	metrics_dict = {}
	acc = accuracy_score(y_true, y_pred)
	f1_macro = f1_score(y_true, y_pred, average='macro')
	mcc = matthews_corrcoef(y_true, y_pred)
	confusion = str(confusion_matrix(y_true, y_pred))
	clf_report = classification_report(y_true, y_pred, output_dict=True)
	metrics_dict['accuracy'] = acc
	metrics_dict['f1_macro'] = f1_macro
	metrics_dict['mcc'] = mcc
	metrics_dict['confusion_matrix'] = confusion
	metrics_dict['clf_report'] = clf_report
	return metrics_dict
	
 
class FSOptimizationResult:
    def __init__(self, method_name, optimized_parameters, train_metrics, test_metrics, X_train_compressed, X_test_compressed):
        self.method_name = method_name
        self.optimized_parameters = optimized_parameters
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.X_train_compressed = X_train_compressed
        self.X_test_compressed = X_test_compressed

    def update(self, method_name, optimized_parameters, train_metrics, test_metrics, X_train_compressed, X_test_compressed):
        self.method_name = method_name
        self.optimized_parameters = optimized_parameters
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.X_train_compressed = X_train_compressed
        self.X_test_compressed = X_test_compressed

    def to_dict(self):
        return {
			'method_name': self.method_name,
            'optimized_parameters': self.optimized_parameters,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'X_train_compressed': self.X_train_compressed,
            'X_test_compressed': self.X_test_compressed
        }
        
    def __str__(self):
        attributes = self.to_dict()
        return '\n'.join(f"{key}: {value}" for key, value in attributes.items())