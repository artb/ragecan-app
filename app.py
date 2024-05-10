import pandas as pd
import os
from icecream import ic
from flask import Flask, render_template, request, redirect, url_for
from ragecan_core.managers import ExperientOrchestrator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	if 'file' not in request.files:
		return "No file selected", 400
	file = request.files['file']
	label_column = request.form['label_column']
	if file.filename == '':
		return "No file found", 400
	if file and file.filename.endswith('.csv'):
		filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(filename)
		df = pd.read_csv(filename)
		return "CSV has been read and saved!", 200
	else:
		return "Wrong format, use only .csv", 400

@app.route('/start-experiment', methods=['POST'])
def start_experiment():
	filter_methods = []
	# Text input fields
	exp_name = request.form.get('name', '')
	label_column = request.form.get('label_column', '')
	target = request.form.get('target', 0)
	patience = request.form.get('numExections', 0)
	opt_tries = request.form.get('optTries', 0)
	metric = request.form.get('metric') 
	classifier = request.form.get('classifier')
 
	# CSV File input
	file = request.files.get('file')

	if file.filename == '':
		print('no csv file submited')
	if file and file.filename.endswith('.csv'):
		filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(filename)
	else:
		print('no csv file submited')

	
	# Checkboxes
	anova = request.form.get('anova')
	chi_square = request.form.get('chiSquare')
	mrmr = request.form.get('mRmr')
	info_gain = request.form.get('infoGain')
	svm_rfe = request.form.get('svmRfe')
	rf_rfe = request.form.get('rfRfe')
	pca = request.form.get('pca')
	autoencoder = request.form.get('autoencoder')
	denoising = request.form.get('denoising')
	siamese_net = request.form.get('siameseNet')
	
	if anova:
		filter_methods.append('anova')
	if chi_square:
		filter_methods.append('chi-square')
	if mrmr:
		filter_methods.append('mrmr')
	if info_gain:
		filter_methods.append('mutual-info')
	if svm_rfe:
		filter_methods.append('svm-rfe')
	if rf_rfe:
		filter_methods.append('rf-rfe')
	if pca:
		filter_methods.append('pca')
	if autoencoder:
		filter_methods.append('autoencoder')
	if denoising:
		filter_methods.append('denoising-autoencoder')
	if siamese_net:
		filter_methods.append('siamese-networks')
	print('\n\n\n>RUNNING THE EXPERIMENT WITH CONFIGURATION:')
	ic(filter_methods)
	ic(patience)
	ic(opt_tries)
	experiment = ExperientOrchestrator(exp_name, filename, label_column)
	experiment.config(filter_methods,classifier,patience,opt_tries,target,metric)
	experiment_result = experiment.run_experiment()

	print(experiment_result.to_dict())
	
	return "Experimento concluido!", 200

@app.route('/experiments')
def experiments():
	return render_template('experiments.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/help')
def help():
	return render_template('help.html')

if __name__ == '__main__':
	app.run(debug=False)\

