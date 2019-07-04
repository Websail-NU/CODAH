
import copy
import itertools
import json
import os
import random
from pprint import pprint

try:
	import drexo
except ImportError:
	drexo_url = 'https://gitlab.com/mdarcy220/drexo/raw/51c2d3db67f0e7b51df2aeb5c4c9e907053c5aae/drexo/drexo.py?inline=false'
	print('Could not import drexo v0.1. Downloading standalone from {}'.format(drexo_url))
	import urllib.request
	urllib.request.urlretrieve(drexo_url, 'drexo.py')
	import drexo

import numpy as np


CV_NUM_FOLDS = 5
NUM_TRIALS = 3

runman = None


def make_cmdargs_from_params(params, output_dir, seed):
	core_args = list(itertools.chain.from_iterable(params['core_args']))

	grid_args = []
	for argname in params['grid_params']:
		if argname == '--train_batch_size':
			batch_size = params['grid_params'][argname]
			grid_args.extend(['--gradient_accumulation_steps', str(batch_size//2)])
		if argname == '--num_train_epochs':
			num_epochs = params['grid_params'][argname]
		grid_args.extend([argname, str(params['grid_params'][argname])])

	final_args = ['python3'] + [params['run_filename']] + core_args + grid_args
	final_args.extend(['--output_dir', output_dir + os.path.sep])
	final_args.extend(['--data_dir', params['data_dir']])
	final_args.extend(['--seed', seed])
	return final_args

def on_job_finished(full_runconfig):
	# Remove saved model after (to save space)
	model_save_filename = full_runconfig['params']['model_save_filename']
	if '{}' in model_save_filename:
		model_save_filename = model_save_filename.format(full_runconfig['params']['grid_params']['--num_train_epochs']-1)
	model_save_filepath = os.path.join(full_runconfig['job_output_dir'], model_save_filename)
	if os.path.exists(model_save_filepath):
		os.remove(model_save_filepath)

def compare_params(cur_params, params_from_cache):
	if 'keys_to_compare' not in cur_params:
		return False

	#if 'keys_to_compare' not in params_from_cache or set(params_from_cache['keys_to_compare']) != set(cur_params['keys_to_compare']):
	if 'keys_to_compare' not in params_from_cache or not set(params_from_cache['keys_to_compare']).issubset(set(cur_params['keys_to_compare'])):
		return False

	cur_params['keys_to_compare']
	for key in cur_params['keys_to_compare']:
		if key not in cur_params or key not in params_from_cache or cur_params[key] != params_from_cache[key]:
			return False
	return True


def get_acc_from_dir(output_dir):
	label_info = None
	with open(os.path.join(output_dir, 'output', 'model_labels.json'), 'r') as f:
		label_info = json.load(f)
	num_total = 0
	num_correct = 0
	for ex in label_info:
		num_total += 1
		if ex['model_label'] == ex['true_label']:
			num_correct += 1
	return num_correct / num_total

def try_until_good(params, data_dir, callback):
	global runman

	failure_cutoff_acc = 0.30
	max_trials = 5

	new_params = copy.deepcopy(params)
	new_params['data_dir'] = data_dir

	results = []
	def end_hook(output_dir, seed):
		nonlocal results, failure_cutoff_acc, max_trials, new_params
		acc = get_acc_from_dir(output_dir)
		results.append(acc)
		if acc < failure_cutoff_acc and len(results) < max_trials:
			runman.add_to_run_queue(new_params, end_hook, front=True)
			return

		if acc < failure_cutoff_acc:
			acc = sum(results)/len(results)

		callback(new_params, output_dir, data_dir, acc)
	runman.add_to_run_queue(new_params, end_hook)

def try_params_on_folds(params, data_dirs, callback):
	fold_results = dict()
	def sub_callback(_1, _2, data_dir, acc):
		nonlocal fold_results
		fold_results[data_dir] = acc
		for d in data_dirs:
			if d not in fold_results:
				return False
		avg_acc = sum([x for x in fold_results.values()]) / len(fold_results)
		callback(params, data_dirs, avg_acc)

	for data_dir in data_dirs:
		try_until_good(params, data_dir, sub_callback)

def get_all_gridsearch_params():
	all_params = []
	for batch_size in [16, 32]:
		for num_epochs in [3, 4, 6]:
			for learning_rate in [1e-5, 2e-5, 3e-5]:
				all_params.append({'--train_batch_size': batch_size, '--num_train_epochs': num_epochs, '--learning_rate': learning_rate})
	return all_params

def combine_params(base_params, grid_params):
	params = copy.copy(base_params)
	params['grid_params'] = copy.copy(grid_params)
	return params

def gridsearch_params_on_subfolds(base_params, main_fold, callback):
	global CV_NUM_FOLDS

	results = []
	num_tested_params = 0
	def sub_callback(params, _, avg_acc):
		nonlocal results, num_tested_params
		results.append((params, avg_acc))
		num_tested_params += 1
		if num_tested_params != len(get_all_gridsearch_params()):
			return False
		best_params = None
		best_params_value = -1
		for p, v in results:
			if best_params_value < v:
				best_params = p
				best_params_value = v
		callback(base_params, main_fold, best_params)

	data_dirs = []
	for fold_num in range(CV_NUM_FOLDS):
		data_dirs.append(os.path.join(main_fold, 'fold{}'.format(fold_num)))

	for grid_params in get_all_gridsearch_params():
		params = combine_params(base_params, grid_params)
		try_params_on_folds(params, data_dirs, sub_callback)

def run_cv(model_info, data_path, modelname, config, do_nested_gridsearch=False):
	global CV_NUM_FOLDS
	arg_pieces = []

	model_common_args = [['--do_eval']]

	for fold_num in range(CV_NUM_FOLDS):
		model_specific_args = []

		if modelname == 'bert':
			model_specific_args.append(['--do_lower_case'])

		fold_path = data_path + str(fold_num)

		config_args = []
		if config == 'ft_swagonly':
			config_args = [copy.copy(model_info[modelname]['swag_load_args'])]
		elif config == 'ft_swag_and_codah':
			config_args = [['--do_train']] + [copy.copy(model_info[modelname]['swag_load_args'])]
		elif config == 'ft_codah_only':
			config_args = [['--do_train']]
		elif config == 'ft_answeronly':
			config_args = [['--do_train'], ['--answer_only']]
		else:
			print('Unknown config {}'.format(config))
			continue

		if modelname == 'bert' and config not in {'ft_swagonly', 'ft_swag_and_codah'}:
			config_args.append(['--bert_model', 'bert-large-uncased'])

		config_args.append(['--model_labels_save_filename', 'model_labels.json'])

		core_args = [model_info[modelname]['base_args']] + model_common_args + config_args
		core_args = sorted(core_args, key=lambda x: str(x))

		base_params = {
			'keys_to_compare': ['run_filename', 'core_args', 'model', 'config', 'fold_num', 'foldtype', 'data_dir', 'grid_params'],
			'run_filename': model_info[modelname]['run_filename'],
			'core_args': core_args,
			'grid_params': model_info[modelname]['default_grid_params'].copy(),
			'model': modelname,
			'config': config,
			'data_dir': fold_path,
			'fold_num': fold_num,
			'foldtype': 'mainfold',
			'model_save_filename': model_info[modelname]['model_save_filename'],
		}

		_run_fold(base_params, fold_path, do_nested_gridsearch)


global_final_results = []
def _run_fold(base_params, fold_path, do_nested_gridsearch):
	global runman, NUM_TRIALS, global_final_results

	base_params = copy.deepcopy(base_params)
	if do_nested_gridsearch:
		def callback(_1, _2, best_params):
			nonlocal fold_path
			def sub_callback(_3, output_dir, _4, acc):
				global_final_results.append([best_params, acc, os.path.abspath(output_dir)])
			best_params = copy.deepcopy(best_params)
			best_params['foldtype'] = 'mainfold'
			for i in range(NUM_TRIALS):
				try_until_good(best_params, fold_path, sub_callback)

			# Only run answer-only with best_params found in codah-only gridsearch
			if best_params['config'] != 'ft_codah_only':
				return

			best_params2 = copy.deepcopy(best_params)
			def sub_callback2(_3, output_dir, _4, acc):
				global_final_results.append([best_params2, acc, os.path.abspath(output_dir)])

			best_params2['core_args'].append(['--answer_only'])
			best_params2['config'] = 'ft_answeronly'
			best_params2['core_args'] = sorted(best_params2['core_args'], key=lambda x: str(x))
			for i in range(NUM_TRIALS):
				try_until_good(best_params2, fold_path, sub_callback2)

		base_params['foldtype'] = 'subfold'
		gridsearch_params_on_subfolds(base_params, fold_path, callback)
	else:

		def sub_callback(output_dir, seed):
			acc = get_acc_from_dir(output_dir)
			global_final_results.append([base_params, acc, output_dir])

		for i in range(NUM_TRIALS):
			runman.add_to_run_queue(base_params, sub_callback)


def calc_result_stats(raw_results):
	results_byfold = dict()
	for tmp in raw_results:
		params = tmp[0]
		dataname = os.path.basename(os.path.dirname(params['data_dir']))
		modelname = os.path.basename(os.path.dirname(params['model']))
		foldname = os.path.basename(params['data_dir'])
		key = (modelname, params['config'], dataname, foldname)
		if key not in results_byfold:
			results_byfold[key] = []
		results_byfold[key].append(tmp[1])

	results = dict()
	for key in results_byfold:
		newkey = (key[0], key[1], key[2])
		if newkey not in results:
			results[newkey] = []
		results[newkey].append(results_byfold[key])

	for key in results:
		# Avg across folds
		tmp = np.mean(results[key], axis=0).tolist()

		# Avg across trials and convert to percent
		results[key] = 'mean={:0.3f}, std={:0.3f}'.format(100*np.mean(tmp), 100*np.std(tmp, ddof=1))

	return results


if __name__ == '__main__':

	runman = drexo.RunManager('./gitignore/outputs/all_experiments/', make_cmdargs_from_params, params_compare_func=compare_params, job_finished_func=on_job_finished)

	model_info = {
		'bert': {'run_filename': 'run_classifier.py',
				 'base_args': ['--save_final_only', '--task_name', 'codah'],
				 'swag_save_file': './gitignore/saved_models/swag_bert_for_cv/',
				 'swag_load_args': ['--bert_model', './gitignore/saved_models/swag_bert_for_cv/'],
				 'model_save_filename': 'pytorch_model_epoch{}.bin',
				 'default_grid_params': {'--train_batch_size': 16, '--learning_rate': 2e-5, '--num_train_epochs': 3},
				},
		'gpt1': {'run_filename': 'run_swag_gpt1.py',
				 'base_args': ['--train_filename', 'train.tsv', '--eval_filename', 'test.tsv', '--data_format', 'codah'],
				 'swag_save_file': './gitignore/saved_models/swag_gpt1_for_cv/',
				 'model_save_filename': 'pytorch_model.bin',
				 'default_grid_params': {'--train_batch_size': 32, '--learning_rate': 6.25e-5, '--num_train_epochs': 3},
				},
	}
	for modelname in model_info:
		if 'swag_load_args' not in model_info[modelname]:
			model_info[modelname]['swag_load_args'] = ['--load_model_from', os.path.join(model_info[modelname]['swag_save_file'], model_info[modelname]['model_save_filename'])]

	configs = ['ft_codah_only', 'ft_swag_and_codah', 'ft_answeronly']
	for data_dirname in ['codah_' + x for x in ['20', '40', '60', '80']]:
		data_path = os.path.join('./gitignore/data/altsizes/', data_dirname, 'fold')
		for config in configs:
			for model in model_info:
				do_gridsearch = (model == 'bert')
				if do_gridsearch and config == 'ft_answeronly':
					# Answeronly gets tested automatically
					# as part of the ft_codah_only gridsearch
					continue
				run_cv(model_info, data_path, model, config, do_nested_gridsearch=do_gridsearch)

	runman.runner.run()

	print()
	print(global_final_results)
	with open('./gitignore/outputs/global_final_results.json', 'w') as f:
		json.dump(global_final_results, f)

	results = calc_result_stats(global_final_results)
	pprint(results)

	if len(runman.get_errors()) > 0:
		for tmp in runman.get_errors():
			print(tmp)
		print('Run had errors!!!')

