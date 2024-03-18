import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


parser = argparse.ArgumentParser(description="Run a set of neuron experiments analysis.")

parser.add_argument(
	"-type",
	type=str,
	default="indirect",
	help="""which type of intervention""",
)


parser.add_argument(
	"-run",type=str,default='main',help="task selection [main, analysis]")

parser.add_argument(
	"-num_classes",type=int, default=2, help='number of final prediction class')


parser.add_argument(
	"-input_data_dir",type=str,default='/finetune_data/story_v4.pkl',help="The input data file path")

opt = parser.parse_args()


def gen_intervention_types():
	return ['direct','indirect']

def gen_data_dicts(input_data_dir):
	input_data = pd.read_pickle(input_data_dir)
	input_rumour_dict = {}
	input_is_turnaround_dict = {}
	for index, row in input_data.iterrows():
		input_rumour_dict[row['source_id']] = row['is_rumour']
		source_turnaround_lst = row['turnaround_lst']
		source_id = row['source_id']
		for i, j in enumerate(source_turnaround_lst):
			k = source_id + '_' +str(i)
			input_is_turnaround_dict[k] = j
	return input_rumour_dict, input_is_turnaround_dict




def compute_total_effect(row):
	"""Compute the total effect based on the bias directionality."""
	if row["base_c1_effect"] >= 1.0:
		return row["alt1_effect"] / row["base_c1_effect"]
	else:
		return row["alt2_effect"] / row["base_c2_effect"]


def filtered_mean(df, column_name, profession_stereotypicality, model_name):
	"""Get the mean effects after excluding strictly definitional professions."""

	def get_profession(s):
		# Discard PADDING TEXT used in XLNet
		if model_name.startswith('xlnet'): s = s.split('<eos>')[-1]
		return s.split()[1]

	def get_stereotypicality(vals):
		return abs(profession_stereotypicality[vals]["definitional"])

	df["profession"] = df["base_string"].apply(get_profession)
	df["definitional"] = df["profession"].apply(get_stereotypicality)
	return df[df["definitional"] < 0.75][column_name].mean()


def extract_source_id_loc(row):
	combo = row['index']
	combo_1 = combo.split('_')[0]
	return combo_1

def extract_loc(row):
	combo = row['index']
	combo_2 = combo.split('_')[1]
	return combo_2

def cal_base_effect(row):
	base_eff = row['candidate1_base_prob']/row['candidate2_base_prob']
	return base_eff

def cal_alt_effect(row):
	alt_eff = row['candidate1_alt1_prob']/row['candidate2_alt1_prob']
	return alt_eff

def cal_total_effect(row):
	total_eff = row['alt_effect'] / row['base_effect']
	return total_eff
from scipy.special import rel_entr, kl_div
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

def cal_total_effect_multi(row):
	#base_prob = [row['candidate1_base_prob'],row['candidate2_base_prob'],row['candidate3_base_prob']]
	#alt_prob = [row['candidate1_alt1_prob'],row['candidate2_alt1_prob'],row['candidate3_alt1_prob']]
	#calculate (P || Q)
	base_prob = row['base_prob']
	alt_prob = row['alt_prob']
	total_effect = sum(rel_entr(alt_prob, base_prob))
	return total_effect

def cal_layer_effect_multi(row):
	raw_intervention_res = row['intervention_res']
	#raw_base_res = row['base_layer_res']
	keys = list(row['intervention_res'].keys())
	#print('keys ',keys)
	final_res = {}
	for k in keys:
		final_res[k]= sum(rel_entr(raw_intervention_res[k][0],row['base_prob']))
	return final_res

def cal_total_effect_multi_w(row):
	base_prob = row['base_prob']
	alt_prob = row['alt_prob']
	total_effect = wasserstein_distance(alt_prob, base_prob)
	return total_effect

def cal_total_effect_multi_j(row):
	base_prob = row['base_prob']
	alt_prob = row['alt_prob']
	total_effect = jensenshannon(alt_prob, base_prob)
	return total_effect

def cal_layer_effect_multi_w(row):
	raw_intervention_res = row['intervention_res']
	#raw_base_res = row['base_prob']
	keys = list(row['intervention_res'].keys())
	#print(keys)
	final_res = {}
	for k in keys:
		#print('raw_intervention_res[k]',raw_intervention_res[k])
		#print("row['base_prob'] ", row['base_prob'])
		final_res[k]= wasserstein_distance(raw_intervention_res[k][0],row['base_prob'])
	return final_res

def cal_layer_effect_multi_j(row):
	raw_intervention_res = row['intervention_res']
	raw_base_res = row['base_prob']
	keys = list(row['intervention_res'].keys())
	final_res = {}
	for k in keys:
		final_res[k]= jensenshannon(raw_intervention_res[k][0],row['base_prob'])
	return final_res

def cal_total_effect_tv_norm(row):
	base_prob = row['base_prob']
	alt_prob = row['alt_prob']
	total_tv = cal_tv_norm(alt_prob, base_prob)
	return total_tv

def cal_layer_tv_norm(row):
	intervention_res = row['intervention_res']
	keys = list(row['intervention_res'].keys())
	layer_tv_norm = {}
	for k in keys:
		layer_tv_norm[k] = cal_tv_norm(intervention_res[k][0],row['base_prob'])
	return layer_tv_norm

def cal_total_effect_linf(row):
	base_prob = row['base_prob']
	alt_prob = row['alt_prob']
	total_linf = cal_rel_linf_metric(alt_prob, base_prob)
	return total_linf

def cal_layer_linf(row):
	keys = list(row['intervention_res'].keys())
	layer_linf = {}
	for k in keys:
		layer_linf[k] = cal_rel_linf_metric(row['intervention_res'][k][0],row['base_prob'])
	return layer_linf



#cal for statistical distance or total variation norm
def cal_tv_norm(p,q):
	P = np.array(p)
	Q = np.array(q)
	tv_norm = 0.5 * np.linalg.norm(P - Q, ord=1)
	return tv_norm

def cal_rel_linf_metric(p,q):
	P = np.array(p)
	Q = np.array(q)
	log_max_ratio = np.log(np.maximum(P/Q,Q/P))
	rel_linf_exp = np.max(log_max_ratio)
	rel_linf = np.exp(rel_linf_exp)
	return rel_linf



def extract_layer_effect(row):
	raw_layer_res = row['intervention_res']
	keys = list(row['intervention_res'].keys())
	final_res = {}
	for k in keys:
		final_res[k]= raw_layer_res[k][0][0] / raw_layer_res[k][0][1]
	return final_res

'''
def extract_layer_effect_multi(row):
	raw_layer_res = row['intervention_res']
	keys = list(row['intervention_res'].keys())
	final_res = {}
	for k in keys:
		final_res[k] = 
	return final_res
'''
def analysis():
	direct_file = 'rumour_test_X_direct_roberta-base_res_analysis.pkl'
	indirect_file = 'rumour_test_X_indirect_roberta-base_res_analysis.pkl'

	direct_df = pd.read_pickle(direct_file)
	indirect_df = pd.read_pickle(indirect_file)

	direct_df_lite = direct_df[['source_id','intervention_loc','base_effect','alt_effect','layer_effect','total_effect']]
	direct_df_lite.rename(columns={'base_effect': 'direct_base_effect', 'alt_effect': 'direct_alt_effect','layer_effect':'direct_layer_effect','total_effect':'direct_total_effect'}, inplace=True)

	indirect_df_lite = indirect_df[['source_id','intervention_loc','base_effect','alt_effect','layer_effect','total_effect']]
	indirect_df_lite.rename(columns={'base_effect': 'indirect_base_effect', 'alt_effect': 'indirect_alt_effect','layer_effect':'indirect_layer_effect','total_effect':'indirect_total_effect'}, inplace=True)

	overall_effect_df_lite = pd.merge(direct_df_lite, indirect_df_lite, on=['source_id','intervention_loc'])

	overall_effect_df_lite.to_pickle('rumour_test_overall_roberta-base.pkl')



def analysis_multi(direct_file, indirect_file):
	direct_df = pd.read_pickle(direct_file)
	indirect_df = pd.read_pickle(indirect_file)


def match_rumour_label(row, input_rumour_dict):
	return input_rumour_dict[row['source_id']]

def match_turnaround_label(row, input_is_turnaround_dict):
	return input_is_turnaround_dict[row['index']]

def decision_chaging(row):
	base_decision = np.argmax(row['base_prob'])
	alt_decision = np.argmax(row['alt_prob'])
	res = 2
	if base_decision == alt_decision:
		res = 0
	else:
		res = 1
	return res


def main_multi(input_data_dir):
	intervention_types = gen_intervention_types()
	input_rumour_dict, input_is_turnaround_dict = gen_data_dicts(input_data_dir)
	#out_lst = []
	for intervention_type in intervention_types:
		data_file_name = 'rumour_test_X_'+intervention_type+'_roberta-base'
		data_file = data_file_name + '.pkl'
		input_data = pd.read_pickle(data_file)
		input_data.reset_index(inplace=True)
		input_data['source_id'] = input_data.apply(lambda row: extract_source_id_loc(row), axis=1)
		input_data['intervention_loc'] = input_data.apply(lambda row: extract_loc(row), axis=1)
		#input_data['base_effect'] = input_data.apply(lambda row: cal_base_effect_multi(row),axis=1)
		#input_data['alt_effect'] = input_data.apply(lambda row: cal_alt_effect_multi(row),axis=1)
		input_data['layer_effect'] = input_data.apply(lambda row: cal_layer_effect_multi(row),axis=1)
		input_data['layer_effect_w'] = input_data.apply(lambda row: cal_layer_effect_multi_w(row),axis=1)
		input_data['layer_effect_j'] = input_data.apply(lambda row: cal_layer_effect_multi_j(row),axis=1)
		input_data['total_effect'] = input_data.apply(lambda row: cal_total_effect_multi(row),axis=1)
		input_data['total_effect_w'] = input_data.apply(lambda row: cal_total_effect_multi_w(row),axis=1)
		input_data['total_effect_j'] = input_data.apply(lambda row: cal_total_effect_multi_j(row),axis=1)
		input_data['total_effect_tv'] = input_data.apply(lambda row: cal_total_effect_tv_norm(row),axis=1)
		input_data['total_effect_linf'] = input_data.apply(lambda row: cal_total_effect_linf(row),axis=1)
		input_data['layer_effect_tv'] = input_data.apply(lambda row: cal_layer_tv_norm(row),axis=1)
		input_data['layer_effect_linf'] = input_data.apply(lambda row: cal_layer_linf(row),axis=1)
		input_data['is_rumour'] = input_data.apply(lambda row: match_rumour_label(row, input_rumour_dict),axis=1)
		input_data['is_turnaround'] = input_data.apply(lambda row: match_turnaround_label(row, input_is_turnaround_dict),axis=1)
		input_data['decision_change'] = input_data.apply(lambda row: decision_chaging(row),axis=1)
		#new_data['base_effect'] = new_data.apply(lambda row: cal_base_effect(row),axis=1)
		#new_data['alt_effect'] = new_data.apply(lambda row: cal_alt_effect(row),axis=1)
		#new_data['layer_effect'] = new_data.apply(lambda row: extract_layer_effect(row),axis=1)
		input_data.to_pickle(data_file_name+'_multi_res_analysis.pkl')
		#out_lst.append(data_file_name+'_multi_res_analysis.pkl')

def main(type='indirect'):
	data_file_name = 'rumour_test_X_'+type+'_roberta-base'
	data_file = data_file_name + '.pkl'
	new_data = pd.read_pickle(data_file)
	new_data.reset_index(inplace=True)
	new_data['source_id'] = new_data.apply(lambda row: extract_source_id_loc(row), axis=1)
	new_data['intervention_loc'] = new_data.apply(lambda row: extract_loc(row), axis=1)
	new_data['base_effect'] = new_data.apply(lambda row: cal_base_effect(row),axis=1)
	new_data['alt_effect'] = new_data.apply(lambda row: cal_alt_effect(row),axis=1)
	new_data['layer_effect'] = new_data.apply(lambda row: extract_layer_effect(row),axis=1)


	new_data['total_effect'] = new_data.apply(lambda row: cal_total_effect(row),axis=1)
	new_data.to_pickle(data_file_name+'_res_analysis.pkl')
	'''
	paths = [os.path.join(folder_name, f) for f in fnames]
	# fnames[:5], paths[:5]
	woman_files = [
		f
		for f in paths
		if "woman_indirect" in f
		if os.path.exists(f.replace("indirect", "direct"))
	]

	means = []
	he_means = []
	she_means = []
	# For correlations.
	all_female_effects = []
	for path in woman_files:
		temp_df = pd.read_csv(path).groupby("base_string").agg("mean").reset_index()
		temp_df["alt1_effect"] = (
			temp_df["candidate1_alt1_prob"] / temp_df["candidate2_alt1_prob"]
		)
		temp_df["alt2_effect"] = (
			temp_df["candidate2_alt2_prob"] / temp_df["candidate1_alt2_prob"]
		)
		temp_df["base_c1_effect"] = (
			temp_df["candidate1_base_prob"] / temp_df["candidate2_base_prob"]
		)
		temp_df["base_c2_effect"] = (
			temp_df["candidate2_base_prob"] / temp_df["candidate1_base_prob"]
		)
		temp_df["he_total_effect"] = temp_df["alt1_effect"] / temp_df["base_c1_effect"]
		temp_df["she_total_effect"] = temp_df["alt2_effect"] / temp_df["base_c2_effect"]
		temp_df["total_effect"] = temp_df.apply(compute_total_effect, axis=1)

		mean_he_total = filtered_mean(
			temp_df, "he_total_effect", profession_stereotypicality, model_name
		)
		mean_she_total = filtered_mean(
			temp_df, "she_total_effect", profession_stereotypicality, model_name
		)
		mean_total = filtered_mean(
			temp_df, "total_effect", profession_stereotypicality, model_name
		)
		he_means.append(mean_he_total)
		she_means.append(mean_she_total)
		means.append(mean_total)
		all_female_effects.append(temp_df[["base_string", "she_total_effect"]])

	print("The total effect of this model is {:.3f}".format(np.mean(means) - 1))
	print(
		"The total (male) effect of this model is {:.3f}".format(np.mean(he_means) - 1)
	)
	print(
		"The total (female) effect of this model is {:.3f}".format(
			np.mean(she_means) - 1
		)
	)

	# Part 2: Get correlations.

	all_female_total_effects = pd.concat(all_female_effects)
	all_female_total_effects = all_female_total_effects.rename(
		columns={"she_total_effect": "total_effect"}
	)
	x_vals = []
	y_vals = []
	labels = []
	for index, row in all_female_total_effects.iterrows():
		labels.append(row["base_string"])
		y_vals.append(row["total_effect"])
		x_vals.append(
			profession_stereotypicality[
				row["base_string"].split()[1] if not model_name.startswith('xlnet')
				else row["base_string"].split('<eos>')[-1].split()[1]
			]["total"]
		)
	profession_df = pd.DataFrame(
		{"example": labels, "Bias": x_vals, "Total Effect": np.log(y_vals)}
	)
	plt.figure(figsize=(10, 3))
	ax = sns.lineplot(
		"Bias", "Total Effect", data=profession_df, markers=True, dashes=True
	)
	ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
	ax.set_yticklabels(["$e^0$", "$e^1$", "$e^2$", "$e^3$", "$e^4$", "$e^5$"])
	sns.despine()
	plt.savefig(os.path.join(folder_name, "neuron_profession_correlation.pdf"))

	effect_corr = pearsonr(profession_df["Bias"], profession_df["Total Effect"])
	print("================")
	print(
		"The correlation between bias value and (log) effect is {:.2f} (p={:.3f})".format(
			effect_corr[0], effect_corr[1]
		)
	)
	'''


if __name__ == "__main__":
	if opt.run == 'main':
		if opt.num_classes == 2:
			main(opt.type)
		elif opt.num_classes > 2:
			#gen_data_dicts()
			main_multi(opt.input_data_dir)
	elif opt.run == 'analysis':
		analysis()
