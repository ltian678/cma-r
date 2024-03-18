"""Run all the extraction for a model across many templates.
"""
import argparse
import os
from datetime import datetime

import torch
from transformers import (
	GPT2Tokenizer, TransfoXLTokenizer, XLNetTokenizer,
	BertTokenizer, DistilBertTokenizer, RobertaTokenizer
)

from rumourexperiment import Intervention, Model, Tweet
from utils import convert_results_to_pd

parser = argparse.ArgumentParser(description="Run a set of neuron experiments.")

parser.add_argument(
	"-model",
	type=str,
	default="bert-base",
	help="""Model type [bert, roberta, etc.].""",
)

parser.add_argument(
	"-out_dir", default=".", type=str, help="""Path of the result folder."""
)
'''
parser.add_argument(
	"-template_indices",
	nargs="+",
	type=int,
	help="Give the indices of templates if you want to run on only a subset",
)
'''
parser.add_argument(
	"--randomize", default=False, action="store_true", help="Randomize model weights."
)

opt = parser.parse_args()



def get_intervention_types():
	return [
		"rumour_direct",
		"rumour_indirect",
		"nonrumour_direct",
		"nonrumour_indirect",
	]




def construct_interventions(tweet_obj_lst, tokenizer, DEVICE):
	interventions = {}
	#all_word_count = 0
	#used_word_count = 0
	#row.source_id,row.source_text, row.reply_text_lst,
	label_lst = ['rumour','nonrumour']
	print('len og lst', len(tweet_obj_lst))
	print('starting to construct interventions')
	for t in tweet_obj_lst:
		#try:
		#print('t.source_id', t.source_id)
		interventions[t.source_id] = Intervention(tokenizer, t.source_text, t.reply_text_lst, t.candidate_comments, label_lst, device=DEVICE)
		#except:
		#    pass
	return interventions



def run_all(
	model_type="bert",
	device="cuda",
	out_dir=".",
	random_weights=False
):
	print("Model:", model_type, flush=True)
	print('Device: ', device)
	intervention_types = get_intervention_types()

	# Initialize Model and Tokenizer.
	model = Model(device=device, model_version=model_type, random_weights=random_weights)

	tokenizer = model.tokenizer

	# Set up folder if it does not exist.
	dt_string = datetime.now().strftime("%Y%m%d")
	folder_name = dt_string + "_neuron_intervention"
	base_path = os.path.join(out_dir, "results", folder_name)
	if random_weights:
		base_path = os.path.join(base_path, "random")
	if not os.path.exists(base_path):
		os.makedirs(base_path)


	# We dont use the templates here
	# But we need to have
	#WIP 
	import pandas as pd
	data_df = pd.read_pickle('/Causal_Data/pheme_rnr_v5_2.pkl')
	listofTweets = [(Tweet(row.source_id,row.source_text, row.reply_text_lst,  row.candidate_comments_lite,row.is_rumour)) for index, row in data_df.iterrows() ]  
	print('len of listofTweets ',len(listofTweets))
	interventions = construct_interventions(listofTweets, tokenizer, device)
	print('number of interventions ', len(interventions.keys()))

	# Iterate over all possible templates.
	#for temp in templates:
	#    print("Running template '{}' now...".format(temp), flush=True)
		# Fill in all professions into current template
	#interventions = construct_interventions(temp, professions, tokenizer, device)
	# Consider all the intervention types
	for itype in intervention_types:
		print("\t Running with intervention: {}".format(itype), flush=True)
		# Run actual exp.
		intervention_results = model.neuron_intervention_experiment(
			interventions, itype, alpha=1.0
		)

		df = convert_results_to_pd(interventions, intervention_results)
		# Generate file name.
		temp_string = "_".join('rumour_test_{}'.replace("{}", "X").split())
		model_type_string = model_type
		fname = "_".join([temp_string, itype, model_type_string])
		# Finally, save each exp separately.
		df.to_csv(os.path.join(base_path, fname + ".csv"))


if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	run_all(
		opt.model,
		device,
		opt.out_dir,
		random_weights=opt.randomize,
		#template_indices=opt.template_indices,
	)
