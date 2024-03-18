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
import pandas as pd

from rumourexplitett import Model
from rumourObj import Tweet, TweetLite, Intervention, RumourIntervention,RumourInterventionBASE,RumourTokenIntervention,RumourComboIntervention,GraphIntervention
from utils import convert_results_to_pd

parser = argparse.ArgumentParser(description="Run a set of neuron experiments.")

parser.add_argument(
	"-model",
	type=str,
	default="distilgpt2",
	help="""Model type [distilgpt2, gpt-2, etc.].""",
)

parser.add_argument(
	"-out_dir", default=".", type=str, help="""Path of the result folder."""
)

parser.add_argument(
	"--randomize", default=False, action="store_true", help="Randomize model weights."
)

parser.add_argument(
	"-pretrained_model",
	type=str,
	default="res/roberta_causal/",)

parser.add_argument(
	"-pretrained_tok",
	type=str,
	default="res/roberta_tok/")

parser.add_argument(
	"-load_pretrained_model",
	default=True,
	help="whether to load pretrained model")


parser.add_argument(
	"-load_pretrained_tok",
	default=False,
	help="whether to load pretrained tokenizer")

parser.add_argument(
	"-debug_mode",
	default=False)

parser.add_argument(
	"-input_data_dir",
	type=str,
	default="../pheme.pkl",
)

parser.add_argument(
	"-rumour_veracity",
	default=False
)
parser.add_argument(
	'-base_model',
	default=False)

parser.add_argument(
	'-tok_mode',
	default=False)


parser.add_argument(
	'-combo_mode', default=False)

parser.add_argument(
	'-combo_test_mode', default=False)


opt = parser.parse_args()


def get_intervention_types():
	return ["direct","indirect"]




def construct_story_interventions(story_input_lst, tokenizer, DEVICE):
	interventions = {}
	custom_tokens = ['[MASK_TOK]']
	max_ln = 12
	for s in story_input_lst:
		story_id = s['story_id']
		story_content = s['story']
		reactions_lst = s['lst']

		total_reactions = len(reactions_lst)
		for alt_loc in range(0, total_reactions):
			intervention_id = story_id + '_' + str(alt_loc)
			try:
				# tokennizer, source_sentence: str, reaction_lst: list, custom_tokens:list, alt_loc, gold_label,max_len, device
				interventions[internvention_id] = GraphIntervention(tokenizer, story_content, reactions_lst, custom_tokens, alt_loc, s['is_rumour'], max_ln, device=DEVICE)
			except:
				pass
	print('#######FINISH STORY INTERVENTION CONSTRUCTION #########')
	return interventions






def run_all(
	model_type='roberta-base',
	device="cuda",
	out_dir=".",
	random_weights=False,
	load_pretrained_model=True,
	pretrained_model='res/',
	debug_mode=False,
	input_data_dir='',
	rumour_veracity=False,
	base_model=False,
	tok_mode=False,
	combo_test_mode=False,
):
	print("Model:", model_type, flush=True)
	print('Device: ', device)
	print('Random_Weights, ',random_weights)
	print('Load Pretrained Model ', load_pretrained_model)
	print('Pretained Model ', pretrained_model)
	print('Debug Mode ', debug_mode)
	print('Tok Mode ', tok_mode)
	print('Combo Test Mode ',combo_test_mode)
	# Set up all the potential combinations


	intervention_types = get_intervention_types()

	# Initialize Model and Tokenizer.
	# TODO: Need to init the twotier transformer properly here!
	model = Model(device=device, gpt2_version=model_type, random_weights=random_weights,load_pretrained_model=load_pretrained_model,pretrained_model=pretrained_model)

	tokenizer = (GPT2Tokenizer if model.is_gpt2 else
	             TransfoXLTokenizer if model.is_txl else
	             XLNetTokenizer if model.is_xlnet else
	             BertTokenizer if model.is_bert else
	             DistilBertTokenizer if model.is_distilbert else
	             RobertaTokenizer).from_pretrained(model_type)

	# Set up folder if it does not exist.
	dt_string = datetime.now().strftime("%Y%m%d")
	folder_name = dt_string + "_neuron_intervention"
	base_path = os.path.join(out_dir, "results", folder_name)
	if random_weights:
		base_path = os.path.join(base_path, "random")
	if not os.path.exists(base_path):
		os.makedirs(base_path)




	data_df = pd.read_pickle(input_data_dir)
	listofTweets = [(Story(row.source_id,row.source_text, row.reply_text_lst, row.is_rumour)) for index, row in data_df.iterrows() ]  
	
	interventions = construct_story_interventions(listofTweets, tokenizer, device)



	# Consider all the intervention types
	for itype in intervention_types:
		print("\t Running with intervention: {}".format(itype), flush=True)
		# Run actual exp.
		print('current intervention type ', itype)
		if not base_model:
			intervention_results = model.neuron_intervention_experiment(
				interventions, intervention_type=itype, alpha=1.0,rumour_veracity=rumour_veracity
			)
		else:
			intervention_results = model.base_neuron_intervention_experiment(
				interventions
			)


		df = pd.DataFrame.from_dict(intervention_results,orient='index')

		# Generate file name.
		temp_string = "_".join('rumour_test_{}'.replace("{}", "X").split())
		model_type_string = model_type
		fname = "_".join([temp_string, itype, model_type_string])
		# Finally, save each exp separately.
		df.to_csv(os.path.join(base_path, fname + ".csv"))
		df.to_pickle(os.path.join(base_path, fname + ".pkl"))



if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	run_all(
		model_type=opt.model,
		device=device,
		out_dir = opt.out_dir,
		random_weights=opt.randomize,
		load_pretrained_model=opt.load_pretrained_model,
		pretrained_model = opt.pretrained_model,
		debug_mode = opt.debug_mode,
		input_data_dir = opt.input_data_dir,
		rumour_veracity = opt.rumour_veracity,
		base_model=opt.base_model,
		tok_mode=opt.tok_mode,
		combo_test_mode=opt.combo_test_mode,
	)
