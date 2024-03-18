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


from rumourexplitee import Model
from rumourObj import Tweet, TweetLite, Intervention, RumourIntervention,RumourInterventionBASE,RumourTokenIntervention,RumourComboIntervention
from utils import convert_results_to_pd

parser = argparse.ArgumentParser(description="Run a set of neuron experiments.")

parser.add_argument(
	"-model",
	type=str,
	default="bert",
	help="""Model type [bert, roberta, etc.].""",
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




def construct_interventions(tweet_obj_lst, tokenizer, DEVICE):
	interventions = {}
	label_lst = ['rumour','nonrumour']
	print('len og lst', len(tweet_obj_lst))
	print('starting to construct interventions')
	for t in tweet_obj_lst:

		total_reply = len(t.reply_text_lst)
		for alt_loc in range(0,total_reply):
			internvention_id = t.source_id + '_' +str(alt_loc)
			try:
				interventions[internvention_id] = RumourIntervention(tokenizer, t.source_text, t.reply_text_lst, alt_loc,t.is_rumour, device=DEVICE)
			except:
				pass

	print('########FINISH INTERVENTION CONSTRUCTION WITH NUMBER OF INTERVENTIONS ',len(interventions))
	return interventions



def construct_debug_interventions(tweet_obj_lst, tokenizer, DEVICE):
	interventions = {}

	label_lst = ['rumour','nonrumour']
	for t in tweet_obj_lst:

		internvention_id = t.source_id + '_' +str(alt_loc)
		try:
			interventions[internvention_id] = RumourIntervention(tokenizer, t.source_text, t.reply_text_lst, alt_loc,t.is_rumour, device=DEVICE)
		except:
			pass

	return interventions


def construct_token_interventions(tweet_obj_lst, tokenizer, DEVICE):
	interventions = {}
	print('starting to construct token level interventions')
	for t in tweet_obj_lst:
		total_reply = len(t.reply_text_lst)
		for alt_loc in range(0, total_reply):
			total_token_len = len(t.reply_text_lst[alt_loc].split())
			print('total_token_len ',total_token_len)
			for token_alt_loc in range(0, total_token_len):
				internvention_id = t.source_id + '_' + str(alt_loc) + '_tok_'+str(token_alt_loc)
				try:
					interventions[internvention_id] = RumourTokenIntervention(tokenizer, t.source_text, t.reply_text_lst, alt_loc, token_alt_loc, device=DEVICE)

				except Exception as e: 
					print(e)
					pass
	print('########FINISH TOKEN LEVEL INTERVENTION CONSTRUCTION WITH NUMBER OF INTERVENTIONS ',len(interventions))
	return interventions





def construct_base_interventions(tweet_obj_lst, tokenizer, DEVICE):
	interventions = {}
	label_lst = ['rumour','nonrumour']
	for t in tweet_obj_lst:
		for alt_loc in range(0,total_reply):
			internvention_id = t.source_id + '_' +str(alt_loc)
			try:
				interventions[internvention_id] = RumourInterventionBASE(tokenizer, t.source_text, t.reply_text_lst, alt_loc,t.is_rumour, device=DEVICE)
			except:
				pass
	return interventions





def construct_combo_interventions(tweet_obj_lst, tokenizer, DEVICE):
	interventions = {}
	print('starting to construct token level interventions')
	for t in tweet_obj_lst:
		total_reply = len(t.reply_text_lst)
		for alt_loc in range(0, total_reply):
			internvention_id = t.source_id + '_' +str(alt_loc)
			try:
				interventions[internvention_id] = RumourComboIntervention(tokenizer, t.source_text, t.reply_text_lst, alt_loc, device=DEVICE)
			except Exception as e: 
				print(e)
	print('########FINISH COMBO LEVEL INTERVENTION CONSTRUCTION WITH NUMBER OF INTERVENTIONS ',len(interventions))
	return interventions



def construct_story_intervention(story_lst, tokenizer, DEVICE):
	interventions = {}
	for s in story_lst:
		total_source = len(s.source_text_lst)
		print('s.story_id,', s.story_id)
		for alt_loc in range(0, total_source):
			intervention_id = str(s.story_id) + '_' + str(alt_loc)
			try:
				interventions[intervention_id] = PairIntervention(tokenizer, source_sentence=s.story_content, reaction_lst=s.source_text_lst, alt_loc=alt_loc, gold_label=s.is_rumour, turnaround_label_lst=s.source_turnaround_lst,max_len=256,device=DEVICE)
			except Exception as e:
				print(e)
	print('########FINISH INTERVENTION CONSTRUCTION WITH NUMBER OF INTERVENTIONS ',len(interventions))
	return interventions

def run_all(
	model_type='roberta-base',
	device="cuda",
	out_dir=".",
	random_weights=False,
	load_pretrained_model=True,
	pretrained_model='res/roberta_causal/',
	debug_mode=False,
	input_data_dir='',
	rumour_veracity=False,
	base_model=False,
	tok_mode=False,
	combo_test_mode=False,
):

	# Set up all the potential combinations

	intervention_types = get_intervention_types()

	# Initialize Model and Tokenizer.
	model = Model(device=device, model_version=model_type, random_weights=random_weights,load_pretrained_model=load_pretrained_model,pretrained_model=pretrained_model)


	tokenizer = model.tokenizer

	# Set up folder if it does not exist.
	dt_string = datetime.now().strftime("%Y%m%d")
	folder_name = dt_string + "_neuron_intervention"
	base_path = os.path.join(out_dir, "results", folder_name)
	if random_weights:
		base_path = os.path.join(base_path, "random")
	if not os.path.exists(base_path):
		os.makedirs(base_path)


	
	data_df = pd.read_pickle(input_data_dir)
	listofTweets = [(TweetLite(row.source_id,row.source_text, row.reply_text_lst, row.is_rumour)) for index, row in data_df.iterrows() ]  

	if not combo_test_mode:
		if not tok_mode:
			if not base_model:
				if not debug_mode:
					interventions = construct_interventions(listofTweets, tokenizer, device)
				else:
					interventions = construct_debug_interventions(listofTweets, tokenizer, device)
			else:
				interventions = construct_base_interventions(listofTweets, tokenizer, device)
		else:
			interventions = construct_token_interventions(listofTweets, tokenizer, device)
	else:
		interventions = construct_combo_interventions(listofTweets, tokenizer, device)
	print('number of interventions constructed ', len(interventions.keys()))


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
			itype = 'base'

		df = pd.DataFrame.from_dict(intervention_results,orient='index')


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
