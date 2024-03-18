
import math
from operator import length_hint
import statistics
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
	GPT2LMHeadModel, GPT2Tokenizer,
	TransfoXLTokenizer,
	XLNetTokenizer,
	BertForMaskedLM, BertTokenizer,
	DistilBertTokenizer,
	RobertaForMaskedLM, RobertaTokenizer
)

#from transformers_modified.modeling_transfo_xl import TransfoXLLMHeadModel
#from transformers_modified.modeling_xlnet import XLNetLMHeadModel
#from transformers_modified.modeling_distilbert import DistilBertForMaskedLM

from attention_intervention_model import (
	AttentionOverride, TXLAttentionOverride, XLNetAttentionOverride,
	BertAttentionOverride, DistilBertAttentionOverride
)
from utils import batch, convert_results_to_pd

from transformers import RobertaForSequenceClassification, RobertaModel

from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F

from rumourObj import Tweet, Intervention,TweetLite
from TTmodel import TwoTierTransformer

np.random.seed(5)
torch.manual_seed(5)



class Model():
	'''
	Wrapper for all model logic
	'''
	def __init__(self,
				 device='cuda:0',
				 load_pretrained_model=True,
				 pretrained_model='res/'):
		super()
		# Load pre-trained BERT tokenizer
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

		# Configuration for the first tier (full 12-layer BERT)
		config_first_tier = BertConfig.from_pretrained('bert-base-uncased')

		# Configuration for the second tier (6-layer BERT)
		config_second_tier = BertConfig.from_pretrained('bert-base-uncased', num_hidden_layers=6)

		# Initialize the custom TwoTierTransformer model with the two configurations
		self.model = TwoTierTransformer(config_first_tier, config_second_tier)

		self.device = device
		self.load_pretrained_model = load_pretrained_model
		self.pretrained_model = pretrained_model


		self.model.eval()
		self.model.to(device)


		if random_weights:
			print('Randomizing weights')
			self.model.init_weights()

		# Options
		self.top_k = 5
		self.num_layers = self.model.config.num_hidden_layers
		self.num_neurons = self.model.config.hidden_size
		self.num_heads = self.model.config.num_attention_heads
		self.masking_approach = masking_approach # Used only for masked LMs
		assert masking_approach in [1, 2, 3, 4, 5, 6]


		# Special token id's: (mask, cls, sep)
		self.st_ids = (tokenizer.mask_token_id,
					   tokenizer.cls_token_id,
					   tokenizer.sep_token_id)

		# To account for switched dimensions in model internals:
		# Default: [batch_size, seq_len, hidden_dim],
		# txl and xlnet: [seq_len, batch_size, hidden_dim]
		self.order_dims = lambda a: a

	def mlm_inputs(self, context, candidate):
		""" Return input_tokens for the masked LM sampling scheme """
		input_tokens = []
		for i in range(len(candidate)):
			combined = context + candidate[:i] + [self.st_ids[0]]
			if self.masking_approach in [2, 5]:
				combined = combined + candidate[i+1:]
			elif self.masking_approach in [3, 6]:
				combined = combined + [self.st_ids[0]] * len(candidate[i+1:])
			if self.masking_approach > 3:
				combined = [self.st_ids[1]] + combined + [self.st_ids[2]]
			pred_idx = combined.index(self.st_ids[0])
			input_tokens.append((combined, pred_idx))
		return input_tokens



	def get_representations_lite(self, input_ids, attention_masks):
		representations = {}

		with torch.no_grad():
			outputs = self.roberta(input_ids,attention_masks)
			print('type of outputs ', type(outputs))
			results = outputs['hidden_states']
			for i, v in enumerate(results):
				representations[i] = v[(0,0)] #[CLS token in each layer] #0 is the last layer
		return representations


	def get_representations_full(self, input_ids, attention_masks):
		representations = {}
		with torch.no_grad():
		  outputs = self.roberta(input_ids)
		  results = outputs['hidden_states']
		  for i, v in enumerate(results):
			representations[i] = v
		return representations



	def get_representations_og(self, context, position):
		# Hook for saving the representation
		def extract_representation_hook(module,
										input,
										output,
										position,
										representations,
										layer):
			# XLNet: ignore the query stream
			if self.is_xlnet and output.shape[0] == 1: return output
			representations[layer] = output[self.order_dims((0, position))]
		handles = []
		representation = {}
		with torch.no_grad():
			# construct all the hooks
			# word embeddings will be layer -1
			handles.append(self.word_emb_layer.register_forward_hook(
				partial(extract_representation_hook,
						position=position,
						representations=representation,
						layer=-1)))
			# hidden layers
			for layer in range(self.num_layers):
				handles.append(self.neuron_layer(layer).register_forward_hook(
					partial(extract_representation_hook,
							position=position,
							representations=representation,
							layer=layer)))
			if self.is_xlnet:
				self.xlnet_forward(context.unsqueeze(0), clen=1)
			else:
				self.model(context.unsqueeze(0))
			for h in handles:
				h.remove()
		# print(representation[0][:5])
		return representation



	def get_representations(self, input_ids, attention_masks, position):
		# Hook for saving the representation

		def extract_representation_hook(module,
										input,
										output,
										position,
										representations,
										layer):
			representations[layer] = output[(0,0)]
		handles = []
		representation = {}
		with torch.no_grad():
			# construct all the hooks
			# word embeddings will be layer -1
			handles.append(self.word_emb_layer.register_forward_hook(
				partial(extract_representation_hook,
						position=position,
						representations=representation,
						layer=-1)))
			# hidden layers
			for layer in range(self.num_layers):
				handles.append(self.neuron_layer(layer).register_forward_hook(
					partial(extract_representation_hook,
							position=position,
							representations=representation,
							layer=layer)))
			rr(input_ids)
			for h in handles:
				h.remove()
		# print(representation[0][:5])
		return representation

	
	def get_probabilities_for_rumours(self, input_ids, attention_masks):
	  outputs = self.model(input_ids, token_type_ids=None, attention_mask=attention_masks)
	  logits = outputs[0]
	  probs = F.softmax(logits, dim=-1)
	  return probs.tolist()

	def get_probabilities_for_rumours_veracity(self, input_ids, attention_masks):
		outputs = self.model(input_ids, token_type_ids=None, attention_mask=attention_masks)
		logits = outputs[0]
		probs = F.softmax(logits, dim=-1)
		return probs.tolist()

	def neuron_intervention_experiment(self, comments2intervention, intervention_type, layers_to_adj=[], neurons_to_adj=[], alpha=1, intervention_loc='layer',rumour_veracity=False):
		'''
		Run multiple intervention experiments
		'''
		comments2intervention_results = {}
		for comm in tqdm(comments2intervention, desc='comments'):
			comments2intervention_results[comm] = {}
			(base_prob, alt_prob, intervention_res) = self.neuron_intervention_single_comment_experiment_veracity(
				comments2intervention[comm], intervention_type, layers_to_adj, neurons_to_adj,
				alpha, intervention_loc=intervention_loc)
			comments2intervention_results[comm]['base_prob'] = base_prob
			comments2intervention_results[comm]['alt_prob'] = alt_prob
			comments2intervention_results[comm]['intervention_res'] = intervention_res
		return comments2intervention_results



	def neuron_intervention_single_comment_experiment(self, 
												intervention,
												intervention_type, layers_to_adj=[],
												neurons_to_adj=[],
												alpha=100,bsize=800, intervention_loc='layer'):
		if self.is_txl or self.is_xlnet: 32 # to avoid GPU memory error
		with torch.no_grad():
		  if self.is_bert or self.is_distilbert or self.is_roberta or self.is_xlnet:
			num_alts = 1
		  base_representations = self.get_representations_full(
				intervention.og_input_ids.unsqueeze(0),
				intervention.og_attention_mask.unsqueeze(0))
		  modified_representations = self.get_representations_full(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))
		  
		  context = intervention.og_input_ids
		  base_rep = base_representations
		  altered_rep = modified_representations
		  replace_or_diff = 'replace'


		  
		  # Probabilities without intervention (Base case)

		  candidate1_base_prob, candidate2_base_prob = self.get_probabilities_for_rumours(
			intervention.og_input_ids.unsqueeze(0),
			intervention.og_attention_mask.unsqueeze(0))[0]

		  candidate1_alt1_prob, candidate2_alt1_prob = self.get_probabilities_for_rumours(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))[0]

		  if intervention_loc == 'layer':
			intervention_res = {}
			for layer in range(-1, self.num_layers):
				for neurons in batch(range(self.num_neurons), bsize):
					neurons_to_search = [[i] + neurons_to_adj for i in neurons]
				probs = self.rumour_direct_intervention(
				  intervention = intervention,
				  layer = layer,
				  base_rep=base_rep,
				  altered_rep=altered_rep,
				  intervention_type=intervention_type,
				  alpha=alpha)

				intervention_res[layer] = probs
		return (candidate1_base_prob, candidate2_base_prob, candidate1_alt1_prob, candidate2_alt1_prob,intervention_res)

	def neuron_intervention_single_comment_experiment_veracity(self, 
												intervention,
												intervention_type, layers_to_adj=[],
												neurons_to_adj=[],
												alpha=100,bsize=800, intervention_loc='layer'):
		if self.is_txl or self.is_xlnet: 32 # to avoid GPU memory error
		with torch.no_grad():
		  if self.is_bert or self.is_distilbert or self.is_roberta or self.is_xlnet:
			num_alts = 1
		  base_representations = self.get_representations_full(
				intervention.og_input_ids.unsqueeze(0),
				intervention.og_attention_mask.unsqueeze(0))
		  modified_representations = self.get_representations_full(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))
		  
		  context = intervention.og_input_ids
		  base_rep = base_representations
		  altered_rep = modified_representations
		  replace_or_diff = 'replace'


		  
		  # Probabilities without intervention (Base case)

		  base_prob = self.get_probabilities_for_rumours(
			intervention.og_input_ids.unsqueeze(0),
			intervention.og_attention_mask.unsqueeze(0))[0]

		  alt_prob = self.get_probabilities_for_rumours(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))[0]

		  if intervention_loc == 'layer':
			intervention_res = {}
			for layer in range(-1, self.num_layers):
				for neurons in batch(range(self.num_neurons), bsize):
					neurons_to_search = [[i] + neurons_to_adj for i in neurons]
				probs = self.rumour_direct_intervention(
				  intervention = intervention,
				  layer = layer,
				  base_rep=base_rep,
				  altered_rep=altered_rep,
				  intervention_type=intervention_type,
				  alpha=alpha)

				intervention_res[layer] = probs
		return (base_prob,alt_prob,intervention_res)
	
	def rumour_direct_intervention(self, intervention, layer, base_rep, altered_rep, intervention_type,alpha=1.):
		def intervention_hook(module,
							  input,
							  output,
							  intervention,
							  layer_base_rep,
							  layer_altered_rep,
							  intervention_type):

			target_pos = intervention.target_locations

			# Overwrite values in the output
			# First define mask where to overwrite
			scatter_mask = torch.zeros_like(output, dtype=torch.bool)
			#print('before scatter_mask shape ', scatter_mask.shape) 
			if intervention_type == 'direct':
				base = layer_altered_rep
				for target_l in target_pos:
					base[0,target_l,:] = layer_base_rep[0,target_l,:]
					scatter_mask[0,target_l,:] = 1
			elif intervention_type == 'indirect':
				base = layer_base_rep
				for target_l in target_pos:
					base[0,target_l,:] = layer_altered_rep[0,target_l,:]
					scatter_mask[0,target_l,:] = 1
			else:
				raise ValueError(f"Invalid intervention_type: {intervention_type}")

			output.masked_scatter_(scatter_mask, base.flatten())

		# Set up the context as batch
		batch_size = 1
		handle_list = []

		if layer == -1:
			layer_base_rep = base_rep[0]
			layer_altered_rep = altered_rep[0]
			handle_list.append(self.word_emb_layer.register_forward_hook(
				partial(intervention_hook,
						intervention=intervention,
						layer_base_rep=layer_base_rep,
						layer_altered_rep=layer_altered_rep,
						intervention_type=intervention_type)))
		else:
			layer_base_rep = base_rep[layer]
			layer_altered_rep = altered_rep[layer]
			handle_list.append(self.neuron_layer(layer).register_forward_hook(
				partial(intervention_hook,
						intervention=intervention,
						layer_base_rep=layer_base_rep,
						layer_altered_rep=layer_altered_rep,
						intervention_type=intervention_type)))
		new_probabilities = []
		if intervention_type == 'direct':
			new_probabilities = self.get_probabilities_for_rumours(intervention.altered_input_ids.unsqueeze(0),attention_masks=intervention.altered_attention_mask.unsqueeze(0))
		if intervention_type == 'indirect':
			new_probabilities = self.get_probabilities_for_rumours(intervention.og_input_ids.unsqueeze(0),attention_masks=intervention.og_attention_mask.unsqueeze(0))
		for hndle in handle_list:
			hndle.remove()
		return new_probabilities


	def attention_intervention_experiment(self, intervention, effect):
		"""
		Run one full attention intervention experiment
		measuring indirect or direct effect.
		"""
		# E.g. The doctor asked the nurse a question. He
		x = intervention.base_strings_tok[0]
		# E.g. The doctor asked the nurse a question. She
		x_alt = intervention.base_strings_tok[1]

		if effect == 'indirect':
			input = x_alt  # Get attention for x_alt
		elif effect == 'direct':
			input = x  # Get attention for x
		else:
			raise ValueError(f"Invalid effect: {effect}")
		if self.is_bert or self.is_distilbert or self.is_roberta:
			attention_override = []
			input = input.tolist()
			for candidate in intervention.candidates_tok:
				mlm_inputs = self.mlm_inputs(input, candidate)
				for i, c in enumerate(candidate):
					combined, _ = mlm_inputs[i]
					batch = torch.tensor(combined).unsqueeze(0).to(self.device)
					attention_override.append(self.model(batch)[-1])
		elif self.is_xlnet:
			batch = input.clone().detach().unsqueeze(0).to(self.device)
			target_mapping = torch.zeros(
				(1, 1, len(input)), dtype=torch.float, device=self.device)
			attention_override = self.model(
				batch, target_mapping=target_mapping)[-1]
		else:
			batch = input.clone().detach().unsqueeze(0).to(self.device)
			attention_override = self.model(batch)[-1]

		batch_size = 1
		seq_len = len(x)
		seq_len_alt = len(x_alt)
		assert seq_len == seq_len_alt

		with torch.no_grad():

			candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
			candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))
			candidate1_probs_layer = torch.zeros(self.num_layers)
			candidate2_probs_layer = torch.zeros(self.num_layers)

			if effect == 'indirect':
				context = x
			else:
				context = x_alt

			# Intervene at every layer and head by overlaying attention induced by x_alt
			model_attn_override_data = [] # Save layer interventions for model-level intervention later
			for layer in range(self.num_layers):
				if self.is_bert or self.is_distilbert or self.is_roberta:
					layer_attention_override = [a[layer] for a in attention_override]
					attention_override_mask = [torch.ones_like(l, dtype=torch.uint8) for l in layer_attention_override]
				elif self.is_xlnet:
					layer_attention_override = attention_override[layer]
					attention_override_mask = torch.ones_like(layer_attention_override[0], dtype=torch.uint8)
				else:
					layer_attention_override = attention_override[layer]
					attention_override_mask = torch.ones_like(layer_attention_override, dtype=torch.uint8)
				layer_attn_override_data = [{
					'layer': layer,
					'attention_override': layer_attention_override,
					'attention_override_mask': attention_override_mask
				}]
				candidate1_probs_layer[layer], candidate2_probs_layer[layer] = self.attention_intervention(
					context=context,
					outputs=intervention.candidates_tok,
					attn_override_data = layer_attn_override_data)
				model_attn_override_data.extend(layer_attn_override_data)
				for head in range(self.num_heads):
					if self.is_bert or self.is_distilbert or self.is_roberta:
						attention_override_mask = [torch.zeros_like(l, dtype=torch.uint8)
												   for l in layer_attention_override]
						for a in attention_override_mask: a[0][head] = 1
					elif self.is_xlnet:
						attention_override_mask = torch.zeros_like(layer_attention_override[0], dtype=torch.uint8)
						attention_override_mask[0][head] = 1
					else:
						attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
						attention_override_mask[0][head] = 1 # Set mask to 1 for single head only
					head_attn_override_data = [{
						'layer': layer,
						'attention_override': layer_attention_override,
						'attention_override_mask': attention_override_mask
					}]
					candidate1_probs_head[layer][head], candidate2_probs_head[layer][head] = self.attention_intervention(
						context=context,
						outputs=intervention.candidates_tok,
						attn_override_data=head_attn_override_data)

			# Intervene on entire model by overlaying attention induced by x_alt
			candidate1_probs_model, candidate2_probs_model = self.attention_intervention(
				context=context,
				outputs=intervention.candidates_tok,
				attn_override_data=model_attn_override_data)

		return candidate1_probs_head, candidate2_probs_head, candidate1_probs_layer, candidate2_probs_layer,\
			candidate1_probs_model, candidate2_probs_model

	def attention_intervention_single_experiment(self, intervention, effect, layers_to_adj, heads_to_adj, search):
		"""
		Run one full attention intervention experiment
		measuring indirect or direct effect.
		"""
		# E.g. The doctor asked the nurse a question. He
		x = intervention.base_strings_tok[0]
		# E.g. The doctor asked the nurse a question. She
		x_alt = intervention.base_strings_tok[1]

		if effect == 'indirect':
			input = x_alt  # Get attention for x_alt
		elif effect == 'direct':
			input = x  # Get attention for x
		else:
			raise ValueError(f"Invalid effect: {effect}")
		batch = torch.tensor(input).unsqueeze(0).to(self.device)
		attention_override = self.model(batch)[-1]

		batch_size = 1
		seq_len = len(x)
		seq_len_alt = len(x_alt)
		assert seq_len == seq_len_alt
		assert len(attention_override) == self.num_layers
		assert attention_override[0].shape == (batch_size, self.num_heads, seq_len, seq_len)

		with torch.no_grad():
			if search:
				candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
				candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))

			if effect == 'indirect':
				context = x
			else:
				context = x_alt

			model_attn_override_data = []
			for layer in range(self.num_layers):
				if layer in layers_to_adj:
					layer_attention_override = attention_override[layer]

					layer_ind = np.where(layers_to_adj == layer)[0]
					heads_in_layer = heads_to_adj[layer_ind]
					attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
					# set multiple heads in layer to 1
					for head in heads_in_layer:
						attention_override_mask[0][head] = 1 # Set mask to 1 for single head only
					# get head mask
					head_attn_override_data = [{
						'layer': layer,
						'attention_override': layer_attention_override,
						'attention_override_mask': attention_override_mask
					}]
					# should be the same length as the number of unique layers to adj
					model_attn_override_data.extend(head_attn_override_data)

			# basically generate the mask for the layers_to_adj and heads_to_adj
			if search:
				for layer in range(self.num_layers):
				  layer_attention_override = attention_override[layer]
				  layer_ind = np.where(layers_to_adj == layer)[0]
				  heads_in_layer = heads_to_adj[layer_ind]

				  for head in range(self.num_heads):
					if head not in heads_in_layer:
						  model_attn_override_data_search = []
						  attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
						  heads_list = [head]
						  if len(heads_in_layer) > 0:
							heads_list.extend(heads_in_layer)
						  for h in (heads_list):
							  attention_override_mask[0][h] = 1 # Set mask to 1 for single head only
						  head_attn_override_data = [{
							  'layer': layer,
							  'attention_override': layer_attention_override,
							  'attention_override_mask': attention_override_mask
						  }]
						  model_attn_override_data_search.extend(head_attn_override_data)
						  for override in model_attn_override_data:
							  if override['layer'] != layer:
								  model_attn_override_data_search.append(override)

						  candidate1_probs_head[layer][head], candidate2_probs_head[layer][head] = self.attention_intervention(
							  context=context,
							  outputs=intervention.candidates_tok,
							  attn_override_data=model_attn_override_data_search)
					else:
						candidate1_probs_head[layer][head] = -1
						candidate2_probs_head[layer][head] = -1


			else:
			  candidate1_probs_head, candidate2_probs_head = self.attention_intervention(
				  context=context,
				  outputs=intervention.candidates_tok,
				  attn_override_data=model_attn_override_data)

		return candidate1_probs_head, candidate2_probs_head
