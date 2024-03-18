
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

from rumourObj import Tweet, Intervention,TweetLite, GeneralModel

np.random.seed(5)
torch.manual_seed(5)



class Model():
	'''
	Wrapper for all model logic
	'''
	def __init__(self,
				 device='cuda:0',
				 output_attentions=False,
				 random_weights=False,
				 masking_approach=1,
				 model_version='bert-base',
				 load_pretrained_model=True,
				 pretrained_model='res/roberta_causal/'):
		super()

		self.is_gpt2 = (model_version.startswith('gpt2') or
						model_version.startswith('distilgpt2'))
		self.is_bert = model_version.startswith('bert')
		self.is_distilbert = model_version.startswith('distilbert')
		self.is_roberta = model_version.startswith('roberta')
		self.is_tt = model_version.startswith('twotier')
		self.is_duck = model_version.startswith('duck')
		assert (self.is_gpt2 or self.is_txl or self.is_xlnet or
				self.is_bert or self.is_distilbert or self.is_roberta
				or self.is_tt or self.is_duck)

		self.device = device
		self.load_pretrained_model = load_pretrained_model
		self.pretrained_model = pretrained_model



		tokenizer = (GPT2Tokenizer if self.is_gpt2 else
					 BertTokenizer if self.is_bert else
					 DistilBertTokenizer if self.is_distilbert else
					 RobertaTokenizer if self.is_roberta else
					 BertTokenizer).from_pretrained(model_version)

		# Add Special tokens
		#For the tokenizer, we need to add [MASK] and [CMT] to it
		tokenizer.add_tokens(["[MASK]","[CMT]"])


		self.model = GeneralModel(model_name=model_version, tokenizer=tokenizer, device=device, load_pretrained_model=load_pretrained_model)
		self.model.resize_token_embeddings(len(tokenizer))
		self.model.embeddings.word_embeddings.padding_idx=1
		self.model.eval()
		self.model.to(device)

		self.tokenizer = tokennizer

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

		if self.is_gpt2:
			self.attention_layer = lambda layer: self.model.transformer.h[layer].attn
			self.word_emb_layer = self.model.transformer.wte
			self.neuron_layer = lambda layer: self.model.transformer.h[layer].mlp
		elif self.is_bert:
			self.attention_layer = lambda layer: self.model.bert.encoder.layer[layer].attention.self
			self.word_emb_layer = self.model.bert.embeddings.word_embeddings
			self.neuron_layer = lambda layer: self.model.bert.encoder.layer[layer].output
		elif self.is_distilbert:
			self.attention_layer = lambda layer: self.model.distilbert.transformer.layer[layer].attention
			self.word_emb_layer = self.model.distilbert.embeddings.word_embeddings
			self.neuron_layer = lambda layer: self.model.distilbert.transformer.layer[layer].output_layer_norm
		elif self.is_roberta:
			self.attention_layer = lambda layer: self.model.roberta.encoder.layer[layer].attention.self
			self.word_emb_layer = self.model.roberta.embeddings.word_embeddings
			self.neuron_layer = lambda layer: self.model.roberta.encoder.layer[layer].output
		elif self.is_tt:
			self.attention_layer = lambda layer: self.model.bert.encoder.layer[layer].attention.self
			self.word_emb_layer = self.model.bert.embeddings.word_embeddings
			self.neuron_layer = lambda layer: self.model.bert.encoder.layer[layer].output
		elif self.is_duck:
			self.attention_layer = lambda layer: self.model.bert.encoder.layer[layer].attention.self
			self.word_emb_layer = self.model.bert.embeddings.word_embeddings
			self.neuron_layer = lambda layer: self.model.bert.encoder.layer[layer].output

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

	# The function of get_representations [CLS] token for each layer

	def get_representations_lite(self, input_ids, attention_masks):
		representations = {}

		with torch.no_grad():
			outputs = self.roberta(input_ids,attention_masks)

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


	def get_representations_full_old(self, input_ids, attention_masks):
		# Hook for saving the representation
		def extract_representation_hook(module, input, output,representations, layer):
			representations[layer] = output[self.order_dims(0,)]
		handles = []
		representations = {}

		with torch.no_grad():
			# construct all the hoooks
			handles.append(self.word_emb_layer.register_forward_hook(partial(extract_representation_hook,representations=representations,layer=-1)))
			# hidden layers
			for layer in range(self.num_layers):
				handles.append(self.neuron_layer(layer).register_forward_hook(partial(extract_representation_hook,representations=representations,layer=layer)))
			self.roberta(input_ids,attention_masks)
			for h in handles:
				h.remove()
		return representations


	def get_representations_og(self, context, position):
		# Hook for saving the representation
		def extract_representation_hook(module,
										input,
										output,
										position,
										representations,
										layer):
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

	

	def neuron_intervention(self,
							context,
							outputs,
							rep,
							layers,
							neurons,
							position,
							intervention_type='diff',
							alpha=1.):
		# Hook for changing representation during forward pass
		print('number of neurons ', len(neurons))
		def intervention_hook(module,
							  input,
							  output,
							  position,
							  neurons,
							  intervention,
							  intervention_type):
			# Get the neurons to intervene on
			neurons = torch.LongTensor(neurons).to(self.device)
			# First grab the position across batch
			# Then, for each element, get correct index w/ gather
			base_slice = self.order_dims((slice(None), position, slice(None)))
			base = output[base_slice].gather(1, neurons)
			intervention_view = intervention.view_as(base)

			if intervention_type == 'replace':
				base = intervention_view
			elif intervention_type == 'diff':
				base += intervention_view
			else:
				raise ValueError(f"Invalid intervention_type: {intervention_type}")
			# Overwrite values in the output
			# First define mask where to overwrite
			scatter_mask = torch.zeros_like(output, dtype=torch.bool)
			for i, v in enumerate(neurons):
				scatter_mask[self.order_dims((i, position, v))] = 1
			# Then take values from base and scatter
			output.masked_scatter_(scatter_mask, base.flatten())

		# Set up the context as batch
		batch_size = len(neurons)

		context = context.unsqueeze(0)

		handle_list = []
		for layer in set(layers):
			neuron_loc = np.where(np.array(layers) == layer)[0]
			n_list = []
			for n in neurons:
				unsorted_n_list = [n[i] for i in neuron_loc]
				n_list.append(list(np.sort(unsorted_n_list)))
			intervention_rep = alpha * rep[layer][n_list]
			if layer == -1:
				handle_list.append(self.word_emb_layer.register_forward_hook(
					partial(intervention_hook,
							position=position,
							neurons=n_list,
							intervention=intervention_rep,
							intervention_type=intervention_type)))
			else:
				handle_list.append(self.neuron_layer(layer).register_forward_hook(
					partial(intervention_hook,
							position=position,
							neurons=n_list,
							intervention=intervention_rep,
							intervention_type=intervention_type)))
		new_probabilities = self.get_probabilities_for_rumours(
			context,
			attention_masks=None)
		for hndle in handle_list:
			hndle.remove()
		return new_probabilities

	def head_pruning_intervention(self,
								  context,
								  outputs,
								  layer,
								  head,
								  model_name, tokenizer, load_pretrained_model,device):
		# Recreate model and prune head
		save_model = self.model

		self.model = GeneralModel(model_name=model_name, tokenizer=tokenizer, device=device, load_pretrained_model=load_pretrained_model)
		self.model.prune_heads({layer: [head]})
		self.model.eval()

		# Compute probabilities without head
		new_probabilities = self.get_probabilities_for_examples(
			context,
			outputs)

		# Reinstate original model
		self.model = save_model

		return new_probabilities

	def attention_intervention(self,
							   context,
							   outputs,
							   attn_override_data):
		#Override attention values in specified layer

		def intervention_hook(module, input, outputs, attn_override, attn_override_mask):
			attention_override_module = (AttentionOverride if self.is_gpt2 else
										 TXLAttentionOverride if self.is_txl else
										 XLNetAttentionOverride if self.is_xlnet else
										 BertAttentionOverride if self.is_bert else
										 DistilBertAttentionOverride if self.is_distilbert else
										 BertAttentionOverride)(
				module, attn_override, attn_override_mask)
			return attention_override_module(*input)

		with torch.no_grad():
			if self.is_bert or self.is_distilbert or self.is_roberta:
				k = 0
				new_probabilities = []
				context = context.tolist()
				for candidate in outputs:
					token_log_probs = []
					mlm_inputs = self.mlm_inputs(context, candidate)
					for i, c in enumerate(candidate):
						hooks = []
						for d in attn_override_data:
							hooks.append(self.attention_layer(d['layer']).register_forward_hook(
								partial(intervention_hook,
										attn_override=d['attention_override'][k],
										attn_override_mask=d['attention_override_mask'][k])))

						combined, pred_idx = mlm_inputs[i]
						batch = torch.tensor(combined).unsqueeze(dim=0).to(self.device)
						logits = self.model(batch)[0]
						log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
						token_log_probs.append(log_probs[pred_idx][c].item())

						for hook in hooks: hook.remove()
						k += 1

					mean_token_log_prob = statistics.mean(token_log_probs)
					mean_token_prob = math.exp(mean_token_log_prob)
					new_probabilities.append(mean_token_prob)
			else:
				hooks = []
				for d in attn_override_data:
					attn_override = d['attention_override']
					attn_override_mask = d['attention_override_mask']
					layer = d['layer']
					hooks.append(self.attention_layer(layer).register_forward_hook(
						partial(intervention_hook,
								attn_override=attn_override,
								attn_override_mask=attn_override_mask)))

				new_probabilities = self.get_probabilities_for_examples_multitoken(
					context,
					outputs)

				for hook in hooks:
					hook.remove()

			return new_probabilities


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



	#this intervention only care about the total effect
	def total_effect_experiment(self, intervention_lst):
		results = {}
		for comm in tqdm(intervention_lst, desc='total_intervention'):
			results[comm] = {}
			base_prob, alt_prob = self.total_effect_experiment_run(intervention_lst[comm])
			results[comm]['base_prob'] = base_prob
			results[comm]['alt_prob'] = alt_prob
		return results

	def total_effect_experiment_run(self, intervention):
		base_prob = self.get_probabilities_for_rumours(intervention.source_input_ids.unsqueeze(0), intervention.source_attention_masks.unsqueeze(0))[0]
		alt_prob = self.get_probabilities_for_rumours(intervention.altered_input_ids.unsqueeze(0), intervention.altered_attention_mask.unsqueeze(0))[0]
		return base_prob, alt_prob

	## this one will replace each neuron in the intervention input sequence ######
	def total_neuron_experiment(self, intervention_lst,intervention_type, layers_to_adj=[], neurons_to_adj=[], alpha=1, intervention_loc='layer'):
		results = {}
		for comm in tqdm(intervention_lst, desc='pair_intervention'):
			#print('comm ', comm)
			results[comm] = {}
			(base_prob, alt_prob, intervention_res) = self.total_neuron_experiment_run(
				intervention_lst[comm], intervention_type, layers_to_adj, neurons_to_adj,
				alpha, intervention_loc=intervention_loc)
			results[comm]['base_prob'] = base_prob
			results[comm]['alt_prob'] = alt_prob
			results[comm]['intervention_res'] = intervention_res
		return results

	def total_neuron_experiment_run(self, intervention,intervention_type, layers_to_adj=[],neurons_to_adj=[],alpha=100,bsize=800, intervention_loc='layer'):
		  base_representations = self.get_representations_full(
				intervention.source_input_ids.unsqueeze(0),
				intervention.source_attention_masks.unsqueeze(0))
		  modified_representations = self.get_representations_full(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))
		  
		  #context = intervention.og_input_ids
		  base_rep = base_representations
		  altered_rep = modified_representations

		  base_prob = self.get_probabilities_for_rumours(
			intervention.source_input_ids.unsqueeze(0),
			intervention.source_attention_masks.unsqueeze(0))[0]

		  alt_prob = self.get_probabilities_for_rumours(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))[0]

		  if intervention_loc == 'layer':
			intervention_res = {}
			for layer in range(-1, self.num_layers):
				for neurons in batch(range(self.num_neurons), bsize):
					neurons_to_search = [[i] + neurons_to_adj for i in neurons]
				probs = self.rumour_direct_each_nueron_intervention(
				  intervention = intervention,
				  layer = layer,
				  base_rep=base_rep,
				  altered_rep=altered_rep,
				  intervention_type=intervention_type,
				  alpha=alpha,batch_size=bsize)
				print('LAYER ', layer)
				print('PROBS ', probs)
				intervention_res[layer] = probs
		  return (base_prob,alt_prob,intervention_res)



	def neuron_intervention_pair_experiment(self, intervention_lst, intervention_type, layers_to_adj=[], neurons_to_adj=[], alpha=1, intervention_loc='layer'):
		'''
		Run multiple intervention experiments
		'''
		# For the first set of results, we only care about the total effect when [MASK] each conversation 
		# The total effect score ranked can be used for turn around point accuracy calculation

		results = {}
		for comm in tqdm(intervention_lst, desc='pair_intervention'):
			print('comm ', comm)
			results[comm] = {}
			(base_prob, alt_prob, intervention_res) = self.neuron_intervention_pair_experiment_veracity(
				intervention_lst[comm], intervention_type, layers_to_adj, neurons_to_adj,
				alpha, intervention_loc=intervention_loc)
			results[comm]['base_prob'] = base_prob
			results[comm]['alt_prob'] = alt_prob
			results[comm]['intervention_res'] = intervention_res
		return results

	def neuron_intervention_pair_experiment_veracity(self, 
												intervention,
												intervention_type, layers_to_adj=[],
												neurons_to_adj=[],
												alpha=100,bsize=800, intervention_loc='layer'):
		  base_representations = self.get_representations_full(
				intervention.source_input_ids.unsqueeze(0),
				intervention.source_attention_masks.unsqueeze(0))
		  modified_representations = self.get_representations_full(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))
		  
		  #context = intervention.og_input_ids
		  base_rep = base_representations
		  altered_rep = modified_representations

		  base_prob = self.get_probabilities_for_rumours(
			intervention.source_input_ids.unsqueeze(0),
			intervention.source_attention_masks.unsqueeze(0))[0]

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

	def neuron_intervention_each_experiment_veracity(self, 
												intervention,
												intervention_type, layers_to_adj=[],
												neurons_to_adj=[],
												alpha=100,bsize=800, intervention_loc='layer'):
		  base_representations = self.get_representations_full(
				intervention.source_input_ids.unsqueeze(0),
				intervention.source_attention_masks.unsqueeze(0))
		  modified_representations = self.get_representations_full(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))
		
		  #context = intervention.og_input_ids
		  base_rep = base_representations
		  altered_rep = modified_representations

		  base_prob = self.get_probabilities_for_rumours(
			intervention.source_input_ids.unsqueeze(0),
			intervention.source_attention_masks.unsqueeze(0))[0]

		  alt_prob = self.get_probabilities_for_rumours(
			  intervention.altered_input_ids.unsqueeze(0),
			  intervention.altered_attention_mask.unsqueeze(0))[0]

		  if intervention_loc == 'layer':
			intervention_res = {}
			for layer in range(-1, self.num_layers):
				for neurons in batch(range(self.num_neurons), bsize):
					neurons_to_search = [[i] + neurons_to_adj for i in neurons]
				probs = self.rumour_direct_each_nueron_intervention(
				  intervention = intervention,
				  layer = layer,
				  base_rep=base_rep,
				  altered_rep=altered_rep,
				  intervention_type=intervention_type,
				  alpha=alpha)
				intervention_res[layer] = probs
		  return (base_prob,alt_prob,intervention_res)


	def base_neuron_intervention_experiment(self, comments2intervention):
		comments2intervention_results = {}
		for comm in tqdm(comments2intervention, desc='comments_base'):
			comments2intervention_results[comm] = {}
			base_prob = self.base_neuron_intervention_single_comment_experiment(comments2intervention[comm])
			comments2intervention_results[comm]['base_prob'] = base_prob
		return comments2intervention_results

	def base_neuron_intervention_single_comment_experiment(self,intervention):
		base_prob = self.get_probabilities_for_rumours(
			intervention.og_input_ids.unsqueeze(0),
			intervention.og_attention_mask.unsqueeze(0))[0]
		return base_prob

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
				print('LAYER ', layer)
				print('PROBS ', probs)
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
			# Get the neurons to intervene on
			target_pos = intervention.target_locations
			# First define mask where to overwrite
			scatter_mask = torch.zeros_like(output, dtype=torch.bool)
			if intervention_type == 'indirect':
				base = layer_altered_rep

				for target_l in target_pos:
					base[0,target_l,:] = layer_base_rep[0,target_l,:]
					scatter_mask[0,target_l,:] = 1
			elif intervention_type == 'direct':
				base = layer_base_rep
				for target_l in target_pos:
					base[0,target_l,:] = layer_altered_rep[0,target_l,:]
					scatter_mask[0,target_l,:] = 1
			else:
				raise ValueError(f"Invalid intervention_type: {intervention_type}")
			
			# Then take values from base and scatter
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
		if intervention_type == 'direct':# NEED TO REWIRE this one here!!! --DONE
			new_probabilities = self.get_probabilities_for_rumours(intervention.source_input_ids.unsqueeze(0),attention_masks=intervention.source_attention_masks.unsqueeze(0))
		if intervention_type == 'indirect':
			new_probabilities = self.get_probabilities_for_rumours(intervention.altered_input_ids.unsqueeze(0),attention_masks=intervention.altered_attention_mask.unsqueeze(0))
		for hndle in handle_list:
			hndle.remove()
		return new_probabilities




	def rumour_direct_each_nueron_intervention(self, intervention, layer, base_rep, altered_rep, intervention_type,alpha=1.,batch_size):
		# For this intervention, what we try to do is to do each neuron at a time
		def intervention_hook(module,
							  input,
							  output,
							  intervention_loc,
							  layer_base_rep,
							  layer_altered_rep,
							  intervention_type):

			scatter_mask = torch.zeros_like(output, dtype=torch.bool)
			#print('before scatter_mask shape ', scatter_mask.shape) 
			if intervention_type == 'indirect':
				base = layer_altered_rep
				base[0,intervention_loc,:] = layer_base_rep[0,intervention_loc,:]
				scatter_mask[0,intervention_loc,:] = 1
			elif intervention_type == 'direct':
				base = layer_base_rep
				base[0,intervention_loc,:] = layer_altered_rep[0,intervention_loc,:]
				scatter_mask[0,intervention_loc,:] = 1
			else:
				raise ValueError(f"Invalid intervention_type: {intervention_type}")
			
			# Then take values from base and scatter
			output.masked_scatter_(scatter_mask, base.flatten())

		# Set up the context as batch
		batch_size = batch_size
		handle_list = []

		if layer == -1:
			layer_base_rep = base_rep[0]
			layer_altered_rep = altered_rep[0]
			target_pos = intervention.target_locations
			for intervention_loc in target_pos:
				handle_list.append(self.word_emb_layer.register_forward_hook(
					partial(intervention_hook,
							intervention_loc=intervention_loc,
							layer_base_rep=layer_base_rep,
							layer_altered_rep=layer_altered_rep,
							intervention_type=intervention_type)))
		else:
			layer_base_rep = base_rep[layer]
			layer_altered_rep = altered_rep[layer]
			target_pos = intervention.target_locations
			for intervention_loc in target_pos:
				handle_list.append(self.neuron_layer(layer).register_forward_hook(
					partial(intervention_hook,
							intervention_loc=intervention_loc,
							layer_base_rep=layer_base_rep,
							layer_altered_rep=layer_altered_rep,
							intervention_type=intervention_type)))
		new_probabilities = []
		if intervention_type == 'direct':
			new_probabilities = self.get_probabilities_for_rumours(intervention.source_input_ids.unsqueeze(0),attention_masks=intervention.source_attention_masks.unsqueeze(0))
		if intervention_type == 'indirect':
			new_probabilities = self.get_probabilities_for_rumours(intervention.altered_input_ids.unsqueeze(0),attention_masks=intervention.altered_attention_mask.unsqueeze(0))
		for hndle in handle_list:
			hndle.remove()
		return new_probabilities



	def rumour_neuron_intervention(self, intervention, layer, neurons, rep, intervention_type='diff',alpha=1.):
		def intervention_hook(module, input, output, position,neurons, intervention, intervention_type):
			# what we need to intervent is for each layer, we figure out the 
			# XLNet: ignore the query stream
			if self.is_xlnet and output.shape[0] == 1: return output
			# Get the neurons to intervene on
			neurons = torch.LongTensor(neurons).to(self.device)
			# First grab the position across batch
			# Then, for each element, get correct index w/ gather
			base_slice = self.order_dims((slice(None), position, slice(None)))
			base = output[base_slice].gather(1, neurons)
			intervention_view = intervention.view_as(base)

			if intervention_type == 'replace':
				base = intervention_view
			elif intervention_type == 'diff':
				base += intervention_view
			else:
				raise ValueError(f"Invalid intervention_type: {intervention_type}")
			# Overwrite values in the output
			# First define mask where to overwrite
			scatter_mask = torch.zeros_like(output, dtype=torch.bool)
			for i, v in enumerate(neurons):
				scatter_mask[self.order_dims((i, position, v))] = 1
			# Then take values from base and scatter
			output.masked_scatter_(scatter_mask, base.flatten())

		n_list = []
		handle_list = []
		layers = [-1,0,1,2,3,4,5,6,7,8,10,11]
		neuron_loc = np.where(np.array(layers) == layer)[0]
		print('grab rep keys ', rep.keys())
		for n in neurons:
			unsorted_n_list = [n[i] for i in neuron_loc]
			n_list.append(list(np.sort(unsorted_n_list)))
		intervention_rep = intervention.altered_input_ids
		if layer == -1:
			handle_list.append(self.word_emb_layer.register_forward_hook(
				  partial(intervention_hook,
						  position=intervention.target_locations,
						  neurons=n_list,
						  intervention=intervention_rep,
						  intervention_type=intervention_type)))
		else:
			handle_list.append(self.neuron_layer(layer).register_forward_hook(
				  partial(intervention_hook,
						  position=intervention.target_locations,
						  neurons=n_list,
						  intervention=intervention_rep,
						  intervention_type=intervention_type)))
		new_probabilities = self.get_probabilities_for_rumours(intervention.og_input_ids.unsqueeze(0),attention_masks=None)
		for hndle in handle_list:
			hndle.remove()
		return new_probabilities
		  

	def neuron_intervention_single_experiment(self,
											  intervention,
											  intervention_type, layers_to_adj=[],
											  neurons_to_adj=[],
											  alpha=100,
											  bsize=800, intervention_loc='layer'):
		"""
		run one full neuron intervention experiment
		"""

		if self.is_txl or self.is_xlnet: 32 # to avoid GPU memory error
		with torch.no_grad():
			'''
			Compute representations for base terms (one for each side of bias)
			'''
			if self.is_bert or self.is_distilbert or self.is_roberta or self.is_xlnet:
				num_alts = 1

			base_representations = self.get_representations_lite(
				intervention.og_input_ids,
				intervention.og_attention_mask)
			modified_representations = self.get_representations_lite(
				intervention.altered_input_ids,
				intervention.altered_attention_mask)
			

			context = intervention.og_input_ids
			rep = base_representations
			replace_or_diff = 'replace'


			
			# Probabilities without intervention (Base case)

			candidate1_base_prob, candidate2_base_prob = self.get_probabilities_for_rumours(
			  intervention.og_input_ids,
			  intervention.og_attention_mask
			)[0]

			candidate1_alt1_prob, candidate2_alt1_prob = self.get_probabilities_for_rumours(
				intervention.altered_input_ids,
				intervention.altered_attention_mask)[0]

			if intervention_loc == 'all':
			  candidate1_probs = torch.zeros((self.num_layers + 1, self.num_neurons))
			  candidate2_probs = torch.zeros((self.num_layers + 1, self.num_neurons))
			  for layer in range(-1, self.num_layers):
				for neurons in batch(range(self.num_neurons), bsize):
					neurons_to_search = [[i] + neurons_to_adj for i in neurons]
					layers_to_search = [layer] + layers_to_adj

					probs = self.neuron_intervention(
						context=context,
						outputs=intervention.label_tok,
						rep=rep,
						layers=layers_to_search,
						neurons=neurons_to_search,
						position=intervention.position_lst[0],
						intervention_type=replace_or_diff,
						alpha=alpha)
					for neuron, (p1, p2) in zip(neurons, probs):
						candidate1_probs[layer + 1][neuron] = p1
						candidate2_probs[layer + 1][neuron] = p2
						# Now intervening on potentially biased example
			elif intervention_loc == 'layer':

			  layers_to_search = (len(neurons_to_adj) + 1)*[layers_to_adj]
			  candidate1_probs = torch.zeros((1, self.num_neurons))
			  candidate2_probs = torch.zeros((1, self.num_neurons))


			  for neurons in batch(range(self.num_neurons), bsize):
				neurons_to_search = [[i] + neurons_to_adj for i in neurons]

				probs = self.neuron_intervention(
					context=context,
					outputs=intervention.label_tok,
					rep=rep,
					layers=layers_to_search,
					neurons=neurons_to_search,
					position=intervention.position_lst[0],
					intervention_type=replace_or_diff,
					alpha=alpha)
				for neuron, (p1, p2) in zip(neurons, probs):
					candidate1_probs[0][neuron] = p1
					candidate2_probs[0][neuron] = p2
			else:
			  candidate1_probs = 0
			  candidate2_probs = 0 
			  probs = self.neuron_intervention(
						context=context,
						outputs=intervention.label_tok,
						rep=rep,
						layers=layers_to_adj,
						neurons=neurons_to_adj,
						position=0,
						intervention_type=replace_or_diff,
						alpha=alpha)
			  for neuron, (p1, p2) in zip(neurons_to_adj, probs):
				  candidate1_probs = p1
				  candidate2_probs = p2


		return (candidate1_base_prob, candidate2_base_prob,
				candidate1_alt1_prob, candidate2_alt1_prob,
				candidate1_probs, candidate2_probs)

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

