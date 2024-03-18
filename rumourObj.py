import torch

# a general model to load up all our models
from transformers import AutoModel, AutoModelForSequenceClassification

class GeneralModel:
    def __init__(self, model_name='bert-base-uncased', tokenizer=None, device='cuda:0', load_pretrained_model=False, pretrained_model=None):
        self.load_pretrained_model = load_pretrained_model
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        self.device = device

        if not self.load_pretrained_model:
            # Load model and model for sequence classification with Auto classes
            self.base_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Resize token embeddings if a tokenizer is provided
            if tokenizer is not None:
                self.model.resize_token_embeddings(len(tokenizer))
                self.base_model.resize_token_embeddings(len(tokenizer))
            
            # Set padding index if applicable (specific to models with word embeddings like BERT and RoBERTa)
            if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
                self.model.embeddings.word_embeddings.padding_idx = 1
                self.base_model.embeddings.word_embeddings.padding_idx = 1
            
            self.model.eval()
            self.model.to(self.device)
            self.base_model.to(self.device)
        else:
            # Load pretrained model with specified configuration
            self.base_model = AutoModel.from_pretrained(pretrained_model, output_hidden_states=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, output_hidden_states=True)
            
            if tokenizer is not None:
                self.model.resize_token_embeddings(len(tokenizer))
                self.base_model.resize_token_embeddings(len(tokenizer))
            
            if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
                self.model.embeddings.word_embeddings.padding_idx = 1
                self.base_model.embeddings.word_embeddings.padding_idx = 1

            self.model.eval()
            self.model.to(self.device)
            self.base_model.to(self.device)




class Tweet(object):
		def __init__(self, source_id, source_text, reply_text_lst, candidate_comments, is_rumour):
				self.source_id = source_id
				self.source_text = source_text
				self.reply_text_lst = reply_text_lst
				self.candidate_comments = candidate_comments
				self.is_rumour = is_rumour

#listofTweets = [(Tweet(row.source_id,row.source_text, row.reply_text_lst, row.is_rumour)) for index, row in df.iterrows() ]  

class Story(object):
	def __init__(self, story_id, story_content, source_text_lst, is_rumour, source_turnaround_lst):
				self.story_id = story_id
				self.story_content = story_content
				self.source_text_lst = source_text_lst
				self.is_rumour = is_rumour #[0,1,2]
				self.source_turnaround_lst = source_turnaround_lst



class TweetLite(object):
	def __init__(self, source_id, source_text, reply_text_lst, is_rumour):
			self.source_id = source_id
			self.source_text = source_text
			self.reply_text_lst = reply_text_lst
			self.is_rumour = is_rumour




def sep_token_help(input_ids, sepecial_token_id):
	#input_ids = input_ids.tolist()
	size = len(input_ids)
	#print('here is the size ', size)
	#input_ids = input_ids.tolist()
	#print('input_ids ',input_ids)
	#print('special_token_id in list ', sepecial_token_id in input_ids)
	idx_list = [idx + 1 for idx, val in enumerate(input_ids) if val == sepecial_token_id]
	#print('len of idx_list ',len(idx_list))
	res = [input_ids[i:j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
	return res

def convert_to_sublists(input_ids, sepecial_token_id):
	#print('converter ',len(input_ids))
	start = 0
	idx_list = [idx + 1 for idx, val in enumerate(input_ids) if val == sepecial_token_id]
	sublists = []
	#print('IM in converter ', len(idx_list))
	for i, item in enumerate(input_ids):
			if i in idx_list:
					end = i
					sublist = list(range(start, end+1))
					sublists.append(sublist)
					start = end + 1
	return sublists





def encode_mask_sentence(mask_tok, sentence, tokenizer,max_len):
	sentence_word_lst = sentence.split()
	sentence_len = len(sentence_word_lst)
	masked_sentence_lst = [mask_tok] * sentence_len
	masked_tok = tokenizer.encode_plus(masked_sentence_lst,  is_split_into_words=True,add_special_tokens=True, max_length=max_len, truncation_strategy='only_second',padding='max_length', pad_to_max_length=True)

	return masked_tok


def gen_mask_sentence(mask_tok, sentence):
	sentence_word_lst = sentence.split()
	sentence_len = len(sentence_word_lst)
	masked_sentence_lst = [mask_tok] * sentence_len
	masked_sentence_str = ''.join(masked_sentence_lst)
	return masked_sentence_lst, masked_sentence_str

class PairIntervention():
	def __init__(self, tokennizer, source_sentence: str, reaction_lst: list, alt_loc, gold_label, turnaround_label_lst: list, max_len, device):
		super()
		self.device = device
		self.enc = tokennizer
		self.custom_tokens = ["[CMT]","[MASK_SENT]"]
		self.max_len = max_len
		self.alt_loc = alt_loc

		self.enc.add_special_tokens({"additional_special_tokens": self.custom_tokens})

		source_tok = self.enc.encode_plus(text=source_sentence,
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = self.max_len,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.

		source_tok_input_ids = source_tok['input_ids']
		source_tok_attention_masks = source_tok['attention_mask']

		reaction_tok_input_ids = []
		reaction_tok_attention_mask = []
		tokenized_reactions = [self.enc.encode_plus(text, add_special_tokens=True, max_length=self.max_len, truncation_strategy='only_second',padding='max_length', pad_to_max_length=True) for text in reaction_lst]
		#debug
		#print('len of tokenized_rections ', len(tokenized_reactions))
		for t_r in tokenized_reactions:
			reaction_tok_input_ids.append(t_r['input_ids'])
			reaction_tok_attention_mask.append(t_r['attention_mask'])


		#altered_tokenized_reactions = tokenized_reactions.copy()
		masked_tok = encode_mask_sentence(self.custom_tokens[0], reaction_lst[alt_loc], self.enc, self.max_len)

		altered_input_ids = []
		altered_attention_mask = []
		for index, val in enumerate(tokenized_reactions):
			if index == alt_loc:
				altered_input_ids.append(masked_tok['input_ids'])
				altered_attention_mask.append(masked_tok['attention_mask'])
			else:
				altered_input_ids.append(val['input_ids'])
				altered_attention_mask.append(val['attention_mask'])


		#print('len of altered_tokenized_reactions ',len(altered_tokenized_reactions))




		self.source_input_ids = torch.tensor(source_tok_input_ids).to(device)
		self.source_attention_masks = torch.tensor(source_tok_attention_masks).to(device)
		#print('self.source_input_ids ', self.source_input_ids.shape)
		self.reaction_tok_input_ids = torch.tensor(reaction_tok_input_ids).to(device)
		self.reaction_tok_attention_mask = torch.tensor(reaction_tok_attention_mask).to(device)
		#print('self.reaction_tok_input_ids ', self.reaction_tok_input_ids.shape)
		self.altered_reaction_tok_input_ids = torch.tensor(altered_input_ids).to(device)
		self.altered_reaction_tok_attention_mask = torch.tensor(altered_attention_mask).to(device)



		#self.rumour_label = gold_label
		self.label_tok = torch.tensor(gold_label,dtype=torch.int8).to(device)
		self.turnaround_lst = turnaround_label_lst
		self.turnaround_tok = torch.tensor(self.turnaround_lst).to(device)



class TotalIntervention():
	def __init__(self, tokennizer, source_sentence: str, reaction_lst: list, alt_loc, gold_label, turnaround_label_lst: list, max_len, device):
		super()
		self.device = device
		self.enc = tokennizer
		self.custom_tokens = ["[CMT]","[MASK_SENT]","[MASKTOK]"]
		self.max_len = max_len
		self.alt_loc = alt_loc

		self.enc.add_special_tokens({"additional_special_tokens": self.custom_tokens})
		reaction_lst_str = ' '.join(reaction_lst)

		source_tok = self.enc.encode_plus(text=source_sentence,
												text_pair=reaction_lst_str,
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = self.max_len,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.

		source_tok_input_ids = source_tok['input_ids']
		source_tok_attention_masks = source_tok['attention_mask']


		altered_reaction_lst = []
		position_lst = []
		for index, value in enumerate(reaction_lst):
			if index == alt_loc:
				masked_lst, masked_ss = gen_mask_sentence(self.custom_tokens[0], value)
				altered_reaction_lst.append(masked_ss)
				position_lst.extend(len(masked_lst)*[1])
			else:
				altered_reaction_lst.append(value)
				value_lst = value.split()
				position_lst.extend(len(value_lst)*[0])


		altered_indices = [i for i, pos in enumerate(position_lst) if pos == 1]
		self.target_locations = altered_indices

		altered_reaction_str = ' '.join(altered_reaction_lst)

		altered_tok = self.enc.encode_plus(
			text = source_sentence,
			text_pair = altered_reaction_str,
			add_special_tokens = True,
			max_length = self.max_len,
			truncation_strategy = 'only_second',
			padding = 'max_length',
			pad_to_max_length=True)
		altered_tok_input_ids = altered_tok['input_ids']
		altered_tok_attention_mask = altered_tok['attention_mask']

		self.source_input_ids = torch.tensor(source_tok_input_ids).to(device)
		self.source_attention_masks = torch.tensor(source_tok_attention_masks).to(device)

		self.altered_input_ids = torch.tensor(altered_tok_input_ids).to(device)
		self.altered_attention_mask = torch.tensor(altered_tok_attention_mask).to(device)


		#self.rumour_label = gold_label
		self.label_tok = torch.tensor(gold_label,dtype=torch.int8).to(device)
		self.turnaround_lst = turnaround_label_lst
		self.turnaround_tok = torch.tensor(self.turnaround_lst).to(device)




class TokenIntervention():
	def __init__(self, tokennizer, source_sentence: str, reaction_sentence: str, alt_loc, gold_label, turnaround_label_lst: list, max_len, device):
		super()
		self.device = device
		self.enc = tokennizer
		self.custom_tokens = ["[CMT]","[MASK_SENT]","[MASKTOK]"]
		self.max_len = max_len
		self.alt_loc = alt_loc

		self.enc.add_special_tokens({"additional_special_tokens": self.custom_tokens})

		source_tok = self.enc.encode_plus(text=source_sentence,
												text_pair=reaction_sentence,
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = self.max_len,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.

		source_tok_input_ids = source_tok['input_ids']
		source_tok_attention_masks = source_tok['attention_mask']

		reaction_sentence_lst = reaction_sentence.split()
		reaction_sentence_lst[alt_loc] = self.custom_tokens[2]
		altered_reaction_sentence = ' '.join(reaction_sentence_lst)

		self.target_locations = [alt_loc]


		altered_tok = self.enc.encode_plus(
			text = source_sentence,
			text_pair = altered_reaction_sentence,
			add_special_tokens = True,
			max_length = self.max_len,
			truncation_strategy = 'only_second',
			padding = 'max_length',
			pad_to_max_length=True)
		altered_tok_input_ids = altered_tok['input_ids']
		altered_tok_attention_mask = altered_tok['attention_mask']

		self.source_input_ids = torch.tensor(source_tok_input_ids).to(device)
		self.source_attention_masks = torch.tensor(source_tok_attention_masks).to(device)

		self.altered_input_ids = torch.tensor(altered_tok_input_ids).to(device)
		self.altered_attention_mask = torch.tensor(altered_tok_attention_mask).to(device)


		#self.rumour_label = gold_label
		self.label_tok = torch.tensor(gold_label,dtype=torch.int8).to(device)
		self.turnaround_lst = turnaround_label_lst
		self.turnaround_tok = torch.tensor(self.turnaround_lst).to(device)

class PairInterventionV2():
	def __init__(self, tokennizer, source_sentence: str, reaction_lst: list, alt_loc, gold_label, turnaround_label_lst: list, max_len, device):
		super()
		self.device = device
		self.enc = tokennizer
		self.custom_tokens = ["[CMT]","[MASK_SENT]","[MASKTOK]"]
		self.max_len = max_len
		self.alt_loc = alt_loc

		self.enc.add_special_tokens({"additional_special_tokens": self.custom_tokens})
		reaction_lst_str = ' '.join(reaction_lst)

		source_tok = self.enc.encode_plus(text=source_sentence,
												text_pair=reaction_lst_str,
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = self.max_len,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.

		source_tok_input_ids = source_tok['input_ids']
		source_tok_attention_masks = source_tok['attention_mask']


		altered_reaction_lst = []
		marked_lst = []
		for index, value in enumerate(reaction_lst):
			if index == alt_loc:
				masked_lst, masked_ss = gen_mask_sentence(self.custom_tokens[0], value)
				altered_reaction_lst.append(masked_ss)
				masked_lst.append(len(masked_lst)*[1])
			else:
				altered_reaction_lst.append(value)
				value_lst = value.split()
				masked_lst.append(len(value_lst)*[0])

		altered_indices = [i for i, pos in enumerate(masked_lst) if pos == 1]
		self.target_locations = altered_indices

		altered_reaction_str = ' '.join(altered_reaction_lst)

		altered_tok = self.enc.encode_plus(
			text = source_sentence,
			text_pair = altered_reaction_str,
			add_special_tokens = True,
			max_length = self.max_len,
			truncation_strategy = 'only_second',
			padding = 'max_length',
			pad_to_max_length=True)
		altered_tok_input_ids = altered_tok['input_ids']
		altered_tok_attention_mask = altered_tok['attention_mask']

		self.source_input_ids = torch.tensor(source_tok_input_ids).to(device)
		self.source_attention_masks = torch.tensor(source_tok_attention_masks).to(device)

		self.altered_input_ids = torch.tensor(altered_tok_input_ids).to(device)
		self.altered_attention_mask = torch.tensor(altered_tok_attention_mask).to(device)


		#self.rumour_label = gold_label
		self.label_tok = torch.tensor(gold_label,dtype=torch.int8).to(device)
		self.turnaround_lst = turnaround_label_lst
		self.turnaround_tok = torch.tensor(self.turnaround_lst).to(device)

class NodeIntervention():
	def __init__(self, tokenizer, parent_content: str, child_content: str, custom_tokens:list, alt_loc, is_turn_label, max_len, device):
		super()
		self.device = device
		self.enc = tokenizer
		self.custom_tokens = custom_tokens
		self.max_len = max_len

		self.enc.add_special_tokens({"additional_special_tokens": self.custom_tokens})

		pair_tok = self.enc.encode_plus(text=parent_content,
												text_pair= child_content,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = max_len,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.)
		og_pair_input_ids = pair_tok['input_ids']
		og_pair_attention_masks = pair_tok['attention_mask']



		# Identify the boundary index
		boundary_idx = pair_input_ids.index(self.enc.sep_token_id)

		# Create a copy of the tokenized input
		pair_input_ids_clone = og_pair_input_ids.clone()

		if alt_loc == 'parent':
			masked_tok = encode_mask_sentence(self.custom_tokens[0], parent_content, self.enc)
		elif alt_loc == 'child':
			masked_tok = encode_mask_sentence(self.custom_tokens[0], child_content, self.enc)


		# Replace the tokens in the first part with [MASK] tokens
		masked_input[1:boundary_idx] = self.enc.mask_token_id

		# Update attention mask for [MASK] tokens
		attention_mask[1:boundary_idx] = 1



		self.og_pair_input_ids = torch.tensor(og_pair_input_ids).to(device)
		self.og_pair_attention_masks = torch.tensor(og_pair_attention_masks).to(device)


		if is_turn_label == True:
			self.turn_label = 1
		else:
			self.turn_label = 0

		self.turn_label_tok = torch.tensor(self.turn_label).to(device)





class GraphIntervention():
	def __init__(self, tokennizer, story_content: str, reaction_lst: list, custom_tokens:list, alt_loc, gold_label,max_len, device):
		super()
		self.device = device
		self.enc = tokennizer
		self.custom_tokens = custom_tokens
		self.max_len = max_len

		self.enc.add_special_tokens({"additional_special_tokens": self.custom_tokens})

		source_tok = self.enc.encode_plus(text=story_content,
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = self.max_len,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.

		source_tok_input_ids = source_tok['input_ids']
		source_tok_attention_masks = source_tok['attention_mask']

		reaction_tok_input_ids = []
		reaction_tok_attention_mask = []
		tokenized_reactions = [self.enc.encode_plus(text, add_special_tokens=True, max_length=self.max_len, truncation_strategy='only_second',padding='max_length', pad_to_max_length=True) for text in reaction_list]

		for t_r in tokenizer_reactions:
			reaction_tok_input_ids.append(t_r['input_ids'])
			reaction_tok_attention_mask.append(t_r['attention_mask'])


		altered_tokenized_reactions = tokenizer_reactions
		masked_tok = encode_mask_sentence(self.custom_tokens[0], reaction_lst[alt_loc], self.enc)
		altered_tokenized_reactions[alt_loc] = masked_tok

		altered_reaction_tok_input_ids = []
		altered_reaction_tok_attention_mask = []

		for a_r in altered_tokenized_reactions:
			altered_reaction_tok_input_ids.append(a_r['input_id'])
			altered_reaction_tok_attention_mask.append(a_r['attention_mask'])




		self.source_input_ids = torch.tensor(source_tok_input_ids).to(device)
		self.source_attention_masks = torch.tensor(source_tok_attention_masks).to(device)
		self.reaction_tok_input_ids = torch.tensor(reaction_tok_input_ids).to(device)
		self.reaction_tok_attention_mask = torch.tensor(reaction_tok_attention_mask).to(device)
		self.altered_reaction_tok_input_ids = torch.tensor(altered_reaction_tok_input_ids).to(device)
		self.altered_reaction_tok_attention_mask = torch.tensor(altered_reaction_tok_attention_mask).to(device)

		if gold_label == 'rumour':
					self.label_tok = 1
		else:
					self.label_tok = 0
		self.label_tok = torch.tensor(self.label_tok).to(device)






class StoryIntervention():
	def __init__(self, tokennizer, story_content: str, reactions_lst: list, alt_loc, gold_label, device='cuda:0'):
			super()
			self.device = device
			self.enc = tokennizer

			custom_tokens = ["[MASK]","[CMT]"]
			self.enc.add_tokens(["[MASK]","[CMT]"])
			self.enc.add_special_tokens({"additional_special_tokens": [special_token]})


			story_tok = self.enc.encode_plus(text=story_content,
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.
			altered_lst = og_comments
			altered_lst[alt_loc] = '[MASK]'
			if len(altered_lst) == 1:
				altered_str = '[CMT]' + altered_lst[0]
			else:
				altered_str = '[CMT]'.join(altered_lst)
			
			#print('len of og_comments ',len(og_comments))
			#print('og_comments str ', og_comments_str)
			#print('altered_str', altered_str)
			altered_comments_tok = self.enc.encode_plus(text=source_string,
												text_pair= altered_str,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.)
			og_comments_input_ids = og_comments_tok['input_ids']
			og_comments_attention_masks = og_comments_tok['attention_mask']
			

			altered_comments_input_ids = altered_comments_tok['input_ids']
			altered_comments_attention_masks = altered_comments_tok['attention_mask']
			


			
			custom_tokens_input_ids = self.enc.convert_tokens_to_ids(custom_tokens)
			mask_token_id = custom_tokens_input_ids[0]
			cmt_token_id = custom_tokens_input_ids[1]

			#need to figure out the corresponding token positions
			sep_og_comments_input_ids = sep_token_help(og_comments_input_ids, cmt_token_id)
			sep_altered_comments_input_ids = sep_token_help(altered_comments_input_ids, cmt_token_id)

			for item in sep_og_comments_input_ids:
				if item not in sep_altered_comments_input_ids:
					target_index = sep_og_comments_input_ids.index(item)
			if not target_index:
				raise ValueError('cannot catch that long [MASK] value')

			sublists = convert_to_sublists(og_comments_input_ids, cmt_token_id)
			#print('value of the sbulists ', sublists)
			#print('target_index ', target_index)
			#print('len of the sublists ',len(sublists))


			target_loc = sublists[target_index-1]
			self.target_locations = [t-1 for t in target_loc]
			self.og_input_ids = torch.tensor(og_comments_input_ids).to(device)
			self.og_attention_mask = torch.tensor(og_comments_attention_masks).to(device)
			self.altered_input_ids = torch.tensor(altered_comments_input_ids).to(device)
			self.altered_attention_mask = torch.tensor(altered_comments_attention_masks).to(device)

			if gold_label == 'rumour':
					self.label_tok = 1
			else:
					self.label_tok = 0
			self.label_tok = torch.tensor(self.label_tok).to(device)



class RumourIntervention():
	def __init__(self, tokennizer, source_string: str, og_comments: list, alt_loc, gold_label, device='cuda:0'):
			super()
			self.device = device
			self.enc = tokennizer

			custom_tokens = ["[MASK]","[CMT]"]
			self.enc.add_tokens(["[MASK]","[CMT]"])

			if len(og_comments) == 1:
				og_comments_str = '[CMT]' + og_comments[0]
			else:
				og_comments_str = '[CMT]'.join(og_comments)
			og_comments_tok = self.enc.encode_plus(text=source_string,
												text_pair= og_comments_str,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.
			altered_lst = og_comments
			altered_lst[alt_loc] = '[MASK]'
			if len(altered_lst) == 1:
				altered_str = '[CMT]' + altered_lst[0]
			else:
				altered_str = '[CMT]'.join(altered_lst)
			
			#print('len of og_comments ',len(og_comments))
			#print('og_comments str ', og_comments_str)
			#print('altered_str', altered_str)
			altered_comments_tok = self.enc.encode_plus(text=source_string,
												text_pair= altered_str,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.)
			og_comments_input_ids = og_comments_tok['input_ids']
			og_comments_attention_masks = og_comments_tok['attention_mask']
			

			altered_comments_input_ids = altered_comments_tok['input_ids']
			altered_comments_attention_masks = altered_comments_tok['attention_mask']
			


			
			custom_tokens_input_ids = self.enc.convert_tokens_to_ids(custom_tokens)
			mask_token_id = custom_tokens_input_ids[0]
			cmt_token_id = custom_tokens_input_ids[1]

			#need to figure out the corresponding token positions
			sep_og_comments_input_ids = sep_token_help(og_comments_input_ids, cmt_token_id)
			sep_altered_comments_input_ids = sep_token_help(altered_comments_input_ids, cmt_token_id)

			##trial and error
			'''
			og_size = len(og_comments_input_ids)
			og_idx_list = [idx + 1 for idx, val in enumerate(og_comments_input_ids) if val == cmt_token_id]
			sep_og_comments_input_ids = [og_comments_input_ids[i:j] for i, j in zip([0] + og_idx_list, og_idx_list + ([og_size] if og_idx_list[-1] != og_size else []))]


			altered_size = len(altered_comments_input_ids)
			altered_idx_list = [idx + 1 for idx, val in enumerate(altered_comments_input_ids) if val == cmt_token_id]
			sep_altered_comments_input_ids = [altered_comments_input_ids[i:j] for i, j in zip([0] + altered_idx_list, altered_idx_list + ([altered_size] if altered_idx_list[-1] != altered_size else []))]
			'''
			#print('sep_og_comments_input_ids len', len(sep_og_comments_input_ids))
			#print('sep_og_comments_input_ids shape ', len(sep_altered_comments_input_ids))
			#find the different index
			for item in sep_og_comments_input_ids:
				if item not in sep_altered_comments_input_ids:
					target_index = sep_og_comments_input_ids.index(item)
			if not target_index:
				raise ValueError('cannot catch that long [MASK] value')

			sublists = convert_to_sublists(og_comments_input_ids, cmt_token_id)
			#print('value of the sbulists ', sublists)
			#print('target_index ', target_index)
			#print('len of the sublists ',len(sublists))


			target_loc = sublists[target_index-1]
			self.target_locations = [t-1 for t in target_loc]
			self.og_input_ids = torch.tensor(og_comments_input_ids).to(device)
			self.og_attention_mask = torch.tensor(og_comments_attention_masks).to(device)
			self.altered_input_ids = torch.tensor(altered_comments_input_ids).to(device)
			self.altered_attention_mask = torch.tensor(altered_comments_attention_masks).to(device)

			if gold_label == 'rumour':
					self.label_tok = 1
			else:
					self.label_tok = 0
			self.label_tok = torch.tensor(self.label_tok).to(device)


class RumourTokenIntervention():
	def __init__(self, tokennizer, source_string: str, og_comments: list, alt_loc,token_alt_loc, device='cuda'):
		super()
		self.device = device
		self.enc = tokennizer

		custom_tokens = ["[MASKTOK]","[CMT]"]
		#[CMT] is used to seperate each comment
		#[MASK] is used to mask whole input sequence, removed [MASK] for now, we do not need [MASK] on token level
		#[MASKTOK] is used to mask out each token in the input sequence
		self.enc.add_tokens(custom_tokens)

		if len(og_comments) == 1:
			og_comments_str = '[CMT]'+og_comments[0]
		else:
			og_comments_str = '[CMT]'.join(og_comments)
		og_comments_tok = self.enc.encode_plus(text=source_string,
												text_pair= og_comments_str,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.
		og_comments_input_ids = og_comments_tok['input_ids']
		og_comments_attention_masks = og_comments_tok['attention_mask']

		target_alt_sentence = og_comments[alt_loc]
		target_alt_tok = target_alt_sentence[token_alt_loc]

		#alt_og_sentence_lst = og_comments
		alt_og_sentence_lst = []
		for og_index, og_content in enumerate(og_comments):
			if og_index == alt_loc:
				tok_lst = og_content.split()
				new_lst = []
				for tok_index, tok_content in enumerate(tok_lst):
					if tok_index == token_alt_loc:
						new_lst.append('[MASKTOK]')
					else:
						new_lst.append(tok_content)
				new_content = ' '.join(new_lst)
				alt_og_sentence_lst.append(new_content)
			else:
				alt_og_sentence_lst.append(og_content)
		#alt_og_sentence_lst[alt_loc][token_alt_loc] = '[MASKTOK]'

		if len(alt_og_sentence_lst) == 1:
			altered_str = '[CMT]'+alt_og_sentence_lst[0]
		else:
			altered_str = '[CMT]'.join(alt_og_sentence_lst)

		altered_comments_tok = self.enc.encode_plus(text=source_string,
												text_pair= altered_str,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.)
		altered_comments_input_ids = altered_comments_tok['input_ids']
		altered_comments_attention_masks = altered_comments_tok['attention_mask']
			


			
		custom_tokens_input_ids = self.enc.convert_tokens_to_ids(custom_tokens)
		mask_token_id = custom_tokens_input_ids[0]
		cmt_token_id = custom_tokens_input_ids[1]

		#need to figure out the corresponding token positions
		sep_og_comments_input_ids = sep_token_help(og_comments_input_ids, cmt_token_id)
		sep_altered_comments_input_ids = sep_token_help(altered_comments_input_ids, cmt_token_id)

		##trial and error
		'''
		og_size = len(og_comments_input_ids)
		og_idx_list = [idx + 1 for idx, val in enumerate(og_comments_input_ids) if val == cmt_token_id]
		sep_og_comments_input_ids = [og_comments_input_ids[i:j] for i, j in zip([0] + og_idx_list, og_idx_list + ([og_size] if og_idx_list[-1] != og_size else []))]


		altered_size = len(altered_comments_input_ids)
		altered_idx_list = [idx + 1 for idx, val in enumerate(altered_comments_input_ids) if val == cmt_token_id]
		sep_altered_comments_input_ids = [altered_comments_input_ids[i:j] for i, j in zip([0] + altered_idx_list, altered_idx_list + ([altered_size] if altered_idx_list[-1] != altered_size else []))]
		'''
		#print('sep_og_comments_input_ids len', len(sep_og_comments_input_ids))
		#print('sep_og_comments_input_ids shape ', len(sep_altered_comments_input_ids))
		#find the different index
		for item in sep_og_comments_input_ids:
			if item not in sep_altered_comments_input_ids:
				target_index = sep_og_comments_input_ids.index(item)
		if not target_index:
			raise ValueError('cannot catch that long [MASK] value')

		sublists = convert_to_sublists(og_comments_input_ids, cmt_token_id)
		#print('value of the sbulists ', sublists)
		#print('target_index ', target_index)
		#print('len of the sublists ',len(sublists))


		target_loc = sublists[target_index-1]
		self.target_locations = [t-1 for t in target_loc]
		self.og_input_ids = torch.tensor(og_comments_input_ids).to(device)
		self.og_attention_mask = torch.tensor(og_comments_attention_masks).to(device)
		self.altered_input_ids = torch.tensor(altered_comments_input_ids).to(device)
		self.altered_attention_mask = torch.tensor(altered_comments_attention_masks).to(device)



class RumourComboIntervention():
	def __init__(self, tokennizer, source_string: str, og_comments: list, alt_loc, device='cuda'):
		super()
		self.device = device
		self.enc = tokennizer

		custom_tokens = ["[MASKTOK]","[CMT]"]
		#[CMT] is used to seperate each comment
		#[MASK] is used to mask whole input sequence, removed [MASK] for now, we do not need [MASK] on token level
		#[MASKTOK] is used to mask out each token in the input sequence
		self.enc.add_tokens(custom_tokens)

		if len(og_comments) == 1:
			og_comments_str = '[CMT]'+og_comments[0]
		else:
			og_comments_str = '[CMT]'.join(og_comments)
		og_comments_tok = self.enc.encode_plus(text=source_string,
												text_pair= og_comments_str,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.
		og_comments_input_ids = og_comments_tok['input_ids']
		og_comments_attention_masks = og_comments_tok['attention_mask']

		target_alt_sentence = og_comments[alt_loc]
		tokens = self.enc.tokenize(target_alt_sentence)
		masked_sequence = " ".join(["[MASKTOK]" for _ in tokens])

		alt_og_sentence_lst = []
		for og_index, og_content in enumerate(og_comments):
			if og_index == alt_loc:
				alt_og_sentence_lst.append(masked_sequence)
			else:
				alt_og_sentence_lst.append(og_content)

		if len(alt_og_sentence_lst) == 1:
			altered_str = '[CMT]'+alt_og_sentence_lst[0]
		else:
			altered_str = '[CMT]'.join(alt_og_sentence_lst)

		altered_comments_tok = self.enc.encode_plus(text=source_string,
												text_pair= altered_str,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.)
		altered_comments_input_ids = altered_comments_tok['input_ids']
		altered_comments_attention_masks = altered_comments_tok['attention_mask']


		custom_tokens_input_ids = self.enc.convert_tokens_to_ids(custom_tokens)
		mask_token_id = custom_tokens_input_ids[0]
		cmt_token_id = custom_tokens_input_ids[1]


		#need to figure out the corresponding token positions #NEED to figure out the internvention position
		sep_og_comments_input_ids = sep_token_help(og_comments_input_ids, cmt_token_id)
		sep_altered_comments_input_ids = sep_token_help(altered_comments_input_ids, cmt_token_id)



		for item in sep_og_comments_input_ids:
			if item not in sep_altered_comments_input_ids:
				target_index = sep_og_comments_input_ids.index(item)
		if not target_index:
			raise ValueError('cannot catch that long [MASK] value')

		sublists = convert_to_sublists(og_comments_input_ids, cmt_token_id)


		target_loc = sublists[target_index-1]

		self.target_locations = [t-1 for t in target_loc]
		print('HERE IS THE CALCULATED TARGET LOCATIONS ', self.target_locations)
		self.og_input_ids = torch.tensor(og_comments_input_ids).to(device)
		self.og_attention_mask = torch.tensor(og_comments_attention_masks).to(device)
		self.altered_input_ids = torch.tensor(altered_comments_input_ids).to(device)
		self.altered_attention_mask = torch.tensor(altered_comments_attention_masks).to(device)





class RumourInterventionBASE():
	def __init__(self, tokennizer, source_string: str, og_comments: list, alt_loc, gold_label, device='cuda:0'):
			super()
			self.device = device
			self.enc = tokennizer

			custom_tokens = ["[MASK]","[CMT]"]
			self.enc.add_tokens(["[MASK]","[CMT]"])

			if len(og_comments) == 1:
				og_comments_str = '[CMT]' + og_comments[0]
			else:
				og_comments_str = '[CMT]'.join(og_comments)
			og_comments_tok = self.enc.encode_plus(text=source_string,
												text_pair= og_comments_str,                     # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = 512,           # Pad & truncate all sentences.
												truncation_strategy = 'only_second',
												padding = 'max_length',
												pad_to_max_length = True)     # Return pytorch tensors.

			og_comments_input_ids = og_comments_tok['input_ids']
			og_comments_attention_masks = og_comments_tok['attention_mask']

			self.og_input_ids = torch.tensor(og_comments_input_ids).to(device)
			self.og_attention_mask = torch.tensor(og_comments_attention_masks).to(device)

			if gold_label == 'rumour':
					self.label_tok = 1
			else:
					self.label_tok = 0
			self.label_tok = torch.tensor(self.label_tok).to(device)




class Intervention():
		'''
		Wrappter for all the possible interventions
		one intervention with one modified comment
		'''
		def __init__(self, tokenizer, source_string: str, og_comments: list, alt_loc, gold_label, device='cpu'):
				super()
				self.device = device

				self.enc = tokenizer

				og_comments_str = '[CMT] '.join(og_comments)

				og_comments_tok = self.enc.tokenize(og_comments_str)

				#cmt_token_index = og_comments_tok.index('[CMT]')
				cmt_token_indexes = self.find_custom_token_indexes(og_comments_tok, '[CMT]')

				if len(og_comments_tok) > 512:
					og_comments_tok = og_comments_tok[:512]

				# Create the attention mask
				og_attention_mask = [1] * len(og_comments_tok)
				for cmt_index in cmt_token_indexes:
					og_attention_mask[cmt_index] = 0
				#og_attention_mask[cmt_token_index] = 0

				# Convert tokenized text to input_ids
				og_input_ids = self.enc.convert_tokens_to_ids(og_comments_tok)


				self.og_input_ids = torch.tensor(og_input_ids).to(device)
				self.og_attention_mask = torch.tensor(og_attention_mask).to(device)
				## Pad the input_ids to max_len
				## padded_inputs = pad_sequence([torch.tensor(input_ids)], 
				##                     batch_first=True, 
				##                     padding_value=0,
				##                     max_len = tokenizer.max_len)

				altered_lst = og_comments
				altered_lst[alt_loc] = '[MASK]'
				altered_str = '[CMT] '.join(altered_lst)
				altered_comments_tok = self.enc.tokenize(altered_str)
				altered_cmt_indexes = self.find_custom_token_indexes(altered_comments_tok, '[CMT]')

				if len(altered_comments_tok) > 512:
					altered_comments_tok = altered_comments_tok[:512]

				#Create attention mask for altered input
				altered_comments_attention_mask = [1] * len(altered_comments_tok)
				for alt_cmt_index in altered_cmt_indexes:
					altered_comments_attention_mask[alt_cmt_index] = 0

				altered_input_ids = self.enc.convert_tokens_to_ids(altered_comments_tok)

				self.altered_input_ids = torch.tensor(altered_input_ids).to(device)
				self.altered_attention_mask = torch.tensor(altered_comments_attention_mask).to(device)

				if gold_label == 'rumour':
					self.label_tok = 1
				else:
					self.label_tok = 0
				self.label_tok = torch.tensor(self.label_tok).to(device)
		
		def find_custom_token_indexes(self, lst, tok):
			indexes = []
			for i in range(lst.count(tok)):
				indexes.append(lst.index(tok, indexes[i-1]+1 if i > 0 else 0))
			return indexes

