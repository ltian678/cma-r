from transformers import BertTokenizer
import re
import torch
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 20

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
	"""Perform required preprocessing steps for pretrained BERT.
	@param    data (np.array): Array of texts to be processed.
	@return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
	@return   attention_masks (torch.Tensor): Tensor of indices specifying which
				  tokens should be attended to by the model.
	"""
	# Create empty lists to store outputs
	input_ids = []
	attention_masks = []

	# For every sentence...
	for sent in data:
		# `encode_plus` will:
		#    (1) Tokenize the sentence
		#    (2) Add the `[CLS]` and `[SEP]` token to the start and end
		#    (3) Truncate/Pad sentence to max length
		#    (4) Map tokens to their IDs
		#    (5) Create attention mask
		#    (6) Return a dictionary of outputs
		encoded_sent = tokenizer.encode_plus(
			text=text_preprocessing(sent),  # Preprocess sentence
			text_pair= b[i],
			add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
			max_length=MAX_LEN,                  # Max length to truncate/pad
			padding=True,
			truncation_strategy='longest_first',
			truncation=True,
			#pad_to_max_length=True,         # Pad sentence to max length
			#return_tensors='pt',           # Return PyTorch tensor
			return_attention_mask=True      # Return attention mask
			)
		
		# Add the outputs to the lists
		input_ids.append(encoded_sent.get('input_ids'))
		attention_masks.append(encoded_sent.get('attention_mask'))

	# Convert lists to tensors
	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)

	return input_ids, attention_masks

def text_preprocessing(text):
	"""
	- Remove entity mentions (eg. '@united')
	- Correct errors (eg. '&amp;' to '&')
	@param    text (str): a string to be processed.
	@return   text (Str): the processed string.
	"""
	# Remove '@name'
	text = re.sub(r'(@.*?)[\s]', ' ', text)

	# Replace '&amp;' with '&'
	text = re.sub(r'&amp;', '&', text)

	# Remove trailing whitespace
	text = re.sub(r'\s+', ' ', text).strip()

	return text



def preprocessing_for_bert_single(tweet_id):
	# Create empty lists to store outputs
	input_ids = []
	attention_masks = []
	target_df = df.loc[df['tweet_id'] == tweet_id]
	sent = target_df['source'].iloc[0]
	comments = target_df['replies'].iloc[0]
	comments_str = ''
	for c in comments:
	  comments_str = comments_str + c
	encoded_sent = tokenizer.encode_plus(
			  text=text_preprocessing(sent),  # Preprocess sentence
			  text_pair= text_preprocessing(comments_str),        # All the comments as one string
			  add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
			  max_length=MAX_LEN,                  # Max length to truncate/pad
			  padding=True,         # Pad sentence to max length
			  truncation_strategy='longest_first',
			  truncation=True,
			  #return_tensors='pt',           # Return PyTorch tensor
			  return_attention_mask=True      # Return attention mask
			  )
	# Add the outputs to the lists
	input_ids.append(encoded_sent.get('input_ids'))
	attention_masks.append(encoded_sent.get('attention_mask'))
	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)
	return input_ids, attention_masks



def preprocessing_for_bert_latest(root_node,node_content):
	# Create empty lists to store outputs
	input_ids = []
	attention_masks = []

	node_lst = []
	for node in node_content:
	  node_lst.append(node)

	whole_lst = []
	whole_lst = root_node.tolist() + node_lst

	for c in whole_lst:

	  encoded_sent = tokenizer.encode_plus(
				text=root_node[0],  # Preprocess sentence
				text_pair= c,
				add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
				max_length=MAX_LEN,                  # Max length to truncate/pad
				padding=True,         # Pad sentence to max length
				truncation_strategy='longest_first',
				truncation=True,
				#return_tensors='pt',           # Return PyTorch tensor
				return_attention_mask=True      # Return attention mask
				)
	  # Add the outputs to the lists
	  input_ids.append(encoded_sent.get('input_ids'))
	  attention_masks.append(encoded_sent.get('attention_mask'))
	#final_input_ids = np.array([input_ids])
	#final_attention_masks = np.array([attention_masks])
	# Convert lists to tensors
	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)

	return input_ids, attention_masks