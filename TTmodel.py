import torch
from transformers import BertTokenizer, BertModel, BertConfig, BertPreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F




class TwoTierTransformer(torch.nn.Module):
	def __init__(self, pretrained_model_name, config_second_tier):
		super().__init__()
		
		# Load the pre-trained model from Hugging Face
		#self.bert = AutoModel.from_pretrained(pretrained_model_name)
		self.bert = BertModel.from_pretrained("bert-base-uncased")
		
		# Use the config from the loaded pre-trained model
		#config_first_tier = AutoConfig.from_pretrained(pretrained_model_name)
		config_first_tier = BertConfig.from_pretrained('bert-base-uncased')
		config_second_tier = BertConfig.from_pretrained('bert-base-uncased', num_hidden_layers=1)
		
		self.encoder = BertModel(config_second_tier)
		self.dropout = torch.nn.Dropout(config_second_tier.hidden_dropout_prob)
		self.classifier = torch.nn.Linear(config_second_tier.hidden_size, 3)
		self.max_length = 512
		self.init_weights()

	def init_weights(self):
		""" Initialize the weights for the second tier """
		self.encoder.init_weights()
		self.classifier.apply(self._init_weights)

	def _init_weights(self, module):
		""" Initialize the weights of a given module """
		if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, torch.nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, torch.nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def forward(self, story_input_ids, story_attention_mask, lst_input_ids, lst_attention_masks, labels=None):
		print('labels ',labels)
		print('story_input_ids shape', story_input_ids.shape)
		print('story_attention_mask shape ', story_attention_mask.shape)
		story_outputs = self.bert(input_ids=story_input_ids, attention_mask=story_attention_mask, return_dict=True)
		story_cls_token = story_outputs.last_hidden_state[:, 0, :].unsqueeze(1)  # Make it a 3D tensor

		cls_tokens = []
		max_sentence_length = max([input_id.shape[1] for input_id in lst_input_ids])

		for sentence, attention_mask in zip(lst_input_ids, lst_attention_masks):
			outputs = self.bert(input_ids=sentence, attention_mask=attention_mask, return_dict=True)
			cls_token = outputs.last_hidden_state[:, 0, :].unsqueeze(1)  # Make it a 3D tensor
			
			# Pad the cls_token in each iteration
			padded_cls_token = F.pad(cls_token, pad=(0, 0, 0, max_sentence_length - cls_token.shape[1]))
			cls_tokens.append(padded_cls_token)

		#max_length = max([token.shape[1] for token in cls_tokens])
		max_length = 1
		padded_tokens = torch.zeros((len(cls_tokens), max_sentence_length, self.bert.config.hidden_size), device=story_input_ids.device)
		#padded_tokens = torch.zeros((len(cls_tokens), max_sentence_length, 256), device=story_input_ids.device)

		#print('len cls_tokens', len(cls_tokens))
		#for i, token in enumerate(cls_tokens):
		#	num_rows = token.size(0)
		#	indices = torch.arange(i, i + num_rows, device=story_input_ids.device)
		#	padded_tokens.index_copy_(0, indices, token)



		cls_tokens = padded_tokens
		print('shape of cls_tokens ', cls_tokens.shape)
		combined_tokens = torch.cat([story_cls_token, cls_tokens], dim=1)
		batch_size = combined_tokens.shape[0]
		new_cls_token = torch.zeros((batch_size, 1, combined_tokens.shape[-1])).to(combined_tokens.device)
		print('new_cls_token ', new_cls_token.shape)
		input_to_second_encoder = torch.cat([new_cls_token, combined_tokens], dim=1)

		# Temporarily disable dropout for the second encoder
		original_dropout_p = self.encoder.embeddings.dropout.p
		self.encoder.embeddings.dropout.p = 0

		second_encoder_output = self.encoder(inputs_embeds=input_to_second_encoder)

		# Restore the original dropout probability
		self.encoder.embeddings.dropout.p = original_dropout_p
		
		updated_cls_token = second_encoder_output.last_hidden_state[:, 0, :]
		logits = self.classifier(self.dropout(updated_cls_token))
		probabilities = F.softmax(logits, dim=-1)
		print('probabilities ', probabilities)

		if labels is not None:
			loss_fct = torch.nn.CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, 3), labels.view(-1))
			print('loss ',loss)
			return loss, probabilities
		else:
			return probabilities




from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup


class CustomDataset(Dataset):
	def __init__(self, input_data, labels, pretrained_model_card, max_length=2):
		self.input_data = input_data
		self.labels = labels
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.max_length = max_length

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		input_item = self.input_data[idx]
		label = self.labels[idx]

		story_input_ids = self.tokenizer(input_item["story"], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)['input_ids']
		story_attention_mask = self.tokenizer(input_item["story"], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)['attention_mask']

		all_input_ids = []
		all_attention_masks = []

		for sentences in input_item["lst"]:
			sentence_input_ids = []
			sentence_attention_masks = []
			for sentence in sentences:
				input_ids = self.tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)['input_ids']
				attention_mask = self.tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)['attention_mask']
				sentence_input_ids.append(input_ids)
				sentence_attention_masks.append(attention_mask)
			all_input_ids.append(torch.cat(sentence_input_ids, dim=0))
			all_attention_masks.append(torch.cat(sentence_attention_masks, dim=0))

		return {"input_data": {"story_input_ids": story_input_ids.squeeze(),
								"story_attention_mask": story_attention_mask,
								"lst": all_input_ids,
								"lst_attention_masks": all_attention_masks},
				"label": label}

class CustomDataset_Sim(Dataset):
	def __init__(self, input_data, labels, pretrained_model_card, max_length=20):
		self.input_data = input_data
		self.labels = labels
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.max_length = max_length

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		input_item = self.input_data[idx]
		label = self.labels[idx]

		story_input_ids = self.tokenizer(input_item["story"], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)['input_ids']
		story_attention_mask = self.tokenizer(input_item["story"], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)['attention_mask']

		all_input_ids = []
		all_attention_masks = []

		for sentence in input_item["lst"]:
			input_ids = self.tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)['input_ids']
			attention_mask = self.tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)['attention_mask']
			all_input_ids.append(input_ids)
			all_attention_masks.append(attention_mask)

		all_input_ids = torch.cat(all_input_ids, dim=0)
		all_attention_masks = torch.cat(all_attention_masks, dim=0)

		return {"input_data": {"story_input_ids": story_input_ids.squeeze(),
							   "story_attention_mask": story_attention_mask, 
							   "lst": all_input_ids, 
							   "lst_attention_masks": all_attention_masks}, 
				"label": label}







