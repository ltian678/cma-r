import pandas as pd
from transformers import BertTokenizer,RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig,RobertaForSequenceClassification,XLMRobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
import argparse
import os
import torch


#Set up the read in file path



parser = argparse.ArgumentParser(description="Run a set of fine-tune experiments.")

parser.add_argument(
		"-train_dir",
		type=str,
		default="../train_data/train_causal_v2.pkl"
)



parser.add_argument(
		"-test_dir", default="../train_data/test_causal_v2.pkl", type=str
)


parser.add_argument(
	"-num_class", default=2, type=int
)

parser.add_argument(
	"-a",default='source_tweet',type=str
)


parser.add_argument(
	"-b",default='cmt_comments',type=str
)

parser.add_argument(
	"-labels", default='binary_label', type=str
)

parser.add_argument(
	"-a_test",default='source_tweet',type=str
)


parser.add_argument(
	"-b_test",default='cmt_comments',type=str
)

parser.add_argument(
	"-labels_test", default='binary_label', type=str
)


parser.add_argument(
	"-pretrained", default=False
)

parser.add_argument(
	"-pretrained_model_path", default='res/models/20230226pretrained_model_causal_3/', type=str
)

parser.add_argument(
	"-version_number",default='v1',type=str
)

opt = parser.parse_args()




#nohup python finetune.py -num_class 4 -labels full_label > fine-tune4.log &
#nohup python finetune.py -train_dir '../finetune_data/train_causal_v3.pkl' -test_dir '../finetune_data/test_causal_v3.pkl' -num_class 3 -labels rumour_label > fine-tune3.log &

def format_time(elapsed):
		'''
		Takes a time in seconds and returns a string hh:mm:ss
		'''
		# Round to the nearest second.
		elapsed_rounded = int(round((elapsed)))
		
		# Format as hh:mm:ss
		return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
		pred_flat = np.argmax(preds, axis=1).flatten()
		labels_flat = labels.flatten()
		return np.sum(pred_flat == labels_flat) / len(labels_flat)

def run(train_dir, test_dir, device, epoch, learning_rate, num_class,a,b,labels,a_test,b_test,labels_test,pretrained, pretrained_model_path,version_number):
	train_df = pd.read_pickle(train_dir)
	test_df = pd.read_pickle(test_dir)
	EP = epoch
	num_class = num_class
	a_input = a
	b_input = b
	labels_input = labels
	a_test=a_test
	b_test=b_test
	labels_test=labels_test
	train(train_df, test_df, EP, learning_rate, device, num_class, a_input, b_input, labels_input,a_test,b_test,labels_test,pretrained,pretrained_model_path,version_number)



def train(df,test_df, EP, learning_rate, device,num_class, a_input,b_input,labels_input, a_test,b_test, labels_test, pretrained, pretrained_model_path,version_number):
	df = df.sample(frac=1,replace=False)
	#a = df.source_tweet.values
	#b = df.cmt_comments.values
	#labels = df.final_label.values
	if a_input == 'soure_tweet':
		a = df.source_tweet.values
	elif a_input == 'source_story':
		a = df.source_story.values
	elif a_input == 'clean_source':
		a = df.clean_source.values
	elif a_input == 'source_text':
		a = df.source_text.values
	else:
		a = df.source_tweet.values
	if b_input == 'cmt_comments':
		b = df.cmt_comments.values
	elif b_input == 'ss_sources':
		b = df.ss_sources.values
	elif b_input == 'reply_text_lst':
		b = df.reply_text_lst.values
	elif b_input == 'clean_replies_content':
		b = df.clean_replies_content.values
	else:
		b = df.cmt_comments.values
	#b = df.cmt_comments.values
	if labels_input == 'binary_label':
		lables = df.binary_label.values
	elif labels_input == 'rumour_label':
		labels = df.rumour_final_label.values
	elif labels_input == 'full_label':
		labels = df.final_label.values
	elif labels_input == 'rumour_label_code':
		labels = df.rumour_label_code.values
	elif labels_input == 'matched_rumour_label':
		labels = df.matched_rumour_label.values
	elif labels_input == 'is_rumour':
		labels = df.is_rumour.values
	else:
		labels = df.final_label.values
	#labels = df.binary_label.values

	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

	#ADD customised tokens
	tokenizer.add_tokens(["[MASK]","[CMT]"])


	input_ids = []
	attention_masks = []

	# For every sentence...
	for i,sent in enumerate(a):
			# `encode_plus` will:
			#   (1) Tokenize the sentence.
			#   (2) Prepend the `[CLS]` token to the start.
			#   (3) Append the `[SEP]` token to the end.
			#   (4) Map tokens to their IDs.
			#   (5) Pad or truncate the sentence to `max_length`
			#   (6) Create attention masks for [PAD] tokens.
			encoded_dict = tokenizer.encode_plus(
													text=sent,
													text_pair= b[i],                     # Sentence to encode.
													add_special_tokens = True, # Add '[CLS]' and '[SEP]'
													max_length = 512,           # Pad & truncate all sentences.
													padding = 'max_length',
													pad_to_max_length = True,
													return_attention_mask = True,   # Construct attn. masks.
													return_tensors = 'pt',     # Return pytorch tensors.
													truncation_strategy = 'only_second'
										)
			
			# Add the encoded sentence to the list.    
			input_ids.append(encoded_dict['input_ids'])
			
			# And its attention mask (simply differentiates padding from non-padding).
			attention_masks.append(encoded_dict['attention_mask'])

	# Convert the lists into tensors.
	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)

	dataset = TensorDataset(input_ids, attention_masks, labels)

	# Create a 90-10 train-validation split.

	# Calculate the number of samples to include in each set.
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size

	# Divide the dataset by randomly selecting samples.
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	batch_size = 16

	# Create the DataLoaders for our training and validation sets.
	# We'll take training samples in random order. 
	train_dataloader = DataLoader(
							train_dataset,  # The training samples.
							sampler = RandomSampler(train_dataset), # Select batches randomly
							batch_size = batch_size # Trains with this batch size.
					)

	# For validation the order doesn't matter, so we'll just read them sequentially.
	validation_dataloader = DataLoader(
							val_dataset, # The validation samples.
							sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
							batch_size = batch_size # Evaluate with this batch size.
					)
	if not pretrained:
		model = RobertaForSequenceClassification.from_pretrained(
			"roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
			num_labels = num_class, # The number of output labels--2 for binary classification.
											# You can increase this for multi-class tasks.   
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
		)
	else:
		model = RobertaForSequenceClassification.from_pretrained(
			pretrained_model_path, # load pretrained model
			num_labels = num_class, # The number of output labels--2 for binary classification.
											# You can increase this for multi-class tasks.   
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
		)



	# SET UP CUSTOMISED TOKENS
	model.resize_token_embeddings(len(tokenizer))
	model.roberta.embeddings.word_embeddings.padding_idx=1
	# FREEZE BERT only keep the classifer layer
	#for param in model.bert.bert.parameters():
	#    param.requires_grad = False

	#print(model)
	# Tell pytorch to run this model on the GPU.
	model.cuda()
	optimizer = AdamW(model.parameters(),
									#weight_decay = 0.1,
									lr = learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
									eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
								)
	epochs = EP

	# Total number of training steps is [number of batches] x [number of epochs]. 
	# (Note that this is not the same as the number of training samples).
	total_steps = len(train_dataloader) * epochs
	print('total_steps', total_steps)

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
	seed_val = 42
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	# We'll store a number of quantities such as training and validation loss, 
	# validation accuracy, and timings.
	training_stats = []

	# Measure the total training time for the whole run.
	total_t0 = time.time()

	# For each epoch...
	for epoch_i in range(0, epochs):
			
			# ========================================
			#               Training
			# ========================================
			
			# Perform one full pass over the training set.

			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
			print('Training...')

			# Measure how long the training epoch takes.
			t0 = time.time()

			# Reset the total loss for this epoch.
			total_train_loss = 0

			# Put the model into training mode. Don't be mislead--the call to 
			# `train` just changes the *mode*, it doesn't *perform* the training.
			# `dropout` and `batchnorm` layers behave differently during training
			# vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
			model.train()

			# For each batch of training data...
			for step, batch in enumerate(train_dataloader):

					# Progress update every 40 batches.
					if step % 40 == 0 and not step == 0:
							# Calculate elapsed time in minutes.
							elapsed = format_time(time.time() - t0)
							
							# Report progress.
							print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

					# Unpack this training batch from our dataloader. 
					#
					# As we unpack the batch, we'll also copy each tensor to the GPU using the 
					# `to` method.
					#
					# `batch` contains three pytorch tensors:
					#   [0]: input ids 
					#   [1]: attention masks
					#   [2]: labels 
					b_input_ids = batch[0].to(device)
					b_input_mask = batch[1].to(device)
					b_labels = batch[2].to(device)

					# Always clear any previously calculated gradients before performing a
					# backward pass. PyTorch doesn't do this automatically because 
					# accumulating the gradients is "convenient while training RNNs". 
					# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
					model.zero_grad()        

					# Perform a forward pass (evaluate the model on this training batch).
					# The documentation for this `model` function is here: 
					# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
					# It returns different numbers of parameters depending on what arguments
					# arge given and what flags are set. For our useage here, it returns
					# the loss (because we provided labels) and the "logits"--the model
					# outputs prior to activation.
					loss, logits = model(b_input_ids, 
															token_type_ids=None, 
															attention_mask=b_input_mask, 
															labels=b_labels)

					# Accumulate the training loss over all of the batches so that we can
					# calculate the average loss at the end. `loss` is a Tensor containing a
					# single value; the `.item()` function just returns the Python value 
					# from the tensor.
					total_train_loss += loss.item()

					# Perform a backward pass to calculate the gradients.
					loss.backward()

					# Clip the norm of the gradients to 1.0.
					# This is to help prevent the "exploding gradients" problem.
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

					# Update parameters and take a step using the computed gradient.
					# The optimizer dictates the "update rule"--how the parameters are
					# modified based on their gradients, the learning rate, etc.
					optimizer.step()

					# Update the learning rate.
					scheduler.step()

			# Calculate the average loss over all of the batches.
			avg_train_loss = total_train_loss / len(train_dataloader)            
			
			# Measure how long this epoch took.
			training_time = format_time(time.time() - t0)

			print("")
			print("  Average training loss: {0:.2f}".format(avg_train_loss))
			print("  Training epcoh took: {:}".format(training_time))
					
			# ========================================
			#               Validation
			# ========================================
			# After the completion of each training epoch, measure our performance on
			# our validation set.

			print("")
			print("Running Validation...")

			t0 = time.time()

			# Put the model in evaluation mode--the dropout layers behave differently
			# during evaluation.
			model.eval()

			# Tracking variables 
			total_eval_accuracy = 0
			total_eval_loss = 0
			nb_eval_steps = 0

			# Evaluate data for one epoch
			for batch in validation_dataloader:
					
					# Unpack this training batch from our dataloader. 
					#
					# As we unpack the batch, we'll also copy each tensor to the GPU using 
					# the `to` method.
					#
					# `batch` contains three pytorch tensors:
					#   [0]: input ids 
					#   [1]: attention masks
					#   [2]: labels 
					b_input_ids = batch[0].to(device)
					b_input_mask = batch[1].to(device)
					b_labels = batch[2].to(device)
					
					# Tell pytorch not to bother with constructing the compute graph during
					# the forward pass, since this is only needed for backprop (training).
					with torch.no_grad():        

							# Forward pass, calculate logit predictions.
							# token_type_ids is the same as the "segment ids", which 
							# differentiates sentence 1 and 2 in 2-sentence tasks.
							# The documentation for this `model` function is here: 
							# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
							# Get the "logits" output by the model. The "logits" are the output
							# values prior to applying an activation function like the softmax.
							(loss, logits) = model(b_input_ids, 
																		token_type_ids=None, 
																		attention_mask=b_input_mask,
																		labels=b_labels)
							
					# Accumulate the validation loss.
					total_eval_loss += loss.item()

					# Move logits and labels to CPU
					logits = logits.detach().cpu().numpy()
					label_ids = b_labels.to('cpu').numpy()

					# Calculate the accuracy for this batch of test sentences, and
					# accumulate it over all batches.
					total_eval_accuracy += flat_accuracy(logits, label_ids)
					

			# Report the final accuracy for this validation run.
			avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
			print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

			# Calculate the average loss over all of the batches.
			avg_val_loss = total_eval_loss / len(validation_dataloader)
			
			# Measure how long the validation run took.
			validation_time = format_time(time.time() - t0)
			
			print("  Validation Loss: {0:.2f}".format(avg_val_loss))
			print("  Validation took: {:}".format(validation_time))

			# Record all statistics from this epoch.
			training_stats.append(
					{
							'epoch': epoch_i + 1,
							'Training Loss': avg_train_loss,
							'Valid. Loss': avg_val_loss,
							'Valid. Accur.': avg_val_accuracy,
							'Training Time': training_time,
							'Validation Time': validation_time
					}
			)

	print("")
	print("Training complete!")

	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
	pd.set_option('precision', 2)

	# Create a DataFrame from our training statistics.
	df_stats = pd.DataFrame(data=training_stats)

	# Use the 'epoch' as the row index.
	df_stats = df_stats.set_index('epoch')


	# Create sentence and label lists
	#test_a = test_df.source_tweet.values
	#test_b = test_df.cmt_comments.values
	#test_labels = test_df.binary_label.values

	if a_test == 'soure_tweet':
		test_a = test_df.source_tweet.values
	elif a_test == 'source_story':
		test_a = test_df.source_story.values
	elif a_test == 'clean_source':
		test_a = test_df.clean_source.values
	elif a_test == 'source_text':
		test_a = test_df.source_text.values
	else:
		test_a = test_df.source_tweet.values
	if b_test == 'cmt_comments':
		test_b = test_df.cmt_comments.values
	elif b_test == 'ss_sources':
		test_b = test_df.ss_sources.values
	elif b_test == 'clean_replies_content':
		test_b = test_df.clean_replies_content.values
	elif b_test == 'reply_text_lst':
		test_b = test_df.reply_text_lst.values
	else:
		test_b = test_df.cmt_comments.values
	#b = df.cmt_comments.values
	if labels_test == 'binary_label':
		test_labels = test_df.binary_label.values
	elif labels_test == 'rumour_label':
		test_labels = test_df.rumour_final_label.values
	elif labels_test == 'full_label':
		test_labels = test_df.final_label.values
	elif labels_test == 'rumour_label_code':
		test_labels = test_df.rumour_label_code.values
	elif labels_test == 'matched_rumour_label':
		test_labels = test_df.matched_rumour_label.values
	elif labels_test == 'is_rumour':
		test_labels = test_df.is_rumour.values
	else:
		test_labels = test_df.binary_label.values






	print('number of test labels',len(test_labels))

	# Tokenize all of the sentences and map the tokens to thier word IDs.
	test_input_ids = []
	test_attention_masks = []

	# For every sentence...
	for i,sent in enumerate(test_a):
			# `encode_plus` will:
			#   (1) Tokenize the sentence.
			#   (2) Prepend the `[CLS]` token to the start.
			#   (3) Append the `[SEP]` token to the end.
			#   (4) Map tokens to their IDs.
			#   (5) Pad or truncate the sentence to `max_length`
			#   (6) Create attention masks for [PAD] tokens.
			encoded_dict = tokenizer.encode_plus(
													text=sent,
													text_pair= test_b[i],                     # Sentence to encode.
													add_special_tokens = True, # Add '[CLS]' and '[SEP]'
													max_length = 512,           # Pad & truncate all sentences.
													padding = 'max_length',
													pad_to_max_length = True,
													return_attention_mask = True,   # Construct attn. masks.
													return_tensors = 'pt',     # Return pytorch tensors.
													truncation_strategy = 'only_second'
										)
			
			# Add the encoded sentence to the list.    
			test_input_ids.append(encoded_dict['input_ids'])
			
			# And its attention mask (simply differentiates padding from non-padding).
			test_attention_masks.append(encoded_dict['attention_mask'])

	# Convert the lists into tensors.
	test_input_ids = torch.cat(test_input_ids, dim=0)
	test_attention_masks = torch.cat(test_attention_masks, dim=0)
	test_labels = torch.tensor(test_labels)

	# Set the batch size.  
	batch_size = 16

	# Create the DataLoader.
	prediction_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
	prediction_sampler = SequentialSampler(prediction_data)
	prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
	print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

	# Put model in evaluation mode
	model.eval()

	# Tracking variables 
	predictions , true_labels = [], []

	# Predict 
	for batch in prediction_dataloader:
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		
		# Unpack the inputs from our dataloader
		b_input_ids, b_input_mask, b_labels = batch
		
		# Telling the model not to compute or store gradients, saving memory and 
		# speeding up prediction
		with torch.no_grad():
				# Forward pass, calculate logit predictions
				outputs = model(b_input_ids, token_type_ids=None, 
												attention_mask=b_input_mask)

		logits = outputs[0]

		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()
		
		# Store predictions and true labels
		predictions.append(logits)
		true_labels.append(label_ids)

	print('    DONE.')
	from sklearn.metrics import classification_report

	# Combine the results across all batches. 
	flat_predictions = np.concatenate(predictions, axis=0)

	#Get the exactly softmax score for each record
	flat_pre = flat_predictions

	# For each sample, pick the label (0 or 1) with the higher score.
	flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

	# Combine the correct labels for each batch into a single list.
	flat_true_labels = np.concatenate(true_labels, axis=0)
	print(classification_report(flat_true_labels, flat_predictions, digits=4))

	#model_dir = 'res/'
	#model.save_pretrained(model_dir + 'roberta_causal')

	#tok_dir = 'res/'
	#tokenizer.save_pretrained(tok_dir + 'roberta_tok')


	dt_string = datetime.datetime.now().strftime("%Y%m%d")
	out_dir = 'res/'
	folder_name = dt_string + "pretrained_model" + version_number
	base_path = os.path.join(out_dir, "models", folder_name)
	if not os.path.exists(base_path):
			os.makedirs(base_path)
	model_name = 'causal'+'_'+str(num_class)
	tokenizer_name = 'tok'+'_'+str(num_class)
	model.save_pretrained(base_path+model_name)
	tokenizer.save_pretrained(base_path+tokenizer_name)

		








if __name__ == "__main__":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		epoch = 5
		learning_rate = 2e-5
		print('opt.num_class ', opt.num_class)
		print('train_dir ',opt.train_dir)
		print('test_dir ',opt.test_dir)
		run(opt.train_dir, opt.test_dir, device, epoch, learning_rate,opt.num_class,opt.a,opt.b,opt.labels,opt.a_test, opt.b_test, opt.labels_test, opt.pretrained,opt.pretrained_model_path,opt.version_number)