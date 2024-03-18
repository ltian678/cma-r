from TTmodel import TwoTierTransformer,CustomDataset,CustomDataset_Sim

import pandas as pd
from transformers import BertTokenizer,RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification,AutoTokenizer
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
 



#Set up the read in file path



parser = argparse.ArgumentParser(description="Run a set of fine-tune TwoTierTransformer experiments.")

parser.add_argument(
		"-train_dir",
		type=str,
		default="../story.pkl"
)



parser.add_argument(
		"-test_dir", default="../finetune_data/test.pkl", type=str
)


parser.add_argument(
	"-num_class", default=3, type=int
)



parser.add_argument(
	"-pretrained", default=False
)

parser.add_argument(
	"-pretrained_model_path", default='res/models/pretrained_causal_3/', type=str
)

parser.add_argument(
	"-version_number",default='v1',type=str
)

opt = parser.parse_args()




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


def prepare_input_data(dataframe):
	input_data = []
	labels = []
	for index, row in dataframe.iterrows():
		input_data_obj = {}
		input_data_obj['story'] = row['source_text']
		input_data_obj['lst'] = row['newnew_twotier_input']
		input_data.append(input_data_obj)
		labels.append(row['is_rumour'])
	labels = torch.tensor(labels)
	return input_data, labels


def collate_fn(batch):
	batch_story_input_ids = [item["input_data"]["story_input_ids"] for item in batch]
	batch_story_attention_masks = [item["input_data"]["story_attention_mask"] for item in batch]
	batch_lst_input_ids = [item["input_data"]["lst"] for item in batch]
	batch_lst_attention_masks = [item["input_data"]["lst_attention_masks"] for item in batch]
	batch_labels = [item["label"] for item in batch]

	batch_story_input_ids = torch.stack(batch_story_input_ids, dim=0)
	batch_story_attention_masks = torch.stack(batch_story_attention_masks, dim=0)
	batch_labels = torch.tensor(batch_labels)

	return {"input_data": {"story_input_ids": batch_story_input_ids,
							"story_attention_mask": batch_story_attention_masks,
							"lst": batch_lst_input_ids,
							"lst_attention_masks": batch_lst_attention_masks},
			"label": batch_labels}




def train(train_dir, test_dir, num_class, pretrained, pretrained_model_path, version_number):
	epoch = 5
	learning_rate = 2e-5
	train_df = pd.read_pickle(train_dir)
	test_df = pd.read_pickle(test_dir)
	train_df = train_df.sample(frac=1, replace=False)
	pretrained_model_name = "finiteautomata/bertweet-base-sentiment-analysis"
	print('Im inside the train function')


	#split the train_df to train and validation
	from sklearn.model_selection import train_test_split
	df_train, df_validation = train_test_split(train_df, test_size=0.1, random_state=42)

	train_input_data, train_labels = prepare_input_data(df_train)
	validation_input_data, validation_labels = prepare_input_data(df_validation)
	test_input_data, test_labels = prepare_input_data(test_df)

	train_dataset = CustomDataset_Sim(train_input_data, train_labels, pretrained_model_card=pretrained_model_name)
	validation_dataset = CustomDataset_Sim(validation_input_data, validation_labels, pretrained_model_card=pretrained_model_name)
	test_dataset = CustomDataset_Sim(test_input_data, test_labels, pretrained_model_card=pretrained_model_name)

	train_batch_size = 2
	test_batch_size = 2


	train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn)
	validation_data_loader = DataLoader(validation_dataset, batch_size= train_batch_size, collate_fn=collate_fn)
	test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size,  collate_fn=collate_fn)

	
	config_second_tier = AutoConfig.from_pretrained(pretrained_model_name)

	model = TwoTierTransformer(pretrained_model_name, config_second_tier)
	model = model.to(device)



	optimizer = AdamW(model.parameters(),
									#weight_decay = 0.1,
									lr = learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
									eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
								)
	epochs = epoch

	# Total number of training steps is [number of batches] x [number of epochs]. 
	# (Note that this is not the same as the number of training samples).
	total_steps = len(train_data_loader) * epochs


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

	num_gpus = 2

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
			#for test in train_data_loader:
			#	print(test.shape)

			# For each batch of training data...
			for step, batch in enumerate(train_data_loader):

					# Progress update every 40 batches.
					if step % 40 == 0 and not step == 0:
							# Calculate elapsed time in minutes.
							elapsed = format_time(time.time() - t0)
							
							# Report progress.
							print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data_loader), elapsed))

					# Unpack this training batch from our dataloader. 
					#
					# As we unpack the batch, we'll also copy each tensor to the GPU using the 
					# `to` method.
					#
					# `batch` contains three pytorch tensors:
					#   [0]: input ids 
					#   [1]: attention masks
					#   [2]: labels 
					#b_input_ids = batch[0].to(device)
					#b_input_mask = batch[1].to(device)
					#b_labels = batch[2].to(device)
					#print('batch.keys() ', batch.keys())

					#b_input_data = batch['input_data']
					#b_labels = batch['label']

					batch["input_data"]["story_input_ids"] = batch["input_data"]["story_input_ids"].to(device)
					batch["input_data"]["story_attention_mask"] = batch["input_data"]["story_attention_mask"].to(device)

					batch["input_data"]["lst"] = [tensor.to(device) for tensor in batch["input_data"]["lst"]]
					batch["input_data"]["lst_attention_masks"] = [tensor.to(device) for tensor in batch["input_data"]["lst_attention_masks"]]

					batch['label'] = batch['label'].to(device)

					#with torch.cuda.amp.autocast(enabled=False):
					model.zero_grad()

					loss, logits = model(story_input_ids=batch['input_data']['story_input_ids'], 
										story_attention_mask=batch['input_data']['story_attention_mask'], 
										lst_input_ids=batch['input_data']['lst'], 
										lst_attention_masks=batch['input_data']['lst_attention_masks'], 
										labels=batch['label'])


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

					# Compute the average loss across all GPUs
					#loss_tensor = torch.tensor([loss.item()], dtype=torch.float32, device=device)
					#dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

					#total_train_loss += loss_tensor.item() / num_gpus

			# Calculate the average loss over all of the batches.
			avg_train_loss = total_train_loss / len(train_dataloader)
			# Compute the total training loss across all processes.
			#total_train_loss = torch.tensor([total_train_loss]).to(device)
			#dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)

			# Compute the average training loss across all processes.
			#avg_train_loss = total_train_loss.item() / (len(train_data_loader) * num_gpus)            
			
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
			for batch in validation_data_loader:
					
					# Unpack this training batch from our dataloader. 
					#
					# As we unpack the batch, we'll also copy each tensor to the GPU using 
					# the `to` method.
					#
					# `batch` contains three pytorch tensors:
					#   [0]: input ids 
					#   [1]: attention masks
					#   [2]: labels 
					b_input_data = batch['input_data'].to(device)
					b_labels = batch['label'].to(device)
					
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
							#(loss, logits) = model(b_input_ids, 
							#											token_type_ids=None, 
							#											attention_mask=b_input_mask,
							#											labels=b_labels)
							(loss, logits) = model(input_data=b_input_data, labels=b_labels)
							
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

	print('Predicting labels for {:,} test sentences...'.format(len(test_data_loader)))

	# Put model in evaluation mode
	model.eval()

	# Tracking variables 
	predictions , true_labels = [], []

	# Predict 
	for batch in test_data_loader:
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		
		# Unpack the inputs from our dataloader
		#b_input_data, b_labels = batch
		b_input_data = batch['input_data']
		b_labels = batch['label']
		
		# Telling the model not to compute or store gradients, saving memory and 
		# speeding up prediction
		with torch.no_grad():
				# Forward pass, calculate logit predictions
				#outputs = model(i, token_type_ids=None, 
				#								attention_mask=b_input_mask)
				outputs = model(input_data=b_input_data)

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
	folder_name = dt_string + "two_tier_pretrained_model" + version_number
	base_path = os.path.join(out_dir, "models", folder_name)
	if not os.path.exists(base_path):
			os.makedirs(base_path)
	model_name = 'roberta_two_tier_causal'+'_'+str(num_class)
	tokenizer_name = 'roberta_two_tier_tok'+'_'+str(num_class)
	model.save_pretrained(base_path+model_name)
	tokenizer.save_pretrained(base_path+tokenizer_name)






def run():
	#os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

	# Call the train function with the necessary arguments
	train(opt.train_dir, opt.test_dir, opt.num_class, opt.pretrained, opt.pretrained_model_path, opt.version_number)

	# Clean up
	#dist.destroy_process_group()




if __name__ == "__main__":
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		num_gpus = torch.cuda.device_count()
		train(opt.train_dir, opt.test_dir, opt.num_class, opt.pretrained, opt.pretrained_model_path, opt.version_number)
		#num_gpus = torch.cuda.device_count()

		# Set the environment variables
		#os.environ['MASTER_ADDR'] = 'localhost'
		#os.environ['MASTER_PORT'] = '29500'
		#mp.spawn(run, nprocs=num_gpus, args=(num_gpus,),join=True)

		#torch.cuda.set_device(device)
		#dist.init_process_group(backend='nccl')

		#epoch = 5
		#learning_rate = 2e-5
		#print('opt.num_class ', opt.num_class)
		#print('train_dir ',opt.train_dir)
		#print('test_dir ',opt.test_dir)
		#train(opt.train_dir, opt.test_dir, epoch, learning_rate, opt.num_class, opt.pretrained, opt.pretrained_model_path, opt.version_number)