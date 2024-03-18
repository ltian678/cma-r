import pandas as pd

class Story(object):
	def __init__(self, story, source_id_lst, source_obj_lst, source_text_lst, source_timestamp_lst, story_rumour_label, turnaround_lst,event):
			self.story = story
			self.source_id_lst = source_id_lst
			self.source_obj_lst = source_obj_lst
			self.source_text_lst = source_text_lst
			self.source_timestamp_lst = source_timestamp_lst
			self.story_rumour_label = story_rumour_label
			self.turnaround_lst = turnaround_lst
			self.event = event


	def to_dict(self):
				return {
						'story': self.story,
						'source_id_lst': self.source_id_lst,
						'source_obj_lst': self.source_obj_lst,
						'source_text_lst': self.source_text_lst,
						'source_timestamp_lst': self.source_timestamp_lst,
						'turnaround_lst': self.turnaround_lst,
						'story_rumour_label': self.story_rumour_label,
						'event': self.event
				}

from datetime import datetime


def convert_date(row):
	dtime = row['source_time']
	new_datetime = datetime.strptime(dtime,'%a %b %d %H:%M:%S +0000 %Y')
	return new_datetime


def run():
	df = pd.read_pickle('../finetune_data/rumour_whole_story.pkl')
	story_lst = list(set(df.category_label.tolist()))
	obj_lst = []
	for story in story_lst:
		df_lite = df[df['category_label'] == story]
		df_lite['date_source_time'] = df_lite.apply(lambda row: convert_date(row),axis=1)
		df_lite.sort_values(by='date_source_time', ascending = True, inplace = True)
		source_id_lst = df_lite.source_id.tolist()
		source_obj_lst = df_lite.source_obj.tolist()
		source_text_lst = df_lite.source_text.tolist()
		source_timestamp_lst = df_lite.source_time.tolist()
		turnaround_lst = df_lite.is_turnaround.tolist()
		story_rumour_label = df_lite.encoded_rumor_label.tolist()
		event = list(set(df_lite.event.tolist()))
		s = Story(story=story, 
							source_id_lst=source_id_lst, 
							source_obj_lst=source_obj_lst, 
							source_text_lst=source_text_lst, 
							source_timestamp_lst=source_timestamp_lst,
							story_rumour_label=story_rumour_label,
							turnaround_lst=turnaround_lst,
							event=event)
		obj_lst.append(s)
	res_df = pd.DataFrame.from_records([s.to_dict() for s in obj_lst])
	res_df.to_pickle('../finetune_data/story.pkl')




if __name__ == "__main__":
		run()