import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
def read_file(file_name):
	"""read file data when specified file name
		argument:
			file_name: str file_name
		return:  	
			data: list data list
	"""
	full_path_name='./OpportunityChallenge/OpportunityChallengeLabeled/'+file_name
	data=[] #define a list to store data
	with open(full_path_name,'rb') as f:
		for line in f.readlines():
			line=line.replace("  "," ").strip().split(" ")
			data.append(line)
			assert len(line)==116 #confirm feature num is equal

	# for index,value in enumerate(data):
	# 	for in_index, in_value in enumerate(value):
	# 		if in_value=='NaN':
	# 			data[index][in_index]='NaN'
	return data

def haddle_missing_data(data):
	"""remove samples which contains too many NaN, and 
	fill in missing values using linear interpolation
		argument: 
			data: list raw data read from file
		return: 
			clear_data: list clear data
	"""

	clear_data=np.array(data,dtype=np.float32) #cast to matrix
	df=pd.DataFrame(clear_data)
	df=df.interpolate().fillna(method='bfill')
	clear_data=df.values.tolist()
	return clear_data

def data_normalization(clear_data):
	"""use std and mean to normalization clear_data.
	   Attention: the fist column is time ,should be deleted. And the last two columns are 
	   locomation and gesture label should not be normalized.
	    argument:
			clear_data: list missed data have been filled
		return:
			clear_data: np.array data have been normalized
	"""
	#delete the time feature in data
	clear_data=np.delete(clear_data, 0,1)
	
	clear_data=np.array(clear_data,dtype=np.float32)
	max_feature=np.max(clear_data,axis=0)
	min_feature=np.min(clear_data,axis=0)
	std_feature=np.std(clear_data,axis=0)
	mean_feature=np.mean(clear_data,axis=0)
	#No need to normalize labels
	for index in [-1,-2]:
		std_feature[index]=1
		mean_feature[index]=0

	#use std and mean value for normalization	
	clear_data=np.nan_to_num((clear_data-mean_feature)/std_feature)
	return clear_data

def data4model(clear_data):
	"""change data shape to fit LSTM  model. the input data have 115
	   columns,whose last two columns are locomation label and gesture label.
	   We need to shape feature to [samples,time_steps,feature], and create 
	   label, whose shape is [samples,1]
	    argument: 
			clear_data: np.array shape=[sequence_length,features]
		return:	
		    X: np.array shape=[samples,time_steps,feuatures]
		    Y: np.array shape=[sampels,1]
	"""
	#use slide window to reshape input data for LSTM model
	#define time_steps.It means each sample contains 16 sequences.
	#Overlape is 50%,which means each sample contains 8 old sequences 
	#and 8 new sequences.
	T=16 


def main():
	""" preprocess file data: 
	1. fill in missing values using linear interpolation
	2. to do a per feature normalization 
	3. make apropriate sequence
	"""
	data=read_file('S2-ADL1.dat')
	clear_data=haddle_missing_data(data)
	clear_data=data_normalization(clear_data)
	# print np.max(clear_data,axis=0)clear_data
	with open("tempdata.dat",'wb') as f:
		f.writelines(" ".join(str(j) for j in i)+'\n' for i in clear_data)
	
if __name__=="__main__":
	main()