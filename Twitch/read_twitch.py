# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:10:27 2020

@author: ragha
"""
import itertools
import re
import operator
import csv
import matplotlib.pyplot as plt
import os
import glob
import numpy as np


regex='(\[\d{1}:\d{1,2}:\d{1,2}\] <(.*?)> (.*))'
def convert2sec(timestamp):
    h, m, s = timestamp.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def convert(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60   
    return "%d:%02d:%02d" % (hour, minutes, seconds) 

def create_timestamp(t_start,t_end):
#    start_secs=convert2sec(t_start)
#    end_secs=convert2sec(t_end)
    time_range=[]
    for i in range(t_start,t_end+1):
        time_range.append(convert(i))
    return time_range

def readfile(filename,t_start,t_end,outname):
    log_file_path = r""+filename
#    478675628.log
    timestamp=[]
    usrid=[]
    comment=[]
    inside=0
    outside=0
    with open(log_file_path, "r",encoding="utf8") as file:
        for line in file:
            outside+=1
            for match in re.finditer(regex, line, re.S):
                inside+=1
                match_text = match.group()
                match_text=match_text.rstrip('\n')
                match_text=match_text.split(' ')
                timeval=match_text[0][1:-1]
                timeval=convert2sec(timeval)
                if (timeval>=t_start) and (timeval<=t_end):
                    timestamp.append(match_text[0][1:-1])
                    usrid.append(match_text[1])
                    cmt=str(match_text[2:])
                    cmt=cmt[2:-2]
                    comment.append(cmt)
    fname=outname+'_twitch.csv'
    time_range=create_timestamp(t_start,t_end)
    _time,_count=np.unique(np.asarray(timestamp),return_counts=True)
    count=[]
    for i in range(0,len(time_range)):
        ind=np.where(_time== time_range[i])[0]
        if len(ind)==0:
            count.append(0)
        else:
            count.append(int(_count[ind]))
    np.savetxt(fname, np.vstack((time_range,count)).T, fmt='%s', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding="utf-8")
#    np.savetxt(fname, np.vstack((timestamp,usrid,comment)).T, fmt='%s', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding="utf-8")

def extractdetails(filename):
    game_name = []
    starttime = []
    endtime=[]
    game_id=[]
    csv_file = open(filename,'r')
#    csv_file.readline()
    for a, b ,c, d in csv.reader(csv_file, delimiter=','):
        game_name.append(a)
        starttime.append(b)
        endtime.append(c)
        game_id.append(d)
    return game_name,starttime,endtime,game_id


                
def main(directory,fname):  
    try:   
        os.chdir(directory)
        print("Directory changed")  
    except OSError:
        print("Can't change the Current Working Directory")
    game_name,starttime,endtime,game_id=extractdetails(fname)
    os.chdir(directory+"\\Twitch Chat")
    for i in range(0,len(starttime)):
        t_start=convert2sec(starttime[i])
        t_end=convert2sec(endtime[i])
        filename=game_id[i]+".txt"
        if os.path.exists(filename):
            readfile(filename,t_start,t_end,game_name[i][:-4]) 
        else:
            print("File "+game_id[i]+".txt doesnt exist")

if __name__=="__main__":
    dir_name="D:\\CSCI 599 data\\"
#    dir_name="C:\\Users\ragha\\Documents\\CSCI 599\\Twitch chat python"
    fname="extract_time.csv"
    main(dir_name,fname)
    
    
###############################################################################    
#### create unique words
###############################################################################
#hash_map={}
#for element in result:
#    if element in hash_map:
#        hash_map[element] = hash_map[element] + 1
#    else:
#        hash_map[element] = 1
#
#sorted_dict = dict(sorted(hash_map.items(), key=operator.itemgetter(1),reverse=True)[:10])
#mostfreqword=list(sorted_dict.keys())
#considered_words=[]
#considered_words_timestamp=[]
#considered_words_usrid=[]
#
#for elem_to_find in mostfreqword:
#    for i in range(0,len(comment)):
#        if elem_to_find in comment[i]:
#            if comment[i] in considered_words:
#                continue
#            else:
#                considered_words.append(comment[i])
#                considered_words_timestamp.append(timestamp[i])
#                considered_words_usrid.append(usrid[i])
#                
###############################################################################
                ######word_embedding#######
###############################################################################

#samples=considered_words
#tokenizer=Tokenizer(num_words=len(considered_words))
#tokenizer.fit_on_texts(samples)
#sequences=tokenizer.texts_to_sequences(samples)
#dimension=1000
#maxlength=20
#word_index=tokenizer.word_index
#data=preprocessing.sequence.pad_sequences(sequences,maxlen=maxlength)
#embeddinglayer=Embedding(1000,64,input_length=maxlength)
#x_train=preprocessing.sequence.pad_sequences(sequences,maxlen=maxlength)
#
#plt.bar(mostfreqword,list(sorted_dict.values()))
#    

    
