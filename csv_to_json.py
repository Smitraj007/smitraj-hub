import pandas as pd
import numpy as np
import csv
import json


# *************************Instruments CSV*************************** #

instrument_csv_file = 'D:/BigDataProject/Csv_to_json/instruments/instruments.csv'
instrument_json_path = 'D:/BigDataProject/Csv_to_json/instruments/'

#Creating a dictionary
inst_data={}

#Splitting file to every 1000 rows JSON - Instrument_

with open(instrument_csv_file) as instrument_csv:
    inst_csvRead = csv.DictReader(instrument_csv)
    for inst_csvrows in inst_csvRead:
        inst_token=inst_csvrows['instrument_token']
        inst_data[inst_token]=inst_csvrows

#inst_data

######## Split the dictonary and write in JSON ########

j=0
while j<len(inst_data):
    inst1 = dict(list(inst_data.items())[j:j+1000])
    with open(instrument_json_path+"inst"+str(j)+".json", 'w') as instrument_json:
        instrument_json.write(json.dumps(inst1,indent=4))
    j=j+1000

# *************************Login CSV*************************** #

logDetails_csv='D:/BigDataProject/Csv_to_json/log_inf/log_inf.csv'
log_json_path='D:/BigDataProject/Csv_to_json/log_inf/'

log_data={}

with open(logDetails_csv) as log_csv:
    log_csvRead = csv.DictReader(log_csv)
    for log_csvrows in log_csvRead:
        timestamp=log_csvrows['timestamp']
        log_data[timestamp]=log_csvrows

len(log_data)

i=0
while i<len(log_data):
    log1 = dict(list(log_data.items())[i:i+1000])
    with open(log_json_path+"log"+str(i)+".json", 'w') as log_json:
        log_json.write(json.dumps(log1,indent=4))
    i=i+1000

