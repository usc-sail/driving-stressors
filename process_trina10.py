import datetime as dt                 # Used to store data and time flags
import pandas as pd                   # Used to extract CSV data
import numpy as np                    # Minor math uses in script
import matplotlib.pyplot as plt       # To plot data

def extract_data(p_name, sample_dt):
    # This function extracts transient sensor data
    # and stores it in the signals dictionary
    df = pd.read_csv(str(p_name)+'_data.txt',sep='\t', header=19)  # Read txt data
    time = np.linspace(0,len(df.values[:,0])*sample_dt,len(df.values[:,0])) # Seconds
    signals = {}                    # Data storage
    signals['ecg'] = df.values[:,1] # All ECG data
    signals['eda'] = df.values[:,2] # All EDA data
    signals['skt'] = df.values[:,3] # All EKT data
    signals['rsp'] = df.values[:,4] # All RSP data
    signals['ppg'] = df.values[:,5] # All PPG data
    return time, signals            

def extract_scene_data(p_name):
    # This function extracts time stamps from laps
    # and stores in a dictionary
    df = pd.read_excel(str(p_name)+'_data.xlsx', sheet_name='Scenes')    # Read CSV Data
    laps = {}                                     # Data storage
    for lap in df.values:                         # Loop through laps
        info = {}                                   # Data storage for given lap
        info['start'] = dt.datetime(int(lap[1]),int(lap[2]),int(lap[3]),int(lap[4]),int(lap[5]),int(lap[6])) # Start time
        info['end'] = dt.datetime(int(lap[1]),int(lap[2]),int(lap[3]),int(lap[7]),int(lap[8]),int(lap[9]))   # End time
        info['events'] = []                         # Creates empty storage for later
        laps[str(lap[0])] = info                # Append data
    return laps

def extract_event_data(p_name, laps):
    # This function extracts and stores time stamps
    # for events. Somewhat manual now, can create
    df = pd.read_excel(str(p_name)+'_data.xlsx', sheet_name='Events')    # Read CSV Data
    scenes = np.unique(df.values[:,0])
    for scene in scenes:                      # Loop through event flags
        if scene == 'Start Time':
            event = df.values[0,:]
            start_time = dt.datetime(event[2],event[3],event[4],event[5],event[6],event[7]) # Start time
        else:
            events = []                    # Data storage
            for event in df.values:        # Loop events
                if event[0] == scene:      # If event is for scene, save
                    events.append([event[1], dt.datetime(event[2],event[3],event[4],event[5],event[6],event[7])]) # Event time
            laps[scene]['events'] = events # Append all event to scene

    return laps, start_time

def pack_data(p_name,sample_dt):
    # This pulls data from excel and stores by participant. 
    laps = extract_scene_data(p_name)                     # Extract lap timeframe
    laps, start_time = extract_event_data(p_name,  laps)  # Extract event flags
    time, signals = extract_data(p_name, sample_dt)       # Extract raw data
    participant = {}                                      # Create storage
    participant['name'] = p_name                          # Participant name
    participant['start_time'] = start_time                # When data collection starts
    participant['time'] = time                            # Save time 
    participant['signals'] = signals                      # Save signals
    participant['laps'] = laps                            # Lap time stamps
    return participant

def convert_time_to_index(current_time, start_time, sample_dt):
    # This simple script takes a date and time
    # and finds index in transient data
    time_change = current_time - start_time       # difference between event and start time
    time_change = time_change.total_seconds()     # convert to seconds
    time_change = np.floor(time_change/sample_dt) # convert to index of data
    return int(time_change)                       # Return time index

def load_participants(p_names, sample_dt):
    participants = []          # Data storage for participants
    for p_name in p_names:     # Loop through participant names
        participants.append(pack_data(p_name, sample_dt)) # Unpack and store data
    return participants

def plot_data(participants, sample_dt, cc, p_names):
    for lap in participants[0]['laps'].keys(): # Generate one plot per event
        fig, axs  = plt.subplots(5,1) # Create with all signals
        fig.suptitle(str(lap)+' Lap')      # Title for lap

        participants = participants if isinstance(participants, list) else [participants]

        for participant in participants:
            start_idx = convert_time_to_index(participant['laps'][str(lap)]['start'], 
                                            participant['start_time'],
                                            sample_dt)
            end_idx = convert_time_to_index(participant['laps'][str(lap)]['end'],
                                            participant['start_time'],
                                            sample_dt)
            
            for key, ax in zip(participant['signals'].keys(), axs):
                data = participant['signals'][key][start_idx:end_idx]
                ax.plot(data, color=cc[participant['name']])
                #print('Start:'+str(start_idx))
                #print('End:'+str(end_idx))
                for event in participant['laps'][lap]['events']:  # If there are events in lap
                    idx = convert_time_to_index(event[1],
                                                participant['start_time'],
                                                    sample_dt)
                    ax.plot([idx-start_idx,idx-start_idx], [min(data), max(data)], 'r')
                
                ax.set_yticks([])
                ax.set_ylabel(str(key))
                ax.set_xticks([])
                ax.legend(p_names, ncol=len(p_names))
            
        #ax.set_xlabel('Time')
        #plt.savefig('figures/'+participant['name']+'_'+lap+'.png', dpi=150)
        plt.show()

def main():
    sample_dt = 0.004  # Data is recorded every 4ms

    p_names = ['PE141166',
               'NY101177',
               'OL141172',
               'QL151179',
               'SZ101195',
               'DI101228',
               'WR111245',
               #'HV111257',
               'HS141262',
               'QD092098',
               'WZ102091'
               ]

    p_names = ['WZ102091']

    #p_names = ['P1'] # Specify which participant(s) to process
    cc = {'PE141166': 'tab:blue', 
          'NY101177': 'tab:orange', 
          'OL141172': 'tab:green', 
          'QL151179': 'tab:purple', 
          'SZ101195': 'tab:olive', 
          'DI101228': 'tab:pink', 
          'WR111245': 'tab:red',
          'HS141262': 'tab:cyan',
          'QD092098': 'tab:brown',
          'WZ102091': 'tab:gray'
          } # Colors for plot!

    participants = load_participants(p_names, sample_dt)
    plot_data(participants, sample_dt, cc, p_names)  


main()