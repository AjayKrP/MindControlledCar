import random
import json, time
#import numpy as np
from telnetlib import Telnet
import sys
import numpy as np 
import pylab 
from scipy import stats,polyval 
from buildModel import buildModel, predictDirection

tn=Telnet('localhost', 13854)
tn.write('{"enableRawOutput": true, "format": "Json"}'.encode('ascii'))
f = open('output', 'a')

def writeToFile(output):
    val = ""
    for i in range(len(output)-1):
            val += str(output[i]) + ","
    val += str(output[len(output)-1])
    val += "\n"
    with open('output.csv', 'a+') as f:
        f.write(val)
    # a = np.asarray(output)
    # np.savetxt("foo.csv", a, delimiter=",")

def checkDataSource(var):

    '''
        {u'eegPower': {u'lowGamma': 5144, u'highGamma': 2510, u'highAlpha': 18055, u'delta': 53387,
        u'highBeta': 13139, u'lowAlpha': 27772, u'lowBeta': 6340, u'theta': 81641}, u'poorSignalLevel': 0,
        u'eSense': {u'meditation': 61, u'attention': 50}}
    '''
    eegData = {
        'blinkstrength' : 0,
        'attention' : 0
    }

    while True:
        line=tn.read_until(b'\r')
        jsonValue=json.loads(line.decode('utf-8'))
        output = []
        if "rawEeg" not in jsonValue:
            #print(str(jsonValue))
            pass
        if 'eegPower' in jsonValue:
            eegData['lowGamma'] = int(jsonValue['eegPower']['lowGamma'])
            eegData['highGamma'] = int(jsonValue['eegPower']['highGamma'])
            eegData['highAlpha'] = int(jsonValue['eegPower']['highAlpha'])
            eegData['delta'] = int(jsonValue['eegPower']['delta'])
            eegData['highBeta'] = int(jsonValue['eegPower']['highBeta'])
            eegData['lowAlpha'] = int(jsonValue['eegPower']['lowAlpha'])
            eegData['lowBeta'] = int(jsonValue['eegPower']['lowBeta'])
            eegData['theta'] = int(jsonValue['eegPower']['theta'])
            output.append(eegData['lowGamma'])
            output.append(eegData['highGamma'])
            output.append(eegData['highAlpha'])
            output.append(eegData['delta'])
            output.append(eegData['highBeta'])
            output.append(eegData['lowAlpha'])
            output.append(eegData['lowBeta'])
            output.append(eegData['theta'])
            #print(eegData['lowGamma'], eegData['lowGamma'], eegData['highAlpha'], eegData['delta'])
        if "eSense" in jsonValue:
            eegData['attention'] = int(jsonValue['eSense']['attention'])
            eegData['meditation'] = int(jsonValue['eSense']['meditation'])
            #print('attention:\t' + str(eegData['attention']))
            output.append(eegData['attention'])
            output.append(eegData['meditation'])
            
        if "blinkStrength" in jsonValue:
            eegData['blinkstrength'] = int(jsonValue['blinkStrength'])
            print('blinkstrength:\t' + str(eegData['blinkstrength']))
            #output.append(eegData['blinkstrength'])
           
        if (len(output) >= 10):
            print(output)
            # if (eegData['blinkstrength'] <= 50):
            #     print("left")
            # else:
            #     print("right")
            # if (eegData['attention'] > 75):
            #     print("forward")
            # elif (eegData['attention'] < 40 and eegData['meditation'] < 40):
            #     print('backward')
            
            print(predictDirection(output))
            if (var == 1):
                writeToFile(output)
            elif (var == 2):
                file_ = './model.h5'
                buildModel(file_)
            elif (var == 3):
                print(predictDirection(output))
            output = []

        
        #sys.stdout.flush()
        #f.write(str(eegData) + "\n")


def console_():
    print("Press 1: to take input from sensor")
    print("Press 2: to train model")
    print("Press 3: to start driving")

if __name__=="__main__":
    
    var = input("Enter operation to perform?")
    checkDataSource(var)
