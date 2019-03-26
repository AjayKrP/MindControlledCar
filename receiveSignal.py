import json
from telnetlib import Telnet


def writeToFile(output):
    val = ""
    for i in range(len(output) - 1):
        val += str(output[i]) + ","
    val += str(output[len(output) - 1])
    val += "\n"
    with open('output.csv', 'a+') as f:
        f.write(val)


class ReceiveSignal:
    def __init__(self):
        self.tn = Telnet('localhost', 13854)
        self.tn.write('{"enableRawOutput": true, "format": "Json"}'.encode('ascii'))
        self.f = open('output', 'a')

    def checkDataSource(self):
        """
            {u'eegPower': {u'lowGamma': 5144, u'highGamma': 2510, u'highAlpha': 18055, u'delta': 53387,
            u'highBeta': 13139, u'lowAlpha': 27772, u'lowBeta': 6340, u'theta': 81641}, u'poorSignalLevel': 0,
            u'eSense': {u'meditation': 61, u'attention': 50}}
        """
        eegData = {
            'blinkstrength': 0,
            'attention': 0
        }

        while True:
            line = self.tn.read_until(b'\r')
            jsonValue = json.loads(line.decode('utf-8'))
            output = []
            if "rawEeg" not in jsonValue:
                # print(str(jsonValue))
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
                # print(eegData['lowGamma'], eegData['lowGamma'], eegData['highAlpha'], eegData['delta'])
            if "eSense" in jsonValue:
                eegData['attention'] = int(jsonValue['eSense']['attention'])
                eegData['meditation'] = int(jsonValue['eSense']['meditation'])
                # print('attention:\t' + str(eegData['attention']))
                output.append(eegData['attention'])
                output.append(eegData['meditation'])

            if "blinkStrength" in jsonValue:
                eegData['blinkstrength'] = int(jsonValue['blinkStrength'])
                print('blinkstrength:\t' + str(eegData['blinkstrength']))
                # output.append(eegData['blinkstrength'])

            if len(output) >= 10:
                print(output)
                writeToFile(output)
