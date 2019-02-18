
#!/usr/bin/env python
from pprint import pprint
from sseclient import SSEClient
import time, sys
import RPi.GPIO as GPIO
import signal
import json
GPIO.setwarnings(False)
messages = SSEClient('http://192.168.43.120:3000/events/sensor')

class State:
    STOPPED = "Stopped"
    FORWARD = "Forward"
    ROTATE = "Rotate"

class StateMachine:

    BLINK_THRESH = 60
    ATTEN_THRESH = 50

    def __init__(self):

        self.state = State.STOPPED

        self.blink = 0
        self.attention = 0

        print('StateMachine Initialized')
        '''
        {"type":"blink","payload":181}
        {"type":"sense","payload":{"attention":48,"meditation":23}}
        '''

    def perform_action( self, source):

            
        if source['type'] == 'blink':
            self.blink = int(source['payload'])
        
        if source['type'] == 'sense':
            self.attention = int(source['payload']['attention'])

        if self.state == State.STOPPED:
            
            if self.blink > StateMachine.BLINK_THRESH:
                self.state = State.ROTATE
                rotate()
            
            elif self.attention > StateMachine.ATTEN_THRESH:
                self.state = State.FORWARD
                forward()
        
        elif self.state == State.ROTATE:
            if self.blink < StateMachine.BLINK_THRESH:
                self.state = State.STOPPED
                stop()
        
        elif self.state == State.FORWARD:
            if self.attention < StateMachine.ATTEN_THRESH:
                self.state = State.STOPPED
                stop()
        
        return self.state




def main():
    s = StateMachine()
    for msg in messages:
        
        msg = json.loads(str(msg))
	msg = dict(msg)
        print(s.perform_action(msg))

def graceful_stop(signum, frame):
    stop()
    GPIO.cleanup()
    sys.exit(0)

def forward():
    GPIO.output(12,GPIO.HIGH)
    GPIO.output(16,GPIO.LOW)
    GPIO.output(20,GPIO.HIGH)
    GPIO.output(21,GPIO.LOW)
    return 'FORWARD'

def stop():
    GPIO.output(12,GPIO.LOW)
    GPIO.output(16,GPIO.LOW)
    GPIO.output(20,GPIO.LOW)
    GPIO.output(21,GPIO.LOW)
    return 'STOP'


def rotate():
    GPIO.output(20,GPIO.LOW)
    GPIO.output(21,GPIO.LOW)
    GPIO.output(12,GPIO.HIGH)
    GPIO.output(16,GPIO.LOW)
    return 'ROTATE'

if __name__ == '__main__':

    signal.signal(signal.SIGINT, graceful_stop)

    GPIO.setmode(GPIO.BCM)

    GPIO.setup(12,GPIO.OUT)
    GPIO.setup(16,GPIO.OUT)
    GPIO.setup(20,GPIO.OUT)
    GPIO.setup(21,GPIO.OUT)
    main()
