
#!/usr/bin/env python
from pprint import pprint
from sseclient import SSEClient
import time, sys
import RPi.GPIO as GPIO
import signal
import time
import json
import collections as cl
GPIO.setwarnings(False)
messages = SSEClient('http://192.168.43.120:3000/events/sensor')

class State:
    STOPPED = "Stopped"
    FORWARD = "Forward"
    ROTATE = "Rotate"
    isForward = False
    isRotate = False
    isStopped = False
class StateMachine:
    BLINK_THRESH = 56
    ATTEN_THRESH = 50
    def __init__(self):
        self.state = State.STOPPED
        self.blink = 0
        self.attention = 0
        self.start_time = time.time()
        self.isTraing = True
        self.b_arr = []
        self.a_arr = []
        self.blink_queue = cl.deque(maxlen=5)
        self.attention_queue = cl.deque(maxlen=5)
        print('StateMachine Initialized')
        '''
        {"type":"blink","payload":181}
        {"type":"sense","payload":{"attention":48,"meditation":23}}
        '''
    def sum(arr):
        sum = 0
        for i in range(len(arr)):
            sum += arr[i]
        return sum


    def find_average(self):
        return [sum(self.b_arr)/len(self.b_arr), sum(self.a_arr)/len(self.a_arr)]


    def find_median(self):
        first = self.b_arr[len(self.b_arr)/2]
        second = self.a_arr[len(self.a_arr)/2]
        return [first, second]


    def perform_action( self, source):
        if source['type'] == 'blink':
            self.blink = int(source['payload'])
            self.blink_queue = self.blink_queue.append(self.blink)
            self.blink_queue.sort()
        if source['type'] == 'sense':
            self.attention = int(source['payload']['attention'])
            self.attention_queue = self.attention_queue.append(self.attention)
            self.attention_queue.sort()
        self.blink = self.blink_queue[len(self.blink_queue)-1]
        self.attention = self.attention_queue[len(self.attention_queue)-1]
        if (self.isTraing):
            print(time.time() - self.start_time)
            if (time.time() - self.start_time < 60):
                self.a_arr.append(self.attention)
                self.b_arr.append(self.blink)
                self.isTraing = True
            else:
                self.isTraing = False
                arr = self.find_median()
                self.ATTEN_THRESH = arr[1]
                self.BLINK_THRESH = arr[0]
                print("Thresold value of blink:\t", self.BLINK_THRESH)
                print("Thresold of Attention:\t", self.ATTEN_THRESH)
        else:
            if (self.state == State.STOPPED):
                if self.blink > StateMachine.BLINK_THRESH:
                    if (not State.isRotate):
                        self.state = State.ROTATE
                        rotate()
                        print('rotate')
                        State.isRotate = False
                    else:
                        State.isRotate = True

                elif self.attention > StateMachine.ATTEN_THRESH:
                    self.state = State.FORWARD
                    forward()
                    print('forward')
                
            elif self.state == State.ROTATE:
                if self.blink < StateMachine.BLINK_THRESH:
                    self.state = State.STOPPED
                    stop()
                    print('stop')

            elif self.state == State.FORWARD:
                if self.attention < StateMachine.ATTEN_THRESH:
                    self.state = State.STOPPED
                    stop()
                    print('stop')
                elif self.blink > StateMachine.BLINK_THRESH:
                    self.state = State.STOPPED
                    stop()
                    print('stop')
            print('Blink:\t:' + str(self.blink) +", "+ "Attention:\t" + str(self.attention))
            #self.blink = 0
            #self.attention = 0
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



