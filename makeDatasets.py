import os
import re
files = os.listdir("./datasets/")

def makeDataset():
    left = []
    right = []
    for file_ in files:
        if (re.search('.*_left\.csv$', file_)):
            print(file_)
            left.append(file_)
        if (re.search('.*_right\.csv$', file_)):
            print(file_)
            right.append(file_)

    if (os.path.exists('./left.csv')):
        os.remove('./left.csv')
    for file_ in left:
        with open('./left.csv', 'a+') as f:
            f_ = open('./datasets/' + file_, 'r')
            f.write(f_.read())
    if (os.path.exists('./right.csv')):
        os.remove('./right.csv')
    for file_ in right:
        with open('./right.csv', 'a+') as f:
            f_ = open('./datasets/' + file_, 'r')
            f.write(f_.read())
        
makeDataset()