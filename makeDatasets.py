import os
import re


class MakeDataSets:
    def __init__(self):
        self.files = os.listdir("./datasets/")
        self.changes = self.isDataSetsAdded()
        self.left_csv = './left.csv'
        self.right_csv = './right.csv'

    def makeDataSet(self):
        left = []
        right = []
        for file_ in self.files:
            if re.search('.*_left\.csv$', file_):
                print(file_)
                left.append(file_)
            if re.search('.*_right\.csv$', file_):
                print(file_)
                right.append(file_)

        if os.path.exists(self.left_csv):
            os.remove(self.left_csv)
        for file_ in left:
            with open(self.left_csv, 'a+') as f:
                f_ = open('./datasets/' + file_, 'r')
                f.write(f_.read())
        if os.path.exists(self.right_csv):
            os.remove(self.right_csv)
        for file_ in right:
            with open(self.right_csv, 'a+') as f:
                f_ = open('./datasets/' + file_, 'r')
                f.write(f_.read())
        file_d = open('changes_file', 'w+')
        file_d.write(str(len(self.files)))
        file_d.close()

    def isDataSetsAdded(self):
        file_ = open('changes_file', 'r')
        if int(file_.read()) == len(self.files):
            return False
        else:
            return True


if __name__ == "__main__":
    MakeDataSets().makeDataSet()
