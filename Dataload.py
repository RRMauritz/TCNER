import pandas as pd

# Load training data --------------------------------
with open('DBLPTrainset.txt') as f:
    lines = f.readlines()

for i in range(len(lines)):
    pre = lines[i].split()[1:]
    lines[i] = [pre[0], ' '.join(pre[1:])]

# Store in pandas data frame
train = pd.DataFrame(lines, columns=['Label', 'Title'])

# Load test data------------------------------------------------
with open('DBLPTestset.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    pre = lines[i].split()[1:]
    lines[i] = [' '.join(pre[1:])]

test = pd.DataFrame(lines, columns=['Title'])

with open('DBLPTestGroundTruth.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()[1:][0]
test['Label'] = lines
