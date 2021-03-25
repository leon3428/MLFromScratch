import random

amazonFile = 'dataset/amazon_cells_labelled.txt'
imdbFile = 'dataset/imdb_labelled.txt'
yelpFile = 'dataset/yelp_labelled.txt'

lines = []

def loadData(file):
    global lines
    with open(file) as f:
        text = f.read()

    lines += text.split('\n')

loadData(amazonFile)
loadData(imdbFile)
loadData(yelpFile)

random.shuffle(lines)

with open('train.txt', 'w') as f:
    for i in range(0, 2500):
        f.write(lines[i] + '\n')
        
with open('test.txt', 'w') as f:
    for i in range(2500, len(lines)):
        f.write(lines[i] + '\n')