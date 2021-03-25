trainFile = 'train.txt'
testFile = 'test.txt'

word_dict = {}
neutral_words = {}
cnt_word = [0,0]
cnt_sent = [0,0]

def isText(word):
    for ch in word:
        if ch < 'a' or ch > 'z':
            return False
    return True

def cleanTokenizedText(text):
    ret = []
    for word in text:
        word = word.lower()
        if isText(word)and not (word in neutral_words):
            ret.append(word)
    return ret


def loadNeutral():
    global neutral_words
    with open('neutral_words.txt') as f:
        words = f.read()
    words = words.split('\n')
    for word in words:
        neutral_words[word] = True

def loadTrainData(file):
    global word_dict, cnt_word, cnt_sent
    with open(file) as f:
        text = f.read()

    for line in text.split('\n'):
        if len(line) < 2:
            continue
        line = line.split()
        label = int(line[-1])
        line = cleanTokenizedText(line)

        cnt_word[label] += len(line)
        cnt_sent[label] += 1

        for word in line:
            if not word in word_dict:
                word_dict[word] = [0,0]
            word_dict[word][label] += 1


def calcProb(word, label):
    ret = 1

    if word in word_dict:
        ret += word_dict[word][label]
    
    ret /= cnt_word[label] + len(word_dict)

    return ret

def predict(text):
    text = cleanTokenizedText(text)

    n_sent = sum(cnt_sent)
    prob_n = cnt_sent[0] / n_sent
    prob_p = cnt_sent[1] / n_sent

    for word in text:
        prob_n *= calcProb(word, 0)
        prob_p *= calcProb(word, 1)

    if prob_p > prob_n:
        return 1
    return 0

def evaluate(file):
    correct = 0
    n = 0

    with open(file) as f:
        text = f.read()

    for line in text.split('\n'):
        n+=1
        if len(line) < 2:
            continue
        line = line.split()
        label = int(line[-1])
        line = cleanTokenizedText(line)

        prediction = predict(line)

        if prediction == label:
            correct += 1 
    
    print(correct/n)



def main():
    loadNeutral()
    loadTrainData(trainFile)
    evaluate(testFile)



if __name__ == '__main__':
    main()

