import os

with open('index.txt') as f:
    with open('new_index.txt','wb') as n:
        for line in f:
            tokens = line.split()
            if tokens[2] == 'PSK4':
                tokens[2] = 'QAM4'
            n.write(' '.join(tokens)+os.linesep)
