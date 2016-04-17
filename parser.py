from pyparsing import Word, alphas, Optional
import pickle

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


symbols="-'"



txt=open("dale_chall.txt")
n_lines=file_len("dale_chall.txt")
n_count=1
word_list=[]
print n_lines

for i in range(n_lines):
    line=txt.readline()
    print 'line', line
    parse1=Word(alphas)
    parsed=parse1.parseString(line)
    print 'parsing', parsed[0],parsed
    word_list.append(parsed[0])

print len(word_list), word_list

pickle.dump( word_list, open('dale_chall.p', "wb" ) )


