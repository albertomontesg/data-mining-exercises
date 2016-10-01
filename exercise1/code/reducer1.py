import sys

current_word = None
current_count = 0

for line in sys.stdin:
    line = line.strip()

    word, count = line.split('\t')
    count = int(count)

    if word == current_word:
        current_count += 1
    else:
        if current_word is not None:
            print('{}\t{}'.format(current_word, current_count))
        current_word = word
        current_count = 1

print('{}\t{}'.format(current_word, current_count))
