import heapq
import sys

A = 10
B = 35
CHAR_LIMIT = 30

current_char = None
current_word = None
current_count = 0
heap = []

def print_heap(heap):
    for word in heap:
        print('{}\t{}'.format(word[0], word[1]))

for line in sys.stdin:
    line = line.strip()

    word, count = line.split('\t', 2)
    char = word[0]
    count = int(count)

    if not current_char:
        # First pass
        current_char = char
        current_word = word
        current_count = count
    elif char != current_char:
        # When char changes the current heap should be printed
        print_heap(heap)
        heap = []
        current_char = char
        current_word = word
        current_count = count
    else:
        if word == current_word:
            current_count += count
        else:
            # When the word on the current letter changes
            if current_count >= A and current_count <= B:
                word = (current_word, current_count)
                if len(heap) == CHAR_LIMIT:
                    heapq.heappushpop(heap, word)
                else:
                    heapq.heappush(heap, word)
            current_word = word
            current_count = count

if heap:
    print_heap(heap)
