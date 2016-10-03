import io
import re
import sys

pattern = re.compile(r'[A-Za-z]+')
# For Python3 helps to ignore decoding problems with a bad input
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
for line in input_stream.readlines():
    for match in pattern.findall(line):
        word = match.lower()
        print('{}\t{}\t{}'.format(word[0], word, 1))
