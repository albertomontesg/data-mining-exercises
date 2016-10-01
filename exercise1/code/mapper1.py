import io
import re
import sys

pattern = re.compile(r'[A-Za-z]+')
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
for line in input_stream.readlines():
    for match in pattern.findall(line):
        word = match.lower()
        print('{}\t{}'.format(word, 1))
