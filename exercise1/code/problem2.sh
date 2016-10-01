cat data/* | python3 mapper2.py | sort | python3 reducer2.py > solution2.txt
head -n 35 solution2.txt
