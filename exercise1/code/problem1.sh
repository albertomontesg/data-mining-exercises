cat data/* | python3 mapper1.py | sort | python3 reducer1.py > result1.dat
grep -e '^my\t' -e '^hello\t' -e '^little\t' -e '^friend\t' \
     -e '^say\t' -e '^to\t' result1.dat > solution1.txt
cat solution1.txt
