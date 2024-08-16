import sys
import pandas as pd

# The first argument is the name of the results file
results_file = sys.argv[1]

# extract all the lines from the file that start with "*******"
lines = open(results_file, 'r').readlines()
lines = [line for line in lines if line.startswith('*******')]

# split lines into two lists -- one for even indices and one for odd indices
even_lines = lines[::2]
odd_lines = lines[1::2]

# split each line into a list of words
even_lines = [line.split() for line in even_lines]
odd_lines = [line.split() for line in odd_lines]

results = []
for even, odd in zip(even_lines, odd_lines):
    benchmark = even[1]
    batchsize = int(even[2])
    time = float(odd[1]) * 1e-6
    result = {'benchmark': benchmark, 'batchsize': batchsize, 'time': time}
    results.append(result)

df = pd.DataFrame(results)
print(df.to_string(index=False))