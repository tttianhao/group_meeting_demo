import csv

name = 'homo'

csvfile = open('./data/' + name + '.csv')
csvreader = csv.reader(csvfile, delimiter = '\t')

fastafile = open('./data/' + name + '.fasta', 'w')

for rows in csvreader:
    fastafile.write('>' + rows[0].strip() + '\n')
    fastafile.write(rows[-1].strip() + '\n')