import pandas as pd

if __name__ == '__main__':

	run_type = 'subjs'
	filename1 = '../2013-03-11/summary/subjs.dat'
	filename2 = '../2013-03-20/summary/subjs.dat'

	a = pd.load(filename1)
	a = pd.DataFrame(a.values, index=a.index, columns=a.columns)
	b = pd.load(filename2)
	b = pd.DataFrame(b.values, index=b.index, columns=b.columns)
	new = pd.merge(a,b, how='outer', left_index=True, right_index=True, on=list(a.columns.values))
	new.index.names = a.index.names
	new.save('summary/' + run_type + '.dat')

