import random

N=1000
D=10

p=2
q=6

f=open("test.tsv","w+")
random.seed(15)
for i in range(N):
	a=random.uniform(p,q)
	if (a>0.5*(p+q)):
		label=1
	else:
		label=0
	for j in range(D-1):
		f.write(str(a)+" ")
	f.write(str(a)+"\t"+str(label)+"\n")
