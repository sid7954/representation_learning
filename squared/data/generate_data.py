import random
import os

N=10000
D=10

f=open("train.tsv","w+")
random.seed(15)
for i in range(N):
	a=random.uniform(-1,1)
	for j in range(D-1):
		f.write(str(a)+" ")
	f.write(str(a*a))	
	f.write("\t"+str(a*a)+"\n")