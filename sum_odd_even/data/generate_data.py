import random

N=10000
D=10

f=open("train.tsv","w+")
random.seed(5)
for i in range(N):
	sum=0
	for j in range(D-1):
		a=random.randint(0,1)
		sum+=a
		f.write(str(a)+" ")
	a=random.randint(0,1)
	sum+=a
	f.write(str(a))	

	if (sum%2==0):
		f.write("\t"+"0"+"\n")
	else:
		f.write("\t"+"1"+"\n")