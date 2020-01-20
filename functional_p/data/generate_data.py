import random
import os


#x_1 + x_D + x_2^2 + x_3^2 ........ + x_(D-1)^2 = 0

N=1000
D=10

f=open("test.tsv","w+")
random.seed(15)
for i in range(N):
	a=random.uniform(-1,1)
	f.write(str(a)+" ")
	sq_sum=0
	for j in range(D-2):
		b=random.uniform(-1,1)
		f.write(str(b)+" ")
		sq_sum=sq_sum+b*b
	c=-a-sq_sum 	
	f.write(str(c))	
	f.write("\t"+str((-3*a) + (5*c) + sq_sum)+"\n")