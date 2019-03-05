"Problem 2 of LA assignment"
import os,sys,copy,time
import LA, numpy as np

if __name__ == '__main__':
    argv = sys.argv
    IP = open(argv[1])
    OP = open('output_problem2.txt','w')
    
    n = int(IP.readline().strip())
    mat = []
    for i in range(n):
        mat.append(list(map(float,IP.readline().strip().split())))

    mat2 = [[LA.Fraction2(mat[i][j])for j in range(n)]for i in range(n)]
    mat3 = LA.Matmul(mat2,mat2)
    
    t1 = time.clock()
    ech,inv,_,_,_ = LA.Echelon_Form(mat2)
    t1 = time.clock()-t1
    
    _,_,_,_,ops = LA.Echelon_Form(mat3)
    
    t2 = time.clock()
    try:
        _ = np.linalg.inv(mat)
    except:
        pass
    t2 = time.clock()-t2
    
    count = 0

    for i in range(n):
        flg = True
        for j in range(n):
            if i==j and not ech[i][j]:
                flg = False
            if i!=j and ech[i][j]:
                flg = False
        if flg:
            count += 1
    
    if count==n:
        LA.my_print('YAAY! FOUND ONE!',out=OP)
        for i in range(n):
            LA.my_print(*(round(float(inv[i][j]),3) for j in range(n)),out=OP)
            #LA.my_print(*(str(inv[i][j]) for j in range(n)),out=OP)
    else:
        LA.my_print('ALAS! DIDN\'T FIND ONE!',out=OP)
    
    for op in ops:
        if op[0]=='Swapup' and op[1]!=op[2]:
            LA.my_print('SWITCH',op[1]+1,op[2]+1,out=OP)
        elif op[0]=='Divide' and float(op[2])!=1:
            s = -1 if (float(op[2])<0) else 1
            #st = '/'.join( map(str,( s*int(x) for x in str(op[2]).split('/')[::-1] )) )
            LA.my_print('MULTIPLY',1/float(op[2]),op[1]+1,out=OP)    #st
        elif op[0]=='SubMul' and float(op[3])!=0:
            LA.my_print('MULTIPLY&ADD',-1*float(op[3]),op[2]+1,op[1]+1,out=OP)   #str(op[3]*-1)
    
    #print(t1,t2)
    
    IP.close()
    OP.close()