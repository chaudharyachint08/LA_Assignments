"Problem 1 of LA assignment"
import os,sys,copy,time
import LA

if __name__ == '__main__':
    argv = sys.argv
    part = argv[1][6:]
    IP = open(argv[2])
    OP = open('output_problem1_part1.txt','w') if part=='one' else open('output_problem1_part2.txt','w')

    N,K = (4,4) if (part=='one') else ( tuple(map(int,IP.readline().strip().split())) )
    C = tuple(map(float,IP.readline().strip().split()))
    mat = []
    for i in range(N):
        mat.append( list(map(float,IP.readline().strip().split())) )
    M = tuple(map(float,IP.readline().strip().split()))
    #N,K = len(C),len(mat[0])
    for j in range(K):
        if sum(mat[i][j] for i in range(N))>1:
            res, finite = False, None
            break
    else:
        #Call to LA Module, with Constraints
        ech,inv,pivots,ops,res,finite,sol,equations = LA.Solve(mat,C,[0]*len(M),M)
        if type(sol) is Exception:
            #print('Constrained Can\'t be Satisfied')
            res, finite = False, False

    flg = False
    if res == True:
        quantity = [float(x) for x in sol]
        if all( (0<=quantity[i] and quantity[i]<=M[i]) for i in range(K) ):
            flg = True            
    if flg:
        if finite==True:
            LA.my_print('EXACTLY ONE!',out=OP)
            LA.my_print(*(round(x,3) for x in quantity),out=OP)
        elif finite==False:
            LA.my_print('MORE THAN ONE!',out=OP)
            LA.my_print(*(round(x,3) for x in quantity),out=OP)
            LA.print_equations(ech,equations,K,out=OP)
    else:
        LA.my_print('NOT POSSIBLE, SNAPE IS WICKED!',out=OP)
        #Below is commented Code that can find maximum Lin_Algebrica preparable under constraints
        '''
        if part=='two' and res:
            res = LA.Matmul(mat,[[i] for i in M])
            LA.my_print(*(round(i[0],3) for i in res),out=OP)
        '''

    IP.close()
    OP.close()
