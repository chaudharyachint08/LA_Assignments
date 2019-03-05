from fractions import Fraction
import copy,math,random

gcd = math.gcd

def lcm(a,b):
    return (a*b)//gcd(a,b)

out = None

def simplex(A,B,Constraints,Weights):
    '''
    Maximize np.dot(Weights,Sol)
    np.dot(A,sol)<=Constraints
    '''
    #This Function is not yet written, & assumed to be not needed for assignment
    pass

def my_print(*arg,**kwarg):
    global out
    sep = kwarg['sep'] if 'sep' in kwarg else ' '
    end = kwarg['end'] if 'end' in kwarg else '\n'
    out = kwarg['out'] if 'out' in kwarg else None
    st = sep.join((str(x) for x in arg)) + end
    if out:
        out.write(st)
    print(st,end='')

class Fraction2:
    
    def __init__(self,x,y=None):
        if y==None:
            tmp = tuple(map(  int,str(Fraction(x).limit_denominator()).strip('()').split('/')  )) + (1,)
        else:
            tmp = tuple(map(  int,str(Fraction(x,y).limit_denominator()).strip('()').split('/')  )) + (1,)
        x,y = tmp[0],tmp[1]
        self.x = x
        self.y = y
    
    def val(self):
        s = -1 if (self.y<0) else 1
        return s*self.x , s*self.y

    def __str__(self):
        x,y = self.val()
        return str(x)+'/'+str(y)
    
    def __float__(self):
        return self.x/self.y

    def __add__(self,obj):
        if type(obj)!=type(Zero):
            obj = Fraction2(obj)
        num   = self.x*obj.y + obj.x*self.y
        denom = self.y * obj.y
        div   = gcd(num,denom)
        return Fraction2(num//div,denom//div)

    def __sub__(self,obj):
        if type(obj)!=type(Zero):
            obj = Fraction2(obj)
        num   = self.x*obj.y - obj.x*self.y
        denom = self.y * obj.y
        div   = gcd(num,denom)
        return Fraction2(num//div,denom//div)

    def __mul__(self,obj):
        if type(obj)!=type(Zero):
            obj = Fraction2(obj)
        num   = self.x * obj.x
        denom = self.y * obj.y
        div   = gcd(num,denom)
        return Fraction2(num//div,denom//div)

    def __truediv__(self,obj):
        if type(obj)!=type(Zero):
            obj = Fraction2(obj)
        if obj==Zero:
            return ZeroDivisionError('2nd Object is Zero')
        num   = self.x * obj.y
        denom = self.y * obj.x
        div   = gcd(num,denom)
        return Fraction2(num//div,denom//div)

    def __eq__(self,obj):
        if type(obj)!=type(Zero):
            obj = Fraction2(obj)
        return self.x * obj.y == self.y * obj.x

    def __ne__(self,obj):
        if type(obj)!=type(Zero):
            obj = Fraction2(obj)
        return self.x * obj.y != self.y * obj.x

    def __bool__(self):
        return bool(self.x)


Unit,Zero = Fraction2(1),Fraction2(0)
m,n = None,None

def Echelon_Form(AB,flg=True):
    'Function Convert the given matrix into Reduced Echelon Form'
    m,n = len(AB), len(AB[0])
    M   = copy.deepcopy(AB)
    I   = [[ (Unit if i==j else Zero) for j in range(n)]for i in range(m)]
    ops  = []
    ops2 = None
    pivots = []
    lst_pivot = -1
    
    for i in range(m):
        pivot_indx = lst_pivot+1
        while pivot_indx < n :
            for j in range(i,m):
                if M[j][pivot_indx] != Zero:
                    M[i],M[j] = M[j],M[i]
                    I[i],I[j] = I[j],I[i]
                    ops.append(('Swapup',i,j))  # R[i],R[j] = R[j],R[i]
                    break
            else:
                if ops2 is None:
                    ops2 = copy.deepcopy(ops)
                pivot_indx+=1
                continue
            lst_pivot = pivot_indx
            break
        else:
            break
        
        pivots.append((i,pivot_indx))
        #Dividing Row by value of pivot
        pivot = copy.deepcopy(M[i][pivot_indx])
        M[i] = M[i][:pivot_indx] + [M[i][j]/pivot for j in range(pivot_indx,n)]
        I[i] = [I[i][j]/pivot for j in range(n)]
        ops.append(('Divide',i,pivot))  # R[i] = R[i]/pivot

        #Row operations for rows below that row
        for k in range(i+1,m):
            val = copy.deepcopy(M[k][pivot_indx])
            M[k] = M[k][:pivot_indx] + [ M[k][j] - val*M[i][j] for j in range(pivot_indx,n)]
            I[k] = [ I[k][j] - val*I[i][j] for j in range(n) ]
            ops.append(('SubMul',k,i,val))  # R[k] = R[k] - R[i]*val (Here 'i' is row having pivot)

    if flg:
        for i,pivot_indx in pivots[::-1]:
            #Row operations for rows above that row
            for k in range(i):
                val = copy.deepcopy(M[k][pivot_indx])
                M[k] = M[k][:pivot_indx] + [ M[k][j] - val*M[i][j] for j in range(pivot_indx,n)]
                I[k] = [ I[k][j] - val*I[i][j] for j in range(n) ]
                ops.append(('SubMul',k,i,val))  # R[k] = R[k] - R[i]*val (Here 'i' is row having pivot)
        
    return M,I,pivots,ops,(ops if ops2 is None else ops2)

def print_equations(ech,equations,K,out):
    all_vars = set(range(K))
    pivots = set(int(x[0][1:].strip('= ')) for x in equations)
    free_vars = all_vars - pivots
    hsh = {}
    for x,y,z in equations:
        hsh[int(x[1:].strip('= '))] = (x,y,z)
    for var in sorted(all_vars):
        if var in free_vars:
            my_print('x'+str(var),end='; ',out=out)
        else:
            x,y,z = hsh[var]
            my_print(x,str(ech[y][n]),' - ',z,sep='',end='; ',out=out)

def Matmul(A,B):
    p,q1,q2,r = len(A),len(A[0]),len(B),len(B[0])
    if q1!=q2:
        return Exception('Dimensions Mismatch ({0},{1}),({2},{3})'.format(p,q1,q2,r))
    else:
        res = []
        for i in range(p):
            res.append([ sum((A[i][k]*B[k][j] for k in range(q1)),  (Zero if type(A[i][0])==type(Zero) else 0) ) for j in range(r) ])
        return res

def Solve(A,B,choices1=False,choices2=False):
    global m,n
    m,n = len(A),len(A[0])
    
    choices_given = choices1 and choices2
    if not choices1: choices1 = [0]  * n
    if not choices2: choices2 = [100]* n
    
    A = [[ Fraction2(A[i][j]) for j in range(n)] for i in range(m)]
    B =  [ Fraction2(B[i]) for i in range(m)]
    
    AB = copy.deepcopy(A)
    for i in range(m):
        AB[i].append(B[i])
    
    ech,inv,pivots,ops,ops2 = Echelon_Form(AB,True)    

    if __name__ == 'main':
        print("\nEchelon Form of Given System's Augmented Matrix is")
        for i in range(len(ech)):
            print('\t'.join(str(ech[i][j])for j in range(len(ech[0]))))
    
    #Finding ranks of A & AB from echelon form
    rank_A, rank_AB = 0,0
    for i in range(m):
        if any(ech[i][j] for j in range(n)):
            rank_A+=1
            rank_AB+=1
        elif ech[i][n]:
            rank_AB+=1
    
    res,finite,sol,equations = None,None,None,None
    
    if rank_A!=rank_AB:
        #No Solution
        res = False
        if __name__ == 'main':
            print('\nInconsistent System of Equations')
        
    elif len(pivots)==n:
        #Finite Solution
        res, finite = True, True
        sol = [ech[i][n] for i in range(m)]
        #print('\n'.join(str(x) for x in sol))
        for i in range(n):
            exec('x{0} = sol[{0}]'.format(i))
        if __name__ == 'main':
            print('\nFinite Solution of System is')
            for i in range(n):
                print('x{0} = {1}'.format(i,eval('x'+str(i))))
    else:
        #Infinitely Many Solutions
        res, finite = True, False
        free = sorted(set(range(n))-set(x[1] for x in pivots))

        if __name__ == '__main__':
            print('\nFree Variables are')
            print(', '.join('x'+str(x)for x in sorted(free)))    
            print('\nRest Equations are')

        equations = []
        for i,pivot_indx in pivots[::-1]:
            part_1 = 'x'+str(pivot_indx)+' = '
            if __name__ == '__main__': print(part_1,end='')   
            if __name__ == '__main__': print(str(ech[i][n])+' - ',end='')
            part_3 = '(' + ' + '.join('x'+str(x)+'*'+str(ech[i][x]) for x in range(pivot_indx+1,n) if ech[i][x]) + ')'
            if __name__ == '__main__': print( part_3 )
            equations.append((part_1,i,part_3))

        if choices_given:
            for i in range(10**max(4,len(free))): #10,000 maximum Random Free variables
                tmp = [random.randint(choices1[i],choices2[i]) for i in range(len(free))]                
                free_val =  [ Fraction2(i) for i in tmp]                
                for i in range(len(free)):
                    exec('x{0} = free_val[{1}]'.format(free[i],i))                    
                for x,y,z in equations:
                    exec(x+'ech[{0}][n]'.format(y)+' - '+z)                    
                sol = []
                for i in range(n):
                    exec('sol.append(x{0})'.format(i))                    
                if all( ( choices1[i]<=float(sol[i]) and float(sol[i])<=choices2[i] ) for i in range(n)):
                    break
            else:
                sol = Exception('Constrained Not Satisifed')
        else:
            tmp = [random.randint(choices1[i],choices2[i]) for i in range(len(free))]                
            free_val =  [ Fraction2(i) for i in tmp]                
            for i in range(len(free)):
                exec('x{0} = free_val[{1}]'.format(free[i],i))                    
            for x,y,z in equations:
                exec(x+'ech[{0}][n]'.format(y)+' - '+z)                    
            sol = []
            for i in range(n):
                exec('sol.append(x{0})'.format(i))                    

        if __name__ == '__main__':
            for i in range(n):
                print('x{0} = {1}'.format(i,eval('x'+str(i))))
        
    return ech,inv,pivots,ops,res,finite,sol,equations

if __name__ == '__main__':
    m,n = tuple(map(int,input('Enter Size of System of Equations\n').strip().split()))
    mat = []
    print('Enter Matrix A')
    for i in range(m):
        mat.append( [float(x) for x in input().strip().split()] )
    
    vec = [float(x) for x in input('Enter Vector B\n').strip().split()]
    
    ech,inv,pivots,ops,res,finite,sol,equations = Solve(mat,vec)
