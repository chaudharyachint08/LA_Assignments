import sys,os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

import warnings
warnings.filterwarnings("ignore")

import LA,network

if 'output_plots' not in os.listdir():
    os.mkdir('output_plots')

if 'output_data' not in os.listdir():
    os.mkdir('output_data')

file_out = open('output_data/output_problem2.txt','w')

LA.streams = [sys.stdout,file_out]

print = LA.my_print

_ = sys.argv[1]
if _ == '-type=gram-schimdt':
    #TASK 3
    ip = open(sys.argv[2])
    mat = np.array([[float(x) for x in line.strip().split()] for line in ip.readlines()])
    print( str( LA.GS(mat) ) )

else:
    test_data_file_path = _
    try:
        train_data_file_path = sys.argv[2]
    except Exception as ex:
        'mnist_train.csv should be in Current Working Directory, Consider values from last fed model'
        pass #PP1, DOUBT ASKED ON PIAZZA
    else:
        'Take Training Set & Build Model, perform all tasks of Problem 2'
        df1 = pd.read_csv(train_data_file_path,index_col=False,header=None)
        #df1 = shuffle(df1)
        #TASK 1
        m,n = df1.shape
        train_mat = np.array(df1)
        mean      = train_mat.sum(axis=0)/train_mat.shape[0]
        center_train_mat = train_mat.T[1:].T-mean[1:]
        cov_mat   = np.dot(center_train_mat.T,center_train_mat)/train_mat.shape[1]
        #TASK 2
        eigv,eigV = np.linalg.eigh(cov_mat)
        eigV = eigV.T
        eig = {}
        for i in range(len(eigv)):
            if eigv[i] not in eig:
                eig[eigv[i]] = [eigV[i]]
            else:
                eig[eigv[i]].append( eigV[i] )
        if any(len(eig[i])>1 for i in eig):
            print('Repeating Eigen Values with there Multiplicity')
            print(*sorted( (x,len(eig[x])) for x in eig if len(eig[x])>1 ))
        else:
            print('No Repeating Eigen values are there')
        flag = True
        for i in range(len(eigV)):
            for j in range(i):
                if np.dot(eigV[i],eigV[j])>1e-12:
                    flag = False
                    break
            if flag==False:
                break
        if flag:
            print('Eigen Vectors are Orthogonal')
        else:
            print('Eigen Vectors are Non-Orthogonal')
        #TASK 4 & 5
        D     = n-1
        a,b,steps = 1,n-1,10
        r = np.exp(np.log(b/a)/(steps-1))
        dim,err = [],[]
        #Data Centrality before PCA
        data_set = (np.array(df1.loc[:,1:]))#-mean[1:]) #CHKPNT
        for i in range(steps):
            M = int(a*r**i)
            red_data_set,rec_err = LA.PCA(eig,M,data_set)
            dim.append(M)
            err.append(rec_err)
        plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
        plt.xlabel('Dimension Chosen')
        plt.ylabel('RMS Error')
        plt.title('Reconstruction Error Plot')
        _ = plt.plot(dim,err,'o-',c='r')
        plt.savefig('output_plots/problem_2_task_5.png')
        plt.clf()
        plt.close()
        #TASK 6
        #Data Centrality before KNN
        train_mat = train_mat - np.array([0]+list(mean[1:]))
        neighbors = list(range(2,5+1))
        knn_2d_acc = np.zeros((steps,len(neighbors)))
        for i in range(steps):
            M = int(a*r**i)
            for k in neighbors:
                #print(M,k,end = ' : ')
                _ = time.time()
                knn_2d_acc[i][k-2] = LA.kNN(eig,train_mat,M,k,0.0005,LA.euc)[0]
                #print(knn_2d_acc[i][k-2],time.time()-_)
        #print(knn_2d_acc)
        network.plot3D(knn_2d_acc,steps,neighbors)
        'Finding Values of Optimal M & K for kNN'
        try:
            M_opt,val = None,0
            dim_bound = n-1 #Set to 784 as of now
            dim_max = knn_2d_acc.max(axis=1)
            for i in range(steps):
                if dim[i]<=dim_bound and dim_max[i]>val:
                    val = dim_max[i]
                    M_opt = dim[i]
            K_opt  = list(knn_2d_acc[dim.index(M_opt)]).index(knn_2d_acc[dim.index(M_opt)].max())+2
        except:
            M_opt,K_opt = 40,2 #for given train data & euclidean distance
        f = open('history.txt','w')
        f.write('{}\n{}\n'.format(M_opt,K_opt))
        f.write(str(eig))
        f.close()
        #BONUS 1 (PROJECT VECTORS on 2d & LOOK FOR ANY PATTERN)
        data_set_2D = LA.PCA(eig,2,data_set)[0].T
        plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
        X,Y = np.array(data_set_2D[0]),np.array(data_set_2D[1])
        if X.var()<Y.var():
            X,Y = Y,X
        for i in list(range(10)):
            condition = train_mat.T[0]==i
            x = np.extract(condition,X)
            y = np.extract(condition,Y)
            _ = plt.scatter(x,y,s=1)
        plt.axis('off')
        plt.title('MNIST data in 2D')
        plt.savefig('output_plots/problem_2_bonus_1.png')
        plt.clf()
        plt.close()


        #SKLEARN KNN
        pass
        #CNN
        pass #PP2

    f = open('history.txt')
    M_opt = eval(f.readline().strip())
    K_opt = eval(f.readline().strip())
    eig   = eval(f.read().replace('array','np.array').strip())
    f.close()
    df1 = pd.read_csv('mnist_train.csv',index_col=False,header=None)

    'Report Accuracy & Metrices on Test data'
    df2 = pd.read_csv(test_data_file_path,index_col=False,header=None)
    train_mat  = np.array(df1)
    test_mat   = np.array(df2)
    acc_e,pred_labels_e = LA.kNN2(eig,train_mat,test_mat,M_opt,K_opt,LA.euc)
    acc_c,pred_labels_c = LA.kNN2(eig,train_mat,test_mat,M_opt,K_opt,LA.cos)
    print('Accuracy using Euclidean Metric on test data is {0}%'.format(round(acc_e*100,3)))
    print('Accuracy using Cosine    Metric on test data is {0}%'.format(round(acc_c*100,3)))

file_out.close()
