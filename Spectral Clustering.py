# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:31:31 2020

Please use python versin 3.7.6

Ensure the files are within same folder as this code

@author: Akshat Chauhan
"""

import numpy as np
import pandas as pd
from scipy import sparse
#from scipy.sparse import spdiags
#from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import networkx as nx



def K_Means(x,cno=3):

    from scipy.sparse import csc_matrix, find
    import copy
    dim=x.shape[0]
    m = x.shape[1]
    
    np.random.RandomState(seed=30)
    
    me=np.mean(x,axis=1)
    std=np.std(x,axis=1)
    c=np.zeros((cno,dim))
    for i in range(cno):
        c[i]=me.T-std.T*(i)
    c=c.T    
        
        
    #c =  np.random.rand(dim, cno)
    c_old = copy.deepcopy(c) + 10
    i = 1
    # check whether the cluster centers still change

    while np.linalg.norm(c - c_old, ord = 'fro') > 1e-6:
        print("--iteration %d \n" % i)
    
        #record previous c;  
        c_old = copy.deepcopy(c)
    
        # Assign data points to current cluster;
        # Squared norm of cluster centers.
        cnorm2 = np.sum(np.power(c,2), axis = 0)
        tmpdiff = 2 * np.dot(x.T,c) - cnorm2
        labels = np.argmax(tmpdiff,axis = 1)
    
        # Update data assignment matrix;
        # The assignment matrix is a sparse matrix,
        # with size m x cno. Only one 1 per row.
        P = csc_matrix( (np.ones(m) ,(np.arange(0,m,1), labels)), shape=(m, cno) )

        # adjust the cluster centers according to current assignment; 
        obj = 0
        for k in range(0,cno):
            idx = find(P[:,k])[0]
            nopoints = idx.shape[0]
            if nopoints == 0:
                # a center has never been assigned a data point; 
                # re-initialize the center; 
                c[:,k] = np.random.rand(dim,1)[:,0]
            else:
                # equivalent to sum(x(:,idx), 2) ./ nopoints;   
                c[:,k] = ((P[:,k].T.dot(x.T)).T / float(nopoints))[:,0] 

            obj = obj + np.sum(np.sum( np.power(x[:,idx] - c[:,k].T.reshape(dim,1), 2) ))
        print ("Objective Function is .......{}".format(obj))  

        i = i + 1

    #toc = time.time()

    #print('Elapsed time is %f seconds \n' % float(toc - tic))
    
    #print('obj =', obj)
    
    return c.T,labels.T








def import_graph():
    # read the graph from 'play_graph.txt'
    #df=pd.read_file f_path = abspath("play_graph.txt")
    df = pd.read_csv('edges.txt', sep="\t", header=None)
    df.columns = ["Source", "Target"]

    return df



def read_team_name():
    # read inverse_teams.txt file
    df = pd.read_csv('nodes.txt', sep="\t", header=None)
    df.columns = ['Node_ID', 'Node_name', 'Label','Blog_domain']
    #df.columns = ["Source", "Target"]

    return df


G = nx.Graph()

idx2name_pd = read_team_name()
idx2name = idx2name_pd.values[:,0]

templst=idx2name_pd['Node_name'].values

old_dic={i+1:j for i,j in enumerate(templst)}


a = import_graph().copy()
b=a.values
#removing self connected edges
edges=[(i,j) for i,j in b if i!=j]


## removing isoltated nodes and edges

G.add_nodes_from(idx2name)
G.add_edges_from(edges)
#nx.draw(G)
len(list(nx.isolates(G)))

G.remove_nodes_from(list(nx.isolates(G)))
#idx2name_new=list(G.nodes())
for component in list(nx.connected_components(G)):
    if len(component)<3:
        lst=[]
        for node in component:
            lst.append(node)
            print(lst)
            G.remove_node(node)
        edges=[(i,j) for i,j in edges if (i!=lst[0] and i!=lst[1])]
        #.remove_edge(lst[0],lst[1])


idx2name_new=list(G.nodes())


#converting edges from node id - node id to name-name
edges_with_name=[(old_dic[i],old_dic[j]) for (i,j) in edges]

len(idx2name_new)

# setting up new dataframe with selected nodes as above 

new_nodes_pd = idx2name_pd[idx2name_pd.Node_ID.isin(idx2name_new)]
new_nodes_pd.reset_index(drop=True,inplace=True)
df1=new_nodes_pd.copy()
df1['Node_ID_new']=list(df1.index)
df1.drop(['Node_ID'],axis=1,inplace=True)

new_nodes=df1['Node_name'].values
new_dic = {j:i for i,j in enumerate(new_nodes)}

edges_new_nodes=np.array([[new_dic[i],new_dic[j]] for (i,j) in edges_with_name ])
type(edges_new_nodes)

#edges_new_nodes=np.array(edges_new_nodes)
#list(G.edges)

n = len(idx2name_new)
k = 2


i = edges_new_nodes[:, 0]

j = edges_new_nodes[:, 1]
v = np.ones((edges_new_nodes.shape[0], 1)).flatten()

A = sparse.coo_matrix((v, (i, j)), shape=(n, n))
A = (A + np.transpose(A))/2

D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1)
#L = D @ A @ D
I=np.identity(A.shape[0])

L=I-D@A@D
L.shape
v, x = np.linalg.eigh(L)

x=x.real
v=v.real

fig = plt.figure(figsize=(9, 9))
plt.plot(v[0:10])
plt.show()

# Max eigen gap is observed between 2 and 3. Hence we will pick eigen vectors 1 and 2 


p = x[:,1:k+1].real

p = p/np.repeat(np.sqrt(np.sum(p*p, axis=1).reshape(-1, 1)), k, axis=1)
  # scatter
#plt.scatter(p[:, 0], p[:, 1])
#plt.show()

#kmeans = KMeans(n_clusters=k).fit(p)
#c_idx = kmeans.labels_


q,c_idx=K_Means(p.T,k)

c_idx.shape

idx2name_new=df1['Node_ID_new'].values

for i in range(k):
    print(f'Cluster {i+1}\n***************')
    idx = [index for index, t in enumerate(c_idx) if t == i]
    for index in idx:
        #print(idx2name_new[index])
        #print('\n')
        pass
        


df1['Predicted_Label']=c_idx


accuracy=len(df1[df1['Label']==df1['Predicted_Label']])/len(df1)
print("False Classifiation rate is {}%".format((1-accuracy)*100))

    # k-means
#kmeans = KMeans(n_clusters=k).fit(x)
#c_idx = kmeans.labels_