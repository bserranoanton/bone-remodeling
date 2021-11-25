# -*- coding: utf-8 -*-

#MIT LICENCE
#AUTHORS: BELÉN SERRANO, CLEMENTE F. ARIAS AND GERARDO E. OLEAGA

import random
import numpy as np 
import copy
from numba import jit

def readInitialitationB(fileName):
    
    with open(fileName, "r") as f:
        content = f.readline().split(",") #assume that content is in first line
    
    B = np.zeros(len(content))
    for i in range(len(content)):
        B[i] = float(content[i])
        
    return B

def makeRandomInitialitationB(rows, alpha, beta, B0, fileNameToSave):
    B = np.zeros(rows)
    # for i in range(rows):
    #     number = 4*random.random()
    #     while (number <2*beta):
    #          number = 4*random.random()
    #     B[i] = number
    
    for i in range(rows):
        #B[i] = 10*random.random()
        B[i] = B0

            #save random initialitation in a file
    with open(fileNameToSave + "N" + str(rows) + ".txt", "w") as f:
        for val in B[:-1]:
            f.write(str(val) + ",")
        f.write(str(B[-1]))
        
    return B

def makeRandomInitialitationC(rows, cols, fileNameToSave):
    C = np.zeros([rows, cols])
    
    #choose randomly the positions with "1"
    numberOf1 = random.randint((rows*cols/2), (rows*cols))
    print(numberOf1)
    positions = random.sample(range(rows*cols), numberOf1)

    #Savepositions with "1"
    with open(fileNameToSave + "N" + str(rows) + "M" + str(cols) + ".txt", "w") as f:
        for pos in positions[:-1]: 
            C[int(pos/cols), pos%cols] = 1
    
            f.write(str(pos) + ",")
            
        f.write(str(positions[-1]))
        
    return C

def readInitialitationC(rows, cols, fileName):
    with open(fileName, "r") as f:
        content = f.readline().split(",") #assume that content is in first line
    
    C = np.zeros([rows, cols])
    for pos in content:
        C[int(int(pos)/cols), int(pos)%cols] = 1
        
    return C

def readTransitionRules(fileName):
    
    transitionRules = []
    
    with open(fileName, "r") as f:
        content = f.readlines()
    
    for line in content:
        cleanRule = line.replace("\n", "").split(",")
        transitionRules.append([int(numeric_string) for numeric_string in cleanRule])
        
    return transitionRules

def countOnNeig(i, j, N, M, n_neig, C):
    
    nOnNeig = 0
    
    if (n_neig == 1):
        for h in range(i - 1, i + 2):
            if h < 0 or h >= N:
                continue
            for w in range(j - 1, j + 2):
                if (h == i and w == j):
                    continue
                
                #make a cilinder
                elif w < 0:
                    if C[h,M-1] == 1:
                        nOnNeig +=1
                        
                elif w >= M:
                    if C[h,0] == 1:
                        nOnNeig +=1
                        
                elif C[h][w] == 1:
                    nOnNeig += 1
    else:
        nOnNeig = -1
                
    return nOnNeig

# cuenta los vecinos globalmente devolviendo una matriz que contiene el número de vecinos activos
def countNeig(C):
  m,n = C.shape
  Cen = np.zeros((m+2,n+2))
  # no active neighbours up or down
  Cen[1:-1,1:-1] = C
  # ciclyc in columns
  Cen[1:-1,0] = C[:,-1]
  Cen[1:-1,-1] = C[:,0] 
  neig = Cen[0:-2,1:-1] + Cen[2:,1:-1] + Cen[1:-1,0:-2] + Cen[1:-1,2:] # up, down, left, right 
  neig +=  Cen[0:-2,0:-2] + Cen[0:-2,2:] + Cen[2:,0:-2] + Cen[2:,2:] # diagonals
  return neig

# función que aplica una regla según el nro de vecinos activos NeigOn y modifica C "in place" (no devuelve valores) 
#@jit(nopython=True)
def applyRule(Cneig,C,rule):
  m,n = C.shape
  for i in range(m):
    for j in range(n):
      C[i,j] = rule[int(C[i,j] * 9 + Cneig[i,j])]
  return None


def runSimulationInTime(t_ini, t_max, C, B, alpha, beta, Jfix, recordFileName, 
                        maxZeros):
    
    zeroCount = 0
    for t in range(t_ini, t_ini + t_max):
    
        Cneig = countNeig(C) # counts the neighbors of each cell
        applyRule(Cneig,C,rule) # C is renewed in place according to the rule
        #update B
        B += alpha * C[:, Jfix] - beta

        with open(recordFileName, "a") as f:
            for i in range(len(B)):
                if(B[i] < 0):
                    #f.write("time: " + str(t) + " b_i: " + str(i) + "\n")
                    f.write(str(t) + "," + str(i) + "\n")
                    C[i,:] = 0
                    B[i] = B0 
                    zeroCount += 1                    
        #if(zeroCount > maxZeros):
        #    break

    return None

if __name__ == "__main__":
    
    #initialitation of params
    alpha = 0.5
    beta = 0.3
    Jfix = 25
    B0 = 10
    t_max = 1000
    n_neig = 1 #only considers the next neigh
    N = 100 #rows
    M = 50 #columns
    zeroCount = 0
    maxZeros = 1500 #we are not having into account
    
    t_max_microFrac = 1000
    maxZerosMicroFrac = 1500 #we are not having into account
    
    #-------------------------------------------------------
    #Initialitation of B (the file contains all the elements of B "," separated)
    
    #read initialitation
    BiniFileName = "randomIniBN100.txt"
    #B = readInitialitationB(BiniFileName) 
    #make sure that the number of elements in the file is the same as N
    
    #make a random initialitaion
    BiniFileName = "randomIniB"
    B = makeRandomInitialitationB(N, alpha, beta, B0, BiniFileName) #no .txt and do not specify 
    #now Bi = B0
    #the number of rows (does it autom)
    Bini = copy.deepcopy(B)
    #-------------------------------------------------------
    
    
    #-------------------------------------------------------
    #Initialitation of C
    
    #read initialitation
    CiniFileName = "randomIniCN100M50.txt"
    C = readInitialitationC(N, M, CiniFileName)
    #make sure that the number of elements in the file is the same as N
    
    #make a random initialitaion
    #CiniFileName = "randomIniC"
    #C = makeRandomInitialitationC(N, M, CiniFileName) #no .txt and do not specify 
    #the number of rows (does it autom)
    Cini = copy.deepcopy(C)
    #-------------------------------------------------------
    
    
    #-------------------------------------------------------
    #Read transition rules (make sure n_neig is coherent)
    #list of transition rules
    nameFileTransitionRules = "reglas_transicion_validas"
    transitionRules = readTransitionRules(nameFileTransitionRules + ".txt")
    #-------------------------------------------------------
    
    #-------------------------------------------------------
    #Record file
    recordFileName = nameFileTransitionRules + "microFracv2_N" + str(N) + "M" + str(M) + ".txt"
    #recordFileName = "aa"
    with open(recordFileName, "w") as f:
        f.write("alpha: " + str(alpha) + "\n" +
                "beta: " + str(beta) + "\n" +
                "b_J: " + str(Jfix) + "\n" +
                "t_max: " + str(t_max) + "\n" +
                "n_neig: " + str(n_neig) + "\n" +
                "B_ini: " + BiniFileName + "\n" +
                "C_ini: " + CiniFileName + "\n" +
                "B0:" + str(B0) + "\n")

#-------------------------------------------------------
    
#-------------------------------------------------------
#Start simulation
    
#randomly chooses 200 rules 
#randomRulesIndexes = random.sample(range(0, len(transitionRules)), 200)
    
for rule in transitionRules:
#for ruleIndex in randomRulesIndexes: 
#    rule = transitionRules[ruleIndex]
    print("Rule " + str(rule))
    C = copy.deepcopy(Cini)
    B = copy.deepcopy(Bini)
    #zeroCount = 0
        
    with open (recordFileName, "a") as f:
        f.write("rule: " + str(rule) + "\n")
        
        
    runSimulationInTime(0, t_max, C, B, alpha, beta, Jfix, recordFileName, 
                        maxZeros)
        
    CbeforeMicro = copy.deepcopy(C)
    #Simulate microfracture (4 types)
    for micro_type in range(4):
        C = copy.deepcopy(CbeforeMicro)
        if(micro_type == 0):#3 cols = 0
            #We select rows 30,31 and 32
            with open (recordFileName, "a") as f:
                f.write("Microfracture_0:\n")
            C[:, 30] = 0
            C[:, 31] = 0
            C[:, 32] = 0
        
        elif(micro_type == 1):#3 rows = 0
            #We selct cols 15,16 and 17
            with open (recordFileName, "a") as f:
                f.write("Microfracture_1:\n")
            C[15,:] = 0
            C[16,:] = 0
            C[17,:] = 0
        
        elif(micro_type == 2):#diagonal of dim 3 = 0
            #We select rows 60,61,62
            with open (recordFileName, "a") as f:
                f.write("Microfracture_2:\n")
            for rowDiag in [60,61,62]:
                indX = rowDiag
                indY = 0
                while(indX >= 0 and indY < M):
                    C[indX, indY] = 0
                    indX -= 1
                    indY += 1
        
        else:#scuare of side = 10
            #We select left down corner in [70,35]
            with open (recordFileName, "a") as f:
                f.write("Microfracture_3:\n")
            for j in range(35,45):
                for i in range(70,80):
                    C[i,j] = 0
        
        runSimulationInTime(t_max, t_max_microFrac, C, B, alpha, beta, Jfix, recordFileName, 
                        maxZerosMicroFrac)
    

        
        
        
        
        
        
        
    
