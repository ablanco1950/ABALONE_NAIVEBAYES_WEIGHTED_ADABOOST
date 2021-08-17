
import numpy as np

# ADABOOST IMPLEMENTATION =================================================
# Copied from Jaime Pastor https://github.com/jaimeps/adaboost-implementation 
# with some modifications
########################################################################

def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    for i in range(M):
       
       
        Y_train, pred_train_i, Y_test, pred_test_i, TotalHits = AbaloneNaiveBayes(Y_train, X_train, Y_test, X_test, w)
      
        miss=[float(0.0)]
        for x in range (len(pred_train_i)-1):
            miss.append(float(0.0))
        for x in range (len(pred_train_i)):
            if (pred_train_i[x] != Y_train[x]):
               miss[x]=1.0
            else:
                 miss[x]=0.0
       
        miss2 = [x if x==1 else -1 for x in miss]
        
        err_m = np.dot(w,miss) 
        err_m = err_m / sum(w)
        
        alpha_m = 0.07 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha_m for x in pred_test_i])]
    
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    
    return TotalHits

""" CLASIFIER FUNCTION ==========================================================="""

def AbaloneNaiveBayes(Y_train, X_Train, Y_test, X_test, W):
    
               
        Max=[float(2.0)]
        Max.append(float(0.815))
        Max.append(float(0.65))
        Max.append(float(1.13))
        Max.append(float(2.8255))
        Max.append(float(1.488))
        Max.append(float(0.76))
        Max.append(float(1.005))
        Min=[float(0.0)]
        Min.append(float(0.075))
        Min.append(float(0.055))
        Min.append(float(0.0))
        Min.append(float(0.002))
        Min.append(float(0.001))
        Min.append(float(0.0005))
        Min.append(float(0.0015))
        NumClases=3
        NumCampos =8
        TopeMemoria = 154
        
        # Got frm https://stackoverflow.com/questions/15448594/how-to-add-elements-to-3-dimensional-array-in-python
        TabVotos = np.zeros((NumCampos,TopeMemoria,NumClases))
        
        Maximo=0.0
        Conta=0.0
        Cont=-1
        
        ContClase=[float(0.0)]
        for j in range(NumClases  -1):
             ContClase.append(float(0.0))
       
        Start=0
        End = 3133
    
       
        f=open("C:\\abalone-1.data ","r")
        
        
        for linea in f:
            
            lineadelTrain =linea.split(",")
            Conta = Conta + 1
            if Conta < Start:
                continue
            if Conta > End:
                break
         
            Cont = Cont +1
            if len(W) == 1:
              FactorPri=1.0
            else:
                FactorPri=W[Cont]
                
            ClaseLeida=float(lineadelTrain[8])
            
            Clase=0    
        	
            if (ClaseLeida > 10.0): 
                Clase=2
            else:
                if (ClaseLeida > 8.0): 
                		Clase=1
          
            ContClase[Clase]=ContClase[Clase] + 1
                       
            
           
            z=-1
            for x  in lineadelTrain:
                
                z=z+1
                if z==NumCampos:
                    break
              
                if z==0:
                    if lineadelTrain[0] == "M":
                      ValorTrain=0.0
                    else:
                        if lineadelTrain[0] == "F":
                           ValorTrain=1.0
                        else:
                                if lineadelTrain[0] == "I":
                                    ValorTrain=2.0
                else:
                    ValorTrain =float(lineadelTrain[z])
                ValorTrain =ValorTrain - Min[z]
                Maximo = Max[z]
                Maximo = Maximo - Min[z]
                indice =(int) (((TopeMemoria - 2.0) * ValorTrain)/ Maximo)
                if ( (indice > (TopeMemoria-2)) or  (indice < 0)):	
                    print("index overflowed=" + str( indice) + " in  the field ="+ str(z-1) )
                    indice=TopeMemoria
                Wvalor=0.0
             
                Wvalor= TabVotos[z,TopeMemoria-1,Clase]
                Wvalor= Wvalor + 1
                TabVotos[z,TopeMemoria-1,Clase]=Wvalor 
                Wvalor= TabVotos[z,indice,Clase]
                Wvalor=Wvalor+FactorPri
                TabVotos[z,indice,Clase]=Wvalor 
       
        f.close()
        
        
        Start=0
        End = 3133
       
       
        SwTestTraining=1
        Y_train, X_train, TotHits = AbaloneNaiveBayes_train_test (TabVotos, ContClase,Max,Min, Start, End, SwTestTraining, Y_train, X_Train, Y_test, X_test, W)
      
        Start=3134
        End = 4177
       
        SwTestTraining=2
        Y_test, X_test, TotHits = AbaloneNaiveBayes_train_test (TabVotos, ContClase,Max,Min, Start, End, SwTestTraining, Y_train, X_Train, Y_test, X_test, W)
        return Y_train, X_train, Y_test, X_test, TotHits
        
    
########################################################################################
# APPLY NAIVE BAYES TO GET THE ODDS OF EACH CLASS FROM
# OF FREQUENCIES EACH INDEX OF EACH FIELD
###################################################################3333333333333333

def AbaloneNaiveBayes_train_test (TabVotos, ContClase,Max,Min, Start, End,SwTestTraining, Y_train, X_Train, Y_test, X_test, W): 
        
        NumClases=3
        NumCampos =8
        TopeMemoria = 154
        P_indice_clase=np.zeros((NumClases))
      
        Producto_P_indice_clase=np.zeros((NumClases))
       
        Valor_clase=np.zeros((NumClases))    
        TotAciertos=0.0
        TotFallos =0.0;
        
        Conta=0.0
        
        SwInicial=0;
        f=open("C:\\abalone-1.data ","r")
        with open("C:\\abalone-1Corrected.txt","w") as  w:
                for linea1 in f:
               
                     lineadelTrain =linea1.split(",")
                     Conta = Conta + 1
                 
                     if Conta < Start:
                         continue
                     if Conta > End:
                         break
                     
                     ClaseLeida=float(lineadelTrain[8])
                     
                     Clase=0.0  
        	
                     if (ClaseLeida > 10): 
                         Clase=2.0
                     else:
                         if (ClaseLeida > 8.0): 
            		             Clase=1.0 
                                 
                     for x in range(NumClases):
                        Producto_P_indice_clase[x]=1.0
                                       
                     z=-1
                     for linea2  in lineadelTrain:
                            z=z+1
                            if z==NumCampos:
                                break
                  
                            if z==0:
                                if lineadelTrain[0] == "M":
                                    ValorTrain=0.0
                                else:
                                        if lineadelTrain[0] == "F":
                                            ValorTrain=1.0
                                        else:
                                            if lineadelTrain[0] == "I":
                                              ValorTrain=2.0
                            else:
                                ValorTrain =float(lineadelTrain[z])
                            ValorTrain = ValorTrain - Min[z]
                            Maximo = Max[z]
                            Maximo = Maximo - Min[z]
                            indice =(int) (((TopeMemoria - 2.0) * ValorTrain)/ Maximo)
                            if ( (indice > (TopeMemoria-2)) or  (indice < 0)):	
                               print("index overfloed=" + str( indice) + " in the field="+ str(z-1) )
                               indice=TopeMemoria
                                               						
                        # Recuperamos las frecuencias
                            for x in range(NumClases):
                                Valor_clase[x]=TabVotos[z,indice,x]
                           
                            for x in range(NumClases):
                                  P_indice_clase[x]  =  Valor_clase[x]/ContClase[x]
                           
                            for x in range(NumClases):
                                  Producto_P_indice_clase[x]  =  Producto_P_indice_clase[x]* P_indice_clase[x]
                                                
                     SumaClases=0.0
                     for x in range(NumClases):
                         SumaClases=SumaClases+ContClase[x]
                     for x in range(NumClases):
                         Producto_P_indice_clase[x] = Producto_P_indice_clase[x] * ContClase[x] /SumaClases
                   
                     longitud=len(linea1)
                     longitud=longitud-1
                     strlinea1 = str(linea1)
                     strlinea1=strlinea1[0:longitud]
                    
                     ClasePredecida=-999999999.0
                     Producto_P_indice_claseMax=-99999999.0
                     for x in range(NumClases):
                         if (Producto_P_indice_clase[x] > Producto_P_indice_claseMax):
                             Producto_P_indice_claseMax=Producto_P_indice_clase[x]
                             ClasePredecida=float(x)
                     if (SwInicial==0):
                         SwInicial=1
                         Y=[float(Clase)]
                         X=[float(ClasePredecida)]
                    
                     else:
                         Y.append(float(Clase))
                         X.append(float(ClasePredecida))
                        
                     if (Clase==ClasePredecida):
                               strlinea1=strlinea1+",0"+"\n"
                               w.write(strlinea1)
                               TotAciertos=TotAciertos+1  
                     else:
                               strlinea1=strlinea1+",1"+"\n"
                               w.write(strlinea1)
                               TotFallos=TotFallos+1
                         
        TotHitsTest=0          
        if   SwTestTraining==1:
            print (" Total hits TRAIN = " + str(TotAciertos))
            print (" Total failures TRAIN = " + str(TotFallos))
            
        else:
            print (" Total hits TEST = " + str(TotAciertos))
            print (" Total failures TEST = " + str(TotFallos))
            TotHitsTest=TotAciertos
        
        f.close 
        w.close 
        return  Y, X, TotHitsTest
############################################################################33
# MAIN
###########################################################################333

Y_train=[float(1.0)]
X_train=[float(1.0)]
Y_test=[float(1.0)]
X_test=[float(1.0)]
W=[float(1.0)]
er_train=[float(1.0)]
er_test=[float(1.0)]
clf_tree=""
TotHits=0.0
Y_train, X_train, Y_test, X_test, TotHits = AbaloneNaiveBayes(Y_train, X_train, Y_test, X_test, W)
TotHitMax=0.0
LoopHitMax=0.0

x_range = range(10, 410, 10)

for i in x_range:    
       
        print("LOOP: "+ str(i))
        TotHits =adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        if TotHits > TotHitMax:
            TotHitMax=TotHits
            LoopHitMax=i
  
print (" ")
print (" MaAXIMUN TOTAL HITS = " + str(TotHitMax)+ " IN THE LOOP " +  str(LoopHitMax))

    