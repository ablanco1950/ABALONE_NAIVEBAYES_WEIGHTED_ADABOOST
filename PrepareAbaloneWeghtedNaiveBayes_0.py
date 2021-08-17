# -*- coding: utf-8 -*-
"""
Created on Sun Aug 8 17:28:30 2021

@author: Alfonso Blanco García
"""
import numpy as np
########################################################################################
# GETS THE FREQUENCIES OF EACH INDEX OF EACH FIELD; CALL THE MODULE THAT GETS
# PREDICTED CLASSES BY APPLYING NAIVE BAYES, RECEIVE A W TABLE WITH THE WEIGHTS
# THAT CALCULATES THE EXTERNAL ADABOOST MODULE, RETURNS THE ARRAYS WITH THE PREDICTED CLASSES
# AND REAL
###################################################################3333333333333333

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
        
        # Got from https://stackoverflow.com/questions/15448594/how-to-add-elements-to-3-dimensional-array-in-python
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
                FactorPri=W(Cont)
                
             ###################################################################3
            # We are dealing with records classified in 29 classes that are 
            # summarized in 3, but that still suppose a case of multiclass
            #classification (see specifications described in the 
            # abalone.names.txt file on the abalone download page)
            ###########################################################333333#3#
            ClaseLeida=float(lineadelTrain[8])
            
            Clase=0    
        	
            if (ClaseLeida > 10.0): 
                Clase=2
            else:
                if (ClaseLeida > 8.0): 
                		Clase=1
          
            ContClase[Clase]=ContClase[Clase] + 1
                       
            FactorPri=1.0
           
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
                    print("index overflowed=" + str( indice) + " en el campo="+ str(z-1) )
                    indice=TopeMemoria
                Wvalor=0.0
             
                Wvalor= TabVotos[z,TopeMemoria-1,Clase]
                Wvalor= Wvalor + 1
                TabVotos[z,TopeMemoria-1,Clase]=Wvalor 
                Wvalor= TabVotos[z,indice,Clase]
                Wvalor=Wvalor+FactorPri
                TabVotos[z,indice,Clase]=Wvalor 
       
        f.close()
        
               
        Start=3134
        End = 4177
        SwTestTraining=2
        Y_test, X_test = AbaloneNaiveBayes_train_test (TabVotos, ContClase,Max,Min, Start, End, SwTestTraining, Y_train, X_Train, Y_test, X_test, W)
        Start=0
        End = 3133
        SwTestTraining=1
        Y_train, X_train = AbaloneNaiveBayes_train_test (TabVotos, ContClase,Max,Min, Start, End, SwTestTraining, Y_train, X_Train, Y_test, X_test, W)
        return Y_train, X_train, Y_test, X_test
    
    
    
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
        with open("C:\AbaloneWeighted_1.txt","w") as  w:
        
                for linea1 in f:
               
                     lineadelTrain =linea1.split(",")
                     Conta = Conta + 1
                 
                     if Conta < Start:
                         continue
                     if Conta > End:
                         break
                 
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
                               print("indice desbordado=" + str( indice) + " en el campo="+ str(z-1) )
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
                     ClaseLeida=float(lineadelTrain[8])
                     Clase=0.0  
        	
                     if (ClaseLeida > 10): 
                         Clase=2.0
                     else:
                         if (ClaseLeida > 8.0): 
            		             Clase=1.0 
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
                         
                   
        if   SwTestTraining==1:
            print (" Total hits TRAIN = " + str(TotAciertos))
            print (" Total failures TRAIN = " + str(TotFallos))
        else:
            print (" Total hits TEST = " + str(TotAciertos))
            print (" Total failures TEST = " + str(TotFallos))
       
        f.close 
        w.close 
        return  Y, X
        
Y_train=[float(1.0)]
X_train=[float(1.0)]
Y_test=[float(1.0)]
X_test=[float(1.0)]
W=[float(1.0)]
Y_train, X_train, Y_test, X_test= AbaloneNaiveBayes(Y_train, X_train, Y_test, X_test, W)              
  
        