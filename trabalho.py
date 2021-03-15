import numpy as np 
import pandas as pd
from pandas.io.formats.format import common_docstring 

# importando dados de treino 
df = pd.read_csv('trainReduzido.csv')


X = df.iloc[:,2:] 
Y = df['label']

# Separando dados de Treino , em treino e teste para validação do modelo
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)
# importando o modelo e treinando 
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
#avalidando modelo de treino
pred = model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print('Classification Report : \n ')

print(classification_report(y_test,pred))
print('Matriz de confusão :\n')
print(confusion_matrix(y_test,pred))
print('Acuracia : ')
print(accuracy_score(y_test,pred))






print('\n Treinando modelo compelto : \n')

model_final = SVC()
df_teste = pd.read_csv('validacao.csv')
x_teste_final = df_teste.iloc[:,1:]

## Treinando modelo com todos os dados
model_final.fit(X,Y)

#valor de y final
y_final = model_final.predict(x_teste_final)
print(y_final)

resultado_final = pd.DataFrame(y_final,columns=['Label'])
print(resultado_final)

resultado_final.to_csv('resultado_final.csv')

resultado_teste = [pred,y_test]
resultado_teste_df = pd.DataFrame({'pred':pred, 'y_test':y_test})

resultado_teste_df.to_csv('resultado_teste.csv')

