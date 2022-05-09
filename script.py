import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

url_dataset = 'https://raw.githubusercontent.com/mk-gurucharan/Classification/master/IrisDataset.csv'
dataset= pd.read_csv(url_dataset)

def getDefaultFeaturesTitles():
    return ['sepal_length','sepal_width','petal_length','petal_width']

def showDatasetPyplot():
    datasetPyplot = sns.pairplot(dataset, hue="species")
    st.caption("Semelhanças e diferenças entre as espécies:")
    st.pyplot(datasetPyplot)

def getMaxSepalLength():
    return float(max(dataset.iloc[:,0].values))

def getMinSepalLength():
    return float(min(dataset.iloc[:,0].values))

def getMaxSepalWidth():
    return float(max(dataset.iloc[:,1].values))

def getMinSepalWidth():
    return float(min(dataset.iloc[:,1].values))

def getMaxPetalLength():
    return float(max(dataset.iloc[:,2].values))

def getMinPetalLength():
    return float(min(dataset.iloc[:,2].values))

def getMaxPetalWidth():
    return float(max(dataset.iloc[:,3].values))

def getMinPetalWidth():
    return float(min(dataset.iloc[:,3].values))

def standardizeFeatureTitle(featureTitle):
    return (featureTitle.lower()).replace(" ","_")


# O streamlit atualiza a página a cada mudança de valor dos componentes. Dessa forma, cada atualização sorteava um novo conjunto de
# dados de treino e teste, e portanto, modificava também a acurácia do modelo de classificação. 
# Por esses motivos, foi necessário estabelecer uma regra, onde os conjuntos de dados de treino e teste só mudam, caso o tamanho da
# base de teste seja alterado. 
@st.cache(suppress_st_warning=True)
def getData(datasetColumns, baseSize):
    irisFeatures = dataset.loc[:,datasetColumns]
    species = dataset['species']
    trainingData, testData, trainingResults, testResults = train_test_split(irisFeatures, species, test_size = baseSize)

    return trainingData, testData, trainingResults, testResults


def getValuesOfDefaultSidebar():
    sepalLengthInput = st.sidebar.slider(label ="Sepal Length",  min_value=getMinSepalLength(), max_value=getMaxSepalLength())
    sepalWidthInput = st.sidebar.slider(label ="Sepal Width",  min_value=getMinSepalWidth(), max_value=getMaxSepalWidth())
    petalLengthInput = st.sidebar.slider(label ="Petal Length",  min_value=getMinPetalLength(), max_value=getMaxPetalLength())
    petalWidthInput = st.sidebar.slider(label ="Petal Width",  min_value=getMinPetalWidth(), max_value=getMaxPetalWidth())
    
    return sepalLengthInput, sepalWidthInput, petalLengthInput, petalWidthInput


def getFirstFeatureInSVM():
    selectOptions = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    return st.sidebar.selectbox("Selecione a primeira feature:", selectOptions)


def getSecondFeatureInSVM(firstFeature):
    selectOptions = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    if firstFeature in selectOptions:
        selectOptions.remove(firstFeature)

    return st.sidebar.selectbox("Selecione a segunda feature:", selectOptions)


def getValuesOfSVMSidebar(firstFeature, secondFeature):
    firstFeatureInput = 0
    secondFeatureInput = 0

    if(firstFeature == 'Sepal Length'): 
        firstFeatureInput = st.sidebar.slider(label ="Sepal Length",  min_value=getMinSepalLength(), max_value=getMaxSepalLength())
    elif(firstFeature == 'Sepal Width'): 
        firstFeatureInput = st.sidebar.slider(label ="Sepal Width",  min_value=getMinSepalWidth(), max_value=getMaxSepalWidth())
    elif(firstFeature == 'Petal Length'): 
        firstFeatureInput = st.sidebar.slider(label ="Petal Length",  min_value=getMinPetalLength(), max_value=getMaxPetalLength())
    elif(firstFeature == 'Petal Width'): 
        firstFeatureInput = st.sidebar.slider(label ="Petal Width",  min_value=getMinPetalWidth(), max_value=getMaxPetalWidth())
    
    if(secondFeature == 'Sepal Length'): 
        secondFeatureInput = st.sidebar.slider(label ="Sepal Length",  min_value=getMinSepalLength(), max_value=getMaxSepalLength())
    elif(secondFeature == 'Sepal Width'): 
        secondFeatureInput = st.sidebar.slider(label ="Sepal Width",  min_value=getMinSepalWidth(), max_value=getMaxSepalWidth())
    elif(secondFeature == 'Petal Length'): 
        secondFeatureInput = st.sidebar.slider(label ="Petal Length",min_value=getMinPetalLength(), max_value=getMaxPetalLength())
    elif(secondFeature == 'Petal Width'): 
        secondFeatureInput = st.sidebar.slider(label ="Petal Width",  min_value=getMinPetalWidth(), max_value=getMaxPetalWidth())
    
    return firstFeatureInput, secondFeatureInput


def standardizeFeatureToPredict(sepalLength, sepalWidth, petalLength, petalWidth):
    features = {
     "sepal_length": sepalLength,
     "sepal_width": sepalWidth,
     "petal_length": petalLength,
     "petal_width": petalWidth
    }

    return pd.DataFrame(features, index=[0])


def standardizeFeatureToPredictInSVM(firstFeatureTitle, secondFeatureTitle, firstFeatureInput, secondFeatureInput):
    firstFeatureTitle = standardizeFeatureTitle(firstFeatureTitle)   
    secondFeatureTitle = standardizeFeatureTitle(secondFeatureTitle)   

    features = {
     firstFeatureTitle: firstFeatureInput,
     secondFeatureTitle: secondFeatureInput,
    }

    return pd.DataFrame(features, index=[0])


def classifyWithNaiveBayes(baseSize):
    trainingData, testData, trainingResults, testResults = getData(getDefaultFeaturesTitles(), baseSize)
    sepalLength, sepalWidth, petalLength, petalWidth = getValuesOfDefaultSidebar()
    featuresToPredict = standardizeFeatureToPredict(sepalLength, sepalWidth, petalLength, petalWidth)

    classifier = GaussianNB()
    classifier.fit(trainingData, trainingResults)

    testPrediction = classifier.predict(testData) 
    accuracy = accuracy_score(testResults, testPrediction) * 100 

    classification = classifier.predict(featuresToPredict)
    classificationProbability = classifier.predict_proba(featuresToPredict) 

    st.title("Classificação com Naive Bayes")
    st.caption("Resultado da classificação:")
    st.write(classification[0])
    st.caption("Acurácia:")
    st.write(round(accuracy,1) ,'%')
    st.caption("Probabilidade de cada classe:")
    classesProbabilityTable  = pd.DataFrame(classificationProbability, columns=('setosa', 'versicolor', 'virginica'))
    st.table(classesProbabilityTable)


def classifyWithDecisionTree(baseSize):
    trainingData, testData, trainingResults, testResults = getData(getDefaultFeaturesTitles(), baseSize)
    sepalLength, sepalWidth, petalLength, petalWidth = getValuesOfDefaultSidebar()
    featuresToPredict = standardizeFeatureToPredict(sepalLength, sepalWidth, petalLength, petalWidth)

    classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    classifier.fit(trainingData, trainingResults)

    testPrediction = classifier.predict(testData) 
    accuracy = accuracy_score(testResults, testPrediction) * 100 

    classification = classifier.predict(featuresToPredict)
    classificationProbability = classifier.predict_proba(featuresToPredict) 

    st.title("Classificação com Árvore de Decisão")
    st.caption("Resultado da classificação:")
    st.write(classification[0])
    st.caption("Acurácia:")
    st.write(round(accuracy,1) ,'%')
    st.caption("Probabilidade de cada classe:")
    classesProbabilityTable  = pd.DataFrame(classificationProbability, columns=('setosa', 'versicolor', 'virginica'))
    st.table(classesProbabilityTable)


def classifyWithLogisticRegression(baseSize):
    trainingData, testData, trainingResults, testResults = getData(getDefaultFeaturesTitles(), baseSize)
    sepalLength, sepalWidth, petalLength, petalWidth = getValuesOfDefaultSidebar()
    featuresToPredict = standardizeFeatureToPredict(sepalLength, sepalWidth, petalLength, petalWidth)

    classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    classifier.fit(trainingData, trainingResults)

    testPrediction = classifier.predict(testData) 
    accuracy = accuracy_score(testResults, testPrediction) * 100 

    classification = classifier.predict(featuresToPredict)
    classificationProbability = classifier.predict_proba(featuresToPredict) 

    st.title("Classificação com Regressão Logística")
    st.caption("Resultado da classificação:")
    st.write(classification[0])
    st.caption("Acurácia:")
    st.write(round(accuracy,1) ,'%')
    st.caption("Probabilidade de cada classe:")
    classesProbabilityTable  = pd.DataFrame(classificationProbability, columns=('setosa', 'versicolor', 'virginica'))
    st.table(classesProbabilityTable)


def classifyWithSVM(baseSize, firstFeature, secondFeature):
    firstFeatureInput, secondFeatureInput = getValuesOfSVMSidebar(firstFeature, secondFeature)
    featuresToPredict = standardizeFeatureToPredictInSVM(firstFeature, secondFeature, firstFeatureInput, secondFeatureInput)

    featuresTitle = [standardizeFeatureTitle(firstFeature),standardizeFeatureTitle(secondFeature)]
    trainingData, testData, trainingResults, testResults = getData(featuresTitle, baseSize)
    
    classifier = SVC(kernel='linear')
    classifier.fit(trainingData, trainingResults)

    testPrediction = classifier.predict(testData) 
    accuracy = accuracy_score(testResults, testPrediction) * 100 

    classification = classifier.predict(featuresToPredict)

    st.title("Classificação com SVM")
    showDatasetPyplot()
    st.caption("Resultado da classificação:")
    st.write(classification[0])
    st.caption("Acurácia:")
    st.write(round(accuracy,1) ,'%')


selectedOption = st.sidebar.selectbox("Selecione um algoritmo de classificação", ("Naive Bayes", "Árvore de Decisão", "Regressão Logística", "Support Vector Machine (SVM)"))
baseSizeInput = st.sidebar.slider(label = 'Tamanho da base de teste (%):', min_value=5, max_value=95, value=20, step=1)

baseSize = baseSizeInput/100

if selectedOption == "Naive Bayes":
    classifyWithNaiveBayes(baseSize)

if selectedOption == "Árvore de Decisão":
    classifyWithDecisionTree(baseSize)

if selectedOption == "Regressão Logística":
    classifyWithLogisticRegression(baseSize)

if selectedOption == "Support Vector Machine (SVM)":
    firstFeature = getFirstFeatureInSVM()
    secondFeature = getSecondFeatureInSVM(firstFeature)
    classifyWithSVM(baseSize, firstFeature, secondFeature)