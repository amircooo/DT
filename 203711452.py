import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

amir = []
sonOf= []
#%%
def entropia(isSpam):
    '''
    decription: calculate entropia by the course equation
    '''
    attributes, numShows =           np.unique(isSpam, return_counts=True)
    entropia =                       np.sum([(-numShows[i] / np.sum(numShows)) * np.log2(numShows[i] / np.sum(numShows)) for i in range(len(attributes))])
    return entropia

def InfoGain(data, q_name, isSpam="spam"):
    '''
    description: will calc the entropia gain from a given q
    1. entropy of all the date
    2. split the node data by q_name
    3. calc entopy of the q_name
    '''
    entropiaAll =                 entropia(data[isSpam])   
    values, numShows =            np.unique(data[q_name], return_counts=True)
    qEntropia =                   np.sum( [(numShows[i] / np.sum(numShows)) * entropia(data.where(data[q_name] == values[i]).dropna()[isSpam])for i in range(len(values))])
    Information_Gain =            entropiaAll - qEntropia
    return Information_Gain

def buildTree(k):
    ''' 
    description:
    1. read and arragne date set where each col-name represents the attribute   
    2. create bundries by avg value of col
    3. use external splitting the data func will split by k ratio
    4. train the tree to use my data
    5. print the tree
    6. calculte eror proportion
    '''
    spamBase =                             pd.read_csv('spambase.data', names= ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28','Q29','Q30','Q31','Q32','Q33','Q34','Q35','Q36','Q37','Q38','Q39','Q40','Q41','Q42','Q43','Q44','Q45','Q46','Q47','Q48','Q49', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54','Q55', 'Q56','Q57','spam'], dtype=np.float64)
    spamBase, Bundries =                   setBundries(spamBase)
    SBdata, checkOn =                      train_test_split(spamBase, test_size= 1-k, random_state=100)
    Mytree =                               training(SBdata, SBdata, SBdata.columns[:-1])
    print(Mytree)
    printTree(amir,sonOf)
    precisionEvel(checkOn, Mytree)
    return Mytree, Bundries

def setBundries(spamBase):
    
    ''' 
    description : this func sets bundrie for each Question 
    will compere to the mean value of the question
    spam base value be assigned with 1 and 0
        
    '''
    Bundries = []
    for i in spamBase.iloc[:, :-4]:
        Bundrie =                                 spamBase[i].mean(axis=0)
        Bundries.append(Bundrie)
        spamBase.loc[spamBase[i] >= Bundrie, i] =       1
        spamBase.loc[spamBase[i] < Bundrie, i] =        0
       
        spamBase[i] =                             spamBase[i].dropna().apply(np.int64)
        pass
    Bundrie1 =                                    spamBase['Q55'].mean()
    Bundrie2 =                                    spamBase['Q56'].mean()
    Bundrie3 =                                    spamBase['Q57'].mean()
    
    
    spamBase['Q55'] =                            spamBase['Q55'] >= Bundrie1
    spamBase['Q56'] =                            spamBase['Q56'] >= Bundrie2
    spamBase['Q57'] =                            spamBase['Q57'] >= Bundrie3
    
    Bundries.append(Bundrie1)
    Bundries.append(Bundrie2)
    Bundries.append(Bundrie3)
    spamBase['Q57'] =                             spamBase['Q57'].dropna().apply(np.int64)
    spamBase['Q56'] =                             spamBase['Q56'].dropna().apply(np.int64)
    spamBase['Q55'] =                             spamBase['Q55'].dropna().apply(np.int64)
    spamBase['spam'] =                            spamBase['spam'].dropna().apply(np.int64)
    return spamBase, Bundries

def printTree(amir,sonOf):
    i=0
    length=len(amir)
    for t in range(len(amir)):
         print('node_num=[{fa}] Q_num=[{mo}] sonOf=[{so}]'.format( fa=len(amir)-i , mo=amir[length-1],so=sonOf[length-1]))   
         length=length-1
         i=i+1
#%%
def training(data, SBdata, examples, answer="spam", fatherMajority=None,father=-1):
    '''
    description: recursive function
    1. leaf node coditions
    2. set father majority 
    3. find the best question
    4. add node to tree
    5. remove the question from the options
    6. create branches by rec call
    '''
    if len(np.unique(data[answer])) <= 1:
        return np.unique(data[answer])[0]
    if len(data) == 0:
        return np.unique(SBdata[answer])[np.argmax(np.unique(SBdata[answer], return_counts=True)[1])]
    if len(examples) == 0:
        return fatherMajority
  
    else:
        fatherMajority =                   np.unique(data[answer])[np.argmax(np.unique(data[answer], return_counts=True)[1])]
        question=                          getQuestion(data, examples, answer)
        amir.append( question)
        sonOf.append(father)
        Mytree =                           {question: {}}
        examples =                         [ex for ex in examples if ex != question]
        for outcome in np.unique(data[question]):
            outcome =                      outcome
            brachData =                    data.where(data[question] == outcome).dropna()
            branch =                       training(brachData, SBdata, examples, answer, fatherMajority,father+1)
            Mytree[question][outcome] =    branch
        return Mytree

       
def precisionEvel(data, Mytree):
    '''
    description:
    1. create examples dictionary, looks like---> ‘records’ : list like [{column -> value}, … , {column -> value}]    
    2. create data frame that stores all the predicted values of all the tree
    3. sum the amount of correct answers and return percentage
    '''
    exampels =                   data.iloc[:, :-1].to_dict(orient="records")
    predicted=                   createDataFrame(data, Mytree,exampels)
    precision =                  (np.sum(predicted["predicted"].values == data["spam"].values) / len(data)) * 100
    error=                       (100-precision)
    print                        ('error percent is: ',error, '%')
    return precision

def createDataFrame(data, Mytree,exampels):
    '''
    description:
    '''   
    predicted =                          pd.DataFrame(columns=["predicted"])
    for i in range(len(data)):
       predicted.loc[i, "predicted"] =   prediction(exampels[i], Mytree)
    return predicted


def getQuestion(data, examples, answer):
        '''
        description:
        '''        
        item_values =            [InfoGain(data, example, answer) for example in examples]  
        questionInd =            np.argmax(item_values)
        question =               examples[questionInd]
        return question
    
def prediction(q, Mytree, default=1):
    '''
    description: 
    1. run down the tree and for every heder check if it's in  myTree (if not return defualt= majority value of the "spam" atribute)
    2. if we arrived to a leaf node return the prediction value
    '''
    for heder in list(q.keys()):
        if heder in list(Mytree.keys()):
           try:
              curr = Mytree[heder][q[heder]]
           except:
              return default
           if isinstance(curr, dict):
              return prediction(q, curr)
           else:
              return curr
#%%   
def treeError(k, Mytree):
    '''
    description:
    1. read and arragne date set where each col-name represents the attribute   
    2. create bundries by avg value of col
    3. use external splitting func will split to k parts 
    4. train the tree on k-1 part where the 1 is the checkOn data
    6. calculte eror proportion for all parts and choose the best one
    '''    
    spamBase =                 pd.read_csv('spambase.data', names=['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28','Q29','Q30','Q31','Q32','Q33','Q34','Q35','Q36','Q37','Q38','Q39','Q40','Q41','Q42','Q43','Q44','Q45','Q46','Q47','Q48','Q49', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54','Q55', 'Q56','Q57','spam'], dtype=np.float64)
    spamBase =                 setBundries(spamBase)[0]
    kf =                       KFold(n_splits= k, shuffle = True, random_state = 2)
    maxi=                      0
    for train, test in kf.split(spamBase):
        test =                    spamBase.iloc[test]
        precision =               precisionEvel(test, Mytree)
        if maxi < precision:
            maxi = precision
    print('most precise: ', maxi, '%')
#%%    
def isThisSpam(arr, Mytree, Bundries):
    '''
    description:
    1. counvert arr values to 0 & 1
    2. create an iterator and heders
    3. transform the given e-mail to dictionary  
    4. predict the outcome   
    '''
    arr =                          intoNumbers(arr) 
    Bundries =                     intoNumbers(Bundries)
    for i in range(len(Bundries)):
        if arr[i] < Bundries[i]:
           arr[i] = 0
        else:
           arr[i] = 1
    heders =                       ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28','Q29','Q30','Q31','Q32','Q33','Q34','Q35','Q36','Q37','Q38','Q39','Q40','Q41','Q42','Q43','Q44','Q45','Q46','Q47','Q48','Q49', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54','Q55', 'Q56','Q57','spam']
    mailPath =                     dict(zip(heders, arr))    
    predict =                      prediction(mailPath, Mytree)
    if predict == 1.0:
        print('this mail is Spam!!! ')
    elif predict==0:
         print('this mail is not Spam!!! ')
    else: 
        print('span in: ', predict, '%')    
        

         
def intoNumbers(arr):
    arr = [float(i) for i in arr]
    return arr


#%%         
  
if __name__== "__main__":
    k = input('choose k value between 0 , 1: ')
    print('Building Mytree please wait 1min & 50sec, have a nice sapmFilttering day')
    Mytree, Bundries = buildTree(float(k))

    k = input('choose int k-cross validation:  ')
    treeError(int(k), Mytree)

    arr = input('test your mail:').split(",")
    isThisSpam(arr, Mytree, Bundries)
