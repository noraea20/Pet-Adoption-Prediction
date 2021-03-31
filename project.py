from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd




numericData = ['Age', 'MaturitySize', 'FurLength', 'Quantity', 'Fee']



# splitting data
def split(data):
    training_data = data.dropna()

    x = training_data.iloc[:, 1:23].values
    y = training_data.iloc[:, 23].values

    trainX, testX, trainY, testY= train_test_split(x, y, test_size=0.2, random_state=0)

    return trainX, testX, trainY, testY





# preprocessing categorical and numerical data
def preprocess_data(trainX, testX, trainY, testY):
    # one hot representation of categorical data
    ids= []
    temp= list(trainX.columns)
    dropped= ['Name', 'RescuerID', 'PhotoAmt', 'VideoAmt', 'State']
    for element in temp:
        if (element not in numericData) and (element not in dropped):
            ids.append(element)


    for element in ids:
        # train
        dummies = pd.get_dummies(trainX[element], prefix=element, drop_first=False)
        trainX = pd.concat([trainX, dummies], axis=1)

        # test
        dummies = pd.get_dummies(testX[element], prefix=element, drop_first=False)
        testX = pd.concat([testX, dummies], axis=1)



    # scaling numeric data
    scaler = MinMaxScaler()

    for element in numericData:
        # train
        scaled = scaler.fit_transform(trainX[[element]])
        trainX.drop([element], axis= 'columns')
        trainX[element]= scaled

        # test
        scaled = scaler.fit_transform(testX[[element]])
        testX.drop([element], axis= 'columns')
        testX[element] = scaled



    # one hot representation for label
    # train
    dummies = pd.get_dummies(trainY['AdoptionSpeed'], prefix='AdoptionSpeed', drop_first=False)
    trainY = pd.concat([trainY, dummies], axis=1)

    # test
    dummies = pd.get_dummies(testY['AdoptionSpeed'], prefix='AdoptionSpeed', drop_first=False)
    testY = pd.concat([testY, dummies], axis=1)

    return trainX, testX, trainY, testY





data= pd.read_csv("/Users/yaraaltwaijry/Desktop/Machine project/dataset.csv")
trainX, testX, trainY, testY= split(data)

# train
trainX= pd.DataFrame(trainX, columns=['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                                      'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
                                      'RescuerID', 'VideoAmt', 'PetID', 'PhotoAmt'])
trainY= pd.DataFrame(trainY, columns=['AdoptionSpeed'])


# test
testX= pd.DataFrame(testX, columns=['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                                      'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
                                      'RescuerID', 'VideoAmt', 'PetID', 'PhotoAmt'])
testY= pd.DataFrame(testY, columns=['AdoptionSpeed'])


trainX, testX, trainY, testY= preprocess_data(trainX, testX, trainY, testY)


print('TRAIN X',trainX)
print('\n\nTEST X',trainX)
print('\n\nTRAIN Y', trainY)
print('\n\nTEST Y', testY)


#name= element + '_types'
        #encoded[element]= columns[index][0]
        #encoded[name]= labelEncoder.fit_transform(trainX[element])

        # train
        #temp= pd.DataFrame(hotEncoder.fit_transform(trainX[[element]]).toarray())
        #trainX.drop([element], axis='columns')
        #trainX[element] = temp
        #trainX= trainX.join(temp)
        #print('temp\n', temp)
        #print(hotEncoder.categories_)

        # test
        #temp = pd.DataFrame(hotEncoder.fit_transform(testX[[element]]).toarray())
        #testX.drop([element], axis='columns')
        #testX[element] = temp
        #testX = testX.join(temp)

        #index+= 1