import pandas as pd
#from matplotlib import pyplot as plt
import random
from numpy import *
import operator
import numbers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import datetime
import time


### The function making up missing values in Continuous or Categorical variable
def MakeupMissing(df,col,type,method):
    '''
    :param df: dataset containing columns with missing value
    :param col: columns with missing value
    :param type: the type of the column, should be Continuous or Categorical
    :return: the made up columns
    '''
    #Take the sample with non-missing value in col
    validDf = df.loc[df[col] == df[col]][[col]]
    if validDf.shape[0] == df.shape[0]:
        return 'There is no missing value in {}'.format(col)

    #copy the original value from col to protect the original dataframe
    missingList = [i for i in df[col]]
    if type == 'Continuous':
        if method not in ['Mean','Random']:
            return 'Please specify the correct treatment method for missing continuous variable!'
        #get the descriptive statistics of col
        descStats = validDf[col].describe()
        mu = descStats['mean']
        std = descStats['std']
        maxVal = descStats['max']
        #detect the extreme value using 3-sigma method
        if maxVal > mu+3*std:
            for i in list(validDf.index):
                if validDf.loc[i][col] > mu+3*std:
                    #decrease the extreme value to normal level
                    validDf.loc[i][col] = mu + 3 * std
            #re-calculate the mean based on cleaned data
            mu = validDf[col].describe()['mean']
        for i in range(df.shape[0]):
            if df.loc[i][col] != df.loc[i][col]:
                #use the mean or sampled data to replace the missing value
                if method == 'Mean':
                    missingList[i] = mu
                elif method == 'Random':
                    missingList[i] = random.sample(validDf[col],1)[0]
    elif type == 'Categorical':
        if method not in ['Mode', 'Random']:
            return 'Please specify the correct treatment method for missing categorical variable!'
        #calculate the probability of each type of the categorical variable
        freqDict = {}
        recdNum = validDf.shape[0]
        for v in set(validDf[col]):
            vDf = validDf.loc[validDf[col] == v]
            freqDict[v] = vDf.shape[0] * 1.0 / recdNum
        #find the category with highest probability
        modeVal = max(freqDict.items(), key=lambda x: x[1])[0]
        freqTuple = freqDict.items()
        # cumulative sum of each category
        freqList = [0]+[i[1] for i in freqTuple]
        freqCumsum = cumsum(freqList)
        for i in range(df.shape[0]):
            if df.loc[i][col] != df.loc[i][col]:
                if method == 'Mode':
                    missingList[i] = modeVal
                if method == 'Random':
                    #determine the sampled category using unifor distributed random variable
                    a = random.random(1)
                    position = [k+1 for k in range(len(freqCumsum)-1) if freqCumsum[k]<a<=freqCumsum[k+1]][0]
                    missingList[i] = freqTuple[position-1][0]
    print 'The missing value in {0} has been made up with the mothod of {1}'.format(col, method)
    return missingList

### Use numerical representative for ategorical variable
def Encoder(df, col, target):
    '''
    :param df: the dataset containing categorical variable
    :param col: the name of categorical variabel
    :param target: class, with value 1 or 0
    :return: the numerical encoding for categorical variable
    '''
    encoder = {}
    for v in set(df[col]):
        if v == v:
            subDf = df[df[col] == v]
        else:
            xList = list(df[col])
            nanInd = [i for i in range(len(xList)) if xList[i] != xList[i]]
            subDf = df.loc[nanInd]
        encoder[v] = sum(subDf[target])*1.0/subDf.shape[0]
    newCol = [encoder[i] for i in df[col]]
    return newCol

### convert the date variable into the days
def Date2Days(df, dateCol, base):
    '''
    :param df: the dataset containing date variable in the format of 2017/1/1
    :param date: the column of date
    :param base: the base date used in calculating day gap
    :return: the days gap
    '''
    base2 = time.strptime(base,'%Y/%m/%d')
    base3 = datetime.datetime(base2[0],base2[1],base2[2])
    date1 = [time.strptime(i,'%Y/%m/%d') for i in df[dateCol]]
    date2 = [datetime.datetime(i[0],i[1],i[2]) for i in date1]
    daysGap = [(date2[i] - base3).days for i in range(len(date2))]
    return daysGap


### Calculate the ratio between two variables
def ColumnDivide(df, colNumerator, colDenominator):
    '''
    :param df: the dataframe containing variable x & y
    :param colNumerator: the numerator variable x
    :param colDenominator: the denominator variable y
    :return: x/y
    '''
    N = df.shape[0]
    rate = [0]*N
    xNum = list(df[colNumerator])
    xDenom = list(df[colDenominator])
    for i in range(N):
        #if the denominator is non-zero, work out the ratio
        if xDenom[i]>0:
            rate[i] = xNum[i]*1.0/xDenom[i]
        # if the denominator is zero, assign 0 to the ratio
        else:
            rate[i] = 0
    return rate


bankChurn = pd.read_csv(path+'/数据/bank attrition/bankChurn.csv',header=0)
externalData = pd.read_csv(path+'/数据/bank attrition/ExternalData.csv',header = 0)
#merge two dataframes
AllData = pd.merge(bankChurn,externalData,on='CUST_ID')


modelData = AllData.copy()
#convert date to days, using minimum date 1999/1/1 as the base to calculate the gap
modelData['days_from_open'] = Date2Days(modelData, 'open_date','1999/1/1')
del modelData['open_date']
indepCols = list(modelData.columns)
indepCols.remove('CHURN_CUST_IND')
indepCols.remove('CUST_ID')

except_var = []
for var in indepCols:
    try:
        x0 = list(set(modelData[var]))
        if var == 'forgntvl':  #something wrong with forgntvl, and I don't know how to deal with it so use this fool method~~~
            x00 = [nan]
            [x00.append(i) for i in x0 if i not in x00 and i==i]
            x0 = x00
        if len(x0) == 1:
            print 'Remove the constant column {}'.format(var)
            indepCols.remove(var)
            continue
        x = [i for i in x0 if i==i]   #we need to eliminate the noise, which is nan type
        if isinstance(x[0],numbers.Real) and len(x)>4:
            if nan in x0:
                print 'nan is found in column {}, so we need to make up the missing value'.format(var)
                modelData[var] = MakeupMissing(modelData,var,'Contiunous','Random')
        else:
            #for categorical variable, at this moment we do not makeup the missing value. Instead we think the missing as a special type
            #if nan in x0:
                #print 'nan is found in column {}, so we need to make up the missing value'.format(var)
                #modelData[var] = MakeupMissing(modelData, var, 'Categorical', 'Random')
            print 'Encode {} using numerical representative'.format(var)
            modelData[var] = Encoder(modelData, var, 'CHURN_CUST_IND')
    except:
        print "something is wrong with {}".format(var)
        except_var.append(var)
        continue

modelData['AVG_LOCAL_CUR_TRANS_TX_AMT'] = ColumnDivide(modelData, 'LOCAL_CUR_TRANS_TX_AMT','LOCAL_CUR_TRANS_TX_NUM')
modelData['AVG_LOCAL_CUR_LASTSAV_TX_AMT'] = ColumnDivide(modelData, 'LOCAL_CUR_LASTSAV_TX_AMT','LOCAL_CUR_LASTSAV_TX_NUM')


#### 1: creating features : max of all
maxValueFeatures = ['LOCAL_CUR_SAV_SLOPE','LOCAL_BELONEYR_FF_SLOPE','LOCAL_OVEONEYR_FF_SLOPE','LOCAL_SAV_SLOPE','SAV_SLOPE']
modelData['volatilityMax']= modelData[maxValueFeatures].apply(max, axis =1)


#### 2: deleting features : some features are coupling so we need to delete the redundant
#本币活期月日均余额占比 = 1 - 本币定期月日均余额占比
del modelData['LOCAL_CUR_MON_AVG_BAL_PROP']

#资产当前总余额 ＝ 本币储蓄当前总余额 ＋ 外币储蓄当前总余额， if we use the regression model, we cannot include all the three in the model,


#### 3: sum up features: some features can be summed up to work out a total number
sumupCols0 = ['LOCAL_CUR_MON_AVG_BAL','LOCAL_FIX_MON_AVG_BAL']
sumupCols1 = ['LOCAL_CUR_WITHDRAW_TX_NUM','LOCAL_FIX_WITHDRAW_TX_NUM']
sumupCols2 = ['LOCAL_CUR_WITHDRAW_TX_AMT','LOCAL_FIX_WITHDRAW_TX_AMT']
sumupCols3 = ['COUNTER_NOT_ACCT_TX_NUM','COUNTER_ACCT_TX_NUM']
sumupCols4 = ['ATM_ALL_TX_NUM','COUNTER_ALL_TX_NUM']
sumupCols5 = ['ATM_ACCT_TX_NUM','COUNTER_ACCT_TX_NUM']
sumupCols6 = ['ATM_ACCT_TX_AMT','COUNTER_ACCT_TX_AMT']
sumupCols7 = ['ATM_NOT_ACCT_TX_NUM','COUNTER_NOT_ACCT_TX_NUM']

modelData['TOTAL_LOCAL_MON_AVG_BAL'] = modelData[sumupCols0].apply(sum, axis = 1)
modelData['TOTAL_WITHDRAW_TX_NUM'] = modelData[sumupCols1].apply(sum, axis = 1)
modelData['TOTAL_WITHDRAW_TX_AMT'] = modelData[sumupCols2].apply(sum, axis = 1)
modelData['TOTAL_COUNTER_TX_NUM'] = modelData[sumupCols3].apply(sum, axis = 1)
modelData['TOTAL_ALL_TX_NUM'] = modelData[sumupCols4].apply(sum, axis = 1)
modelData['TOTAL_ACCT_TX_NUM'] = modelData[sumupCols5].apply(sum, axis = 1)
modelData['TOTAL_ACCT_TX_AMT'] = modelData[sumupCols6].apply(sum, axis = 1)
modelData['TOTAL_NOT_ACCT_TX_NUM'] = modelData[sumupCols7].apply(sum, axis = 1)


### creating features 3: ratio
numeratorCols = ['LOCAL_SAV_CUR_ALL_BAL','SAV_CUR_ALL_BAL','ASSET_CUR_ALL_BAL','LOCAL_CUR_WITHDRAW_TX_NUM','LOCAL_CUR_WITHDRAW_TX_AMT','COUNTER_NOT_ACCT_TX_NUM',
                 'ATM_ALL_TX_NUM','ATM_ACCT_TX_AMT','ATM_NOT_ACCT_TX_NUM']
denominatorCols = ['LOCAL_SAV_MON_AVG_BAL','SAV_MON_AVG_BAL','ASSET_MON_AVG_BAL','TOTAL_WITHDRAW_TX_NUM','TOTAL_WITHDRAW_TX_AMT','TOTAL_COUNTER_TX_NUM',
                   'TOTAL_ACCT_TX_NUM','TOTAL_ACCT_TX_AMT','TOTAL_NOT_ACCT_TX_NUM']

newColName = ["RATIO_"+str(i) for i in range(len(numeratorCols))]
for i in range(len(numeratorCols)):
    modelData[newColName[i]] = ColumnDivide(modelData, numeratorCols[i], denominatorCols[i])