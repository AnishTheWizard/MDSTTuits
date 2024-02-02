import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

__author__ = 'Anish Chandra'

'''
The goal is to see if I can take a statistic approach to multiple linear regression 
in such a way that I can easily just input all variables and as long as the data is 
spread out, the correlation coefficient should be able to determine the 'accuracy of the data'
and as a result, determine the best 'weight/bias' for the equation.
'''

def find_regression(X, Y, r):
    return (
        (np.std(Y)/np.std(X)) * r,
        ((-np.std(Y)/np.std(X)) * r * np.mean(X)) + np.mean(Y)
    )

def execute_eq(eq, x):
    return eq[0] * x + eq[1]


def predict(eqs, X, B, headers):
    avg = 0
    for head in headers:
        avg += (B[head] * execute_eq(eqs[head], X[head]))
    return avg

def compare(pred, guess):
    print("PRED:", pred, "GUESS:", guess, "ERROR:", abs(pred-guess))


if __name__ == "__main__":

    data = pd.read_csv('states_edu.csv').dropna(subset=['ENROLL','TOTAL_REVENUE','FEDERAL_REVENUE','STATE_REVENUE','LOCAL_REVENUE','TOTAL_EXPENDITURE','INSTRUCTION_EXPENDITURE','SUPPORT_SERVICES_EXPENDITURE','OTHER_EXPENDITURE','CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G','GRADES_KG_G','GRADES_4_G','GRADES_8_G','GRADES_12_G','GRADES_1_8_G','GRADES_9_12_G','GRADES_ALL_G'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    #size = 5000
    data.to_csv('states_edit.csv')
    for i in range(20):
        print("\n------------ENTERING ITERATION---------------", i)
        size=0.35
        X = data[['ENROLL','TOTAL_REVENUE','FEDERAL_REVENUE','STATE_REVENUE','LOCAL_REVENUE','TOTAL_EXPENDITURE','INSTRUCTION_EXPENDITURE','SUPPORT_SERVICES_EXPENDITURE','OTHER_EXPENDITURE','CAPITAL_OUTLAY_EXPENDITURE','GRADES_PK_G','GRADES_KG_G','GRADES_4_G','GRADES_8_G','GRADES_12_G','GRADES_1_8_G','GRADES_9_12_G','GRADES_ALL_G']]
        #X = data[['median_income']]
        Y = data['AVG_MATH_8_SCORE']
        
        # train_X = X[:size]
        # train_Y = Y[:size]

        # test_X = X[size:]
        # test_Y = Y[size:]
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=size)

        # Create R

        r = dict()
        sumr = 0

        for head in train_X:
            r[head] = np.corrcoef(train_X[head], train_Y)[1][0]
            sumr += r[head]

        print("\nTHE R COEFS ARE", r)
        # Find B
        
        B = dict()

        for k, v in r.items():
            B[k] = v/sumr

        #print("\nTHE WEIGHTS ARE", B)

        # Find the Linear Regressions

        eq = dict()

        for head in train_X:
            eq[head] = find_regression(train_X[head], train_Y, r[head])

        #print("\nTHE EQUATIONS ARE", eq)

        #plt.plot(train_X['median_income'], train_Y, 'o')

        #lin_space = np.linspace(0, 14, 140)

        #plt.plot(lin_space, execute_eq(eq['median_income'], lin_space))



        # Make Predictions
            
        pred_y = list()

        err = list()


        for i in range(len(data.index)-len(train_Y)):
            pred_y.append(predict(eq, test_X.iloc(0)[i], B, test_X.columns))
            #compare(pred_y[i], test_Y[i+15000])
            err.append(abs(pred_y[i] - test_Y.iloc(0)[0]))

        print("\nFOR SIZE=", size,"THE ERROR STATS: MEAN=",np.mean(err), "STD=",np.std(err))

        #plt.show()