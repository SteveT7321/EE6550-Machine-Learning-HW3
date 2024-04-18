import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import pymc3 as pm



def MLR(train_data, test_data_feature):
    train_data_feature = train_data[:, :7]
    train_data_label = train_data[:, 7]

    train_data_feature = np.c_[np.ones(train_data_feature.shape[0]), train_data_feature]
    coefficients = np.linalg.inv(train_data_feature.T.dot(train_data_feature)).dot(train_data_feature.T).dot(train_data_label)

    test_data_feature = np.c_[np.ones(test_data_feature.shape[0]), test_data_feature]
    MLR_predictions = test_data_feature.dot(coefficients)

    return MLR_predictions


def BLR(train_data, test_data_feature):
    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]

    train_X = np.c_[np.ones(train_X.shape[0]), train_X]
    X_transpose_X = train_X.T.dot(train_X)
    X_transpose_y = train_X.T.dot(train_y)
    posterior_precision = np.linalg.inv(X_transpose_X + np.eye(X_transpose_X.shape[0]))
    posterior_mean = posterior_precision.dot(X_transpose_y)

    test_X = np.c_[np.ones(test_data_feature.shape[0]), test_data_feature]
    BLR_predictions = test_X.dot(posterior_mean)

    return BLR_predictions


def MSE(data, prediction):
    mse = np.mean((prediction - data) ** 2)
    return mse


def linear_regression(X,y):
    _coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    return _coeffs

def main():
    '''Q1: Data preprocessing (Only do once)'''
    # To merge the csv file :
    calories_df = pd.read_csv('calories.csv')
    exercise_df = pd.read_csv('exercise.csv')
    exercise_df = exercise_df.drop(calories_df.columns[0], axis=1)
    calories_df = calories_df.drop(calories_df.columns[0], axis=1)
    merged_df = pd.concat([exercise_df, calories_df], axis=1)
    merged_df['Gender'] = merged_df['Gender'].replace({'male': 1, 'female': 0})
    # print(merged_df)
    merged_df.to_csv('merged_data.csv', index=False)

    total_samples = len(merged_df)
    train_size = 10500
    val_size = 1500
    test_size = 3000

    train_data = merged_df[:train_size]
    val_data = merged_df[train_size:train_size + val_size]
    test_data = merged_df[train_size + val_size:]
    # print(train_data)
    # print(val_data)
    # print(test_data)

    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)


    '''Q2: MLR and BLR'''
    train_data = pd.read_csv('train_data.csv').to_numpy()
    test_data = pd.read_csv('test_data.csv').to_numpy()
    val_data = pd.read_csv('val_data.csv').to_numpy()
    train_data_feature = train_data[:, :7]
    train_data_label = train_data[:, 7]
    test_data_feature = test_data[:, :7]
    test_data_label = test_data[:, 7]
    val_data_feature = val_data[:, :7]
    val_data_label = val_data[:, 7]

    mlr_predictions = MLR(train_data, test_data_feature)
    blr_predictions = BLR(train_data, val_data_feature)

    print("MLR Predicted Calories:")
    print(mlr_predictions)

    print("BLR Predicted Calories:")
    print(blr_predictions)


    # Calculate and print the mean squared error for MLR
    mlr_mse = MSE(test_data_label, mlr_predictions)
    print("MLR Mean Squared Error:", mlr_mse)

    blr_mse = MSE(val_data_label, blr_predictions)
    print("BLR Mean Squared Error:", blr_mse)


    '''Q3: Comparison BLR and MLR'''
    # (1) OLS
    test_df = pd.read_csv('test_data.csv')
    test_df['Intercept'] = 1
    X = test_df.loc[:, ['Intercept', 'Duration']]
    y = test_df.loc[:, 'Calories']

    coefs = linear_regression(X, y)
    xs = np.linspace(4, 31, 1000)
    ys = coefs[0] + coefs[1] * xs

    # (2) Bayesian
    with pm.Model() as linear_model:
        intercept = pm.Normal('Intercept', mu = 0, sd = 10)
        slope = pm.Normal('slope', mu = 0, sd = 10)
        sigma = pm.HalfNormal('sigma', sd = 10)
        mean = intercept + slope * X.loc[:, 'Duration']
        Y_obs = pm.Normal('Y_obs', mu = mean, sd = sigma, observed = y.values)
        step = pm.NUTS()
        linear_trace = pm.sample(1000, step)

    plt.figure(figsize=(8, 8))
    pm.plot_posterior_predictive_glm(linear_trace, samples = 100, eval=np.linspace(2, 30, 100), linewidth = 1, 
                                    color = 'red', alpha = 0.8, label = 'Bayesian Posterior Fits',
                                    lm = lambda x, sample: sample['Intercept'] + sample['slope'] * x);
    
    plt.plot(test_df['Duration'], test_df['Calories'], 'bo',label = 'observations', alpha = 0.8);
    plt.xlabel('Duration (min)', size = 18);
    plt.ylabel('Calories', size = 18); 
    plt.plot(xs, ys, 'k--', label = 'OLS Fit', linewidth = 3)
    plt.legend(prop={'size': 16})
    plt.title('Calories burned vs Duration of Exercise', size = 20);
    plt.show()


    '''Q4: Using the other regressor'''
    model = GradientBoostingRegressor()
    # model = SVR()

    model.fit(train_data_feature, train_data_label)
    gbr_predictions = model.predict(test_data_feature)
    gbr_mse = MSE(test_data_label, gbr_predictions)
    print("GBR Mean Squared Error:", gbr_mse)


if __name__ == '__main__':
    main()