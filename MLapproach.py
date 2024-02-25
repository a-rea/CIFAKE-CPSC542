import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# function to calculate statistics from an image
def calculate_image_stats(image):
    # convert the image to grayscale
    grayscale_image = image.convert('L')
    
    # grey scale image to array to calculate statistics
    image_array = np.array(grayscale_image)
    
    # gather image statisttics as features for a dataframe
    mean_intensity = np.mean(image_array)
    median_intensity = np.median(image_array)
    std_intensity = np.std(image_array)
    min_intensity = np.min(image_array)
    max_intensity = np.max(image_array)
    intensity_ratio = max_intensity / min_intensity
    max_variance = max_intensity - min_intensity
    IQRIntensity = np.percentile(image_array, 75) - np.percentile(image_array, 25)

    return {
        'Mean Intensity': mean_intensity,
        'Median Intensity': median_intensity,
        'Std Intensity': std_intensity,
        'Min Intensity': min_intensity,
        'Max Intensity': max_intensity,
        'Intensity Ratio': intensity_ratio,
        'Max Variance': max_variance,
        'IQR Intensity': IQRIntensity
    }

# extract statistics from images in a directory
def extract_stats_from_images(directory):
    image_stats = []
    
    # iterate through each image file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            filepath = os.path.join(directory, filename)
            try:
                # open the image using Pillow
                image = Image.open(filepath)
                # calculate statistics for the image
                stats = calculate_image_stats(image)
                # get the class of each image
                stats['label'] = directory.split('/')[-1]
                image_stats.append(stats)
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
    
    # convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(image_stats)
    return df


def construct_dataframes():
    # string paths to the directories containing the images
    train_fake_stats = 'train/FAKE'
    train_real_stats = 'train/REAL'
    test_fake_stats = 'test/FAKE'
    test_real_stats = 'test/REAL'

    # extract statistics from images in the directory
    train_fake_img_stats = extract_stats_from_images(train_fake_stats)
    train_real_img_stats = extract_stats_from_images(train_real_stats)
    test_fake_img_stats = extract_stats_from_images(test_fake_stats)
    test_real_img_stats = extract_stats_from_images(test_real_stats)

    # combine all the img stats into one dataframe
    train_image_stats_df = pd.concat([train_fake_img_stats, train_real_img_stats])
    test_image_stats_df = pd.concat([test_fake_img_stats, test_real_img_stats]) 

    # ensure there are no infinities or NaN values
    train_image_stats_df = train_image_stats_df.replace([np.inf, -np.inf], np.nan).dropna()
    test_image_stats_df = test_image_stats_df.replace([np.inf, -np.inf], np.nan).dropna()

    # encode the label column to be 0 for FAKE and 1 for REAL
    train_image_stats_df['label'] = train_image_stats_df['label'].map({'FAKE': 0, 'REAL': 1})
    test_image_stats_df['label'] = test_image_stats_df['label'].map({'FAKE': 0, 'REAL': 1})


    # save the dataframe to a csv file
    train_image_stats_df.to_csv('image_stats_train.csv', index=False)
    test_image_stats_df.to_csv('image_stats_test.csv', index=False)

    return train_image_stats_df, test_image_stats_df



def exploratory_data_analysis():
    train_data = pd.read_csv('image_stats_train.csv')

    # Exploratory Data Analysis

    # plot and print the class distribution with grey grid background and reddish orange bars
    plt.figure()
    sns.countplot(x='label', data=train_data)
    plt.title("Class Distribution")
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.savefig('RF_images/class_distribution.png')

    corr = train_data.corr()
    plt.figure(figsize=(12, 12))
    #plot heatmap of the correlation matrix
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Image Statistics Correlation Matrix")
    # show the plot
    plt.show()
    plt.savefig('RF_images/correlation_matrix.png')


    # plot histograms of the image statistics
    plt.figure()
    train_data.hist(figsize=(12, 12))
    plt.title("Image Statistics Histogram")
    plt.savefig('RF_images/train_histogram.png')


def hyperparameter_tuning(train_image_stats_df, flag):
    if flag:
        # define the features and the target
        X_train = train_image_stats_df.drop(columns=['label'])
        y_train = train_image_stats_df['label']

        param_grid = { 
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'] 
        }   
        
        grid_search = GridSearchCV(RandomForestClassifier(), 
                            param_grid=param_grid) 
        grid_search.fit(X_train, y_train) 
        print(grid_search.best_estimator_)
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        return grid_search.best_estimator_
    


# function to create random forest model that uses the image_stats_df to predict if an image is real or fake
def define_RF(train_image_stats_df, test_image_stats_df):

    # define the features and the target
    X_train = train_image_stats_df.drop(columns=['label'])
    y_train = train_image_stats_df['label']

    X_test = test_image_stats_df.drop(columns=['label'])
    y_test = test_image_stats_df['label']
    
    # define the model
    model = RandomForestClassifier()
    
    # train the model
    model.fit(X_train, y_train)
    
    # evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)


    print(f"Train score: {train_score:.2f}")
    print(f"Test score: {test_score:.2f}")

    y_pred_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # create ROC curve with white background and grid
    plt.figure()
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid()
    plt.legend(loc=4)
    plt.savefig("RF_images/roc_curve.png")

    #confusion matrix of the model
    y_pred = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    plt.figure()
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=model.classes_)
    disp.plot()
    plt.show()
    plt.savefig("RF_images/RF_confusion_matrix.png")
    
    return model


if __name__ == "__main__":
    # define a random forest model

    #train_data, test_data = construct_dataframes()
    #model = hyperparameter_tuning(train_data, False)
    exploratory_data_analysis()
    #model = define_RF(train_data, test_data)



    

