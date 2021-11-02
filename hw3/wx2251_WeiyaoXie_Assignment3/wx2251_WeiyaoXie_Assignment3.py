import numpy as np
import pandas as pd
from scipy import io
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import nibabel as nib


# function to extract brain voxel area and exclude background and non-voxel area

def create_mask_image(img_data, threshold):
    # create a numpy array to store new images
    new_image_data = np.zeros(img_data.shape)
    # for each 30*30 image slice
    
    for slice_num in range(img_data.shape[2]):
        for sample_num in range(img_data.shape[3]):
            # apply a threshold to supress non-brain area to 0
            # retain the original value of the brain area
            mask_image = np.where(img_data[:,:,slice_num, sample_num] > threshold, img_data[:,:,slice_num, sample_num], 0)
            # overwrite the new image
            new_image_data[:,:,slice_num, sample_num]=mask_image
    return new_image_data



def validation(image_file, label_file):
    # load image from the fmri file
    # then convert it into numpy file
    img = nib.load('data/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii')
    img = nib.load(image_file)
    img_data = img.get_fdata()
    labels = io.loadmat('data/label.mat')['label']


    # get brain mask
    masked_image = create_mask_image(img_data, 200)

    # make sure the masked images have the sampe shape as original images
    assert(img_data.shape==masked_image.shape)
    # reshape labels and flatten images for training
    masked_image = masked_image.reshape(184,-1)
    labels = labels.reshape(184,)
    # normalize the mask images 
    masked_image = normalize(masked_image, norm='l2')

    # upsample the minority classes to have better performence
    labels=list(labels)
    masked_image = list(masked_image)
    for i in range(700):
        if labels[i]!=1:
            labels.append(labels[i])
            masked_image.append(masked_image[i])


    # flatten the masked image
    labels=np.array(labels)
    masked_image=np.array(masked_image)

    # use pca to reduce training dimension of data
    pca = PCA(n_components=90)
    pca_train = pca.fit_transform(masked_image)

    # convert the data to 0 mean and unit variance
    scaler = StandardScaler()
    pca_train=scaler.fit_transform(pca_train)

    # use kfold to get select model
    # shuffle the data to prevent imbalanced class to dominate
    pca_train, labels = shuffle(pca_train, labels, random_state=43)


    ## SVM model
    # 
    clf_svm = SVC(kernel='rbf', C=100, gamma="auto")
    svm_scores = cross_val_score(clf_svm, pca_train, labels, cv=8)
    avg_svm = sum(svm_scores)/len(svm_scores)
    print("the average validation accuracy for svm model is: {}".format(avg_svm))

    clf_gbm = GradientBoostingClassifier(n_estimators = 200, max_depth=4, random_state=43)
    gbm_scores = cross_val_score(clf_gbm, pca_train, labels, cv=8)
    avg_gbm = sum(gbm_scores)/len(gbm_scores)
    print("the average validation accuracy for gradient boosting model is: {}".format(avg_gbm))






if __name__ == "__main__":
    
    image = 'data/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii'
    labels = 'data/label.mat'
    validation(image, labels)
    
    image_retest = 'data/sub-01/ses-retest/func/sub-01_ses-retest_task-fingerfootlips_bold.nii'
    labels = 'data/label.mat'
    validation(image_retest, labels)
