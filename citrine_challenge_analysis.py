import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#for laptop
os.chdir('C:/Users/nts21/Documents/Citrine_challenge/challenge_data/')
#for linux workstation
os.chdir('/home/nick/Citrine_challenge/challenge_data/')

#load the data
training=pd.read_csv('training_data.csv')

#separate the output vector
output=training['stabilityVec']

#remove the square brackets from the string
output=[m.strip('[') for m in output]
output=[m.strip(']') for m in output]

#split the string on commas
output=[m.split(',') for m in output]

#convert the list of lists into integers instead of strings
temp=list(output)
output=[]
for x in temp:
    output.append(list(map(lambda y: int(float(y)), x)))

#check that the first is always a 1, uncomment to rerun
#temp=[item[0] for item in output]
#print(len(temp))
#print(sum(temp))
#those should be the same

#check that the last is always a 1, uncomment to rerun
#temp=[item[10] for item in output]
#print(len(temp))
#print(sum(temp))
#again, those should be the same

#great, we can get rid of the first and last value since they are always the same
output=[item[1:10] for item in output]

#create a new value of the sum of the stability vector
sum_output=[sum(item) for item in output]

training['stabilityVec_total']=sum_output
training['stabilityVec']=output

for n in range(0,9):
    temp=[]
    [temp.append(x[n]) for x in training['stabilityVec']]
    training['stabilityVec%s' % (n)]=temp

#Looking at atomic volume and its relation to the total
plt.plot(np.log(training['formulaB_elements_AtomicVolume']), training['stabilityVec_total'], 'bx')
plt.show()

#Look at the values to see how many large/small ones there are
training['formulaA_elements_AtomicVolume'].value_counts().sort_index()

#which ones are the ones that are so large? Does that make sense?
training['formulaA'][training['formulaA_elements_AtomicVolume']>10000].value_counts()
#It seems like noble gases are really high, which makes sense, but the difference seems larger than other groups suggest

training['stabilityVec_total'][training['formulaA_elements_AtomicVolume']>10000].value_counts()
#from this command, these groups never react, so they can be safely removed, especially if we want to predict with atomic volume

#this command tells me that only the noble gases are the ones that never react with anything
temp=training['formulaA'][training['stabilityVec_total']==0].value_counts()
for x in temp.index:
    print('Sum for element %s is ' % (x) + str(np.sum(training['stabilityVec_total'][training['formulaA']==x])))

#make a separate dataset with noble gases removed, to play with
training_nonoble=training[~training['formulaA'].isin(['Xe', 'Ne', 'Ar', 'He', 'Kr'])]
training_nonoble=training_nonoble[~training_nonoble['formulaB'].isin(['Xe', 'Ne', 'Ar', 'He', 'Kr'])]

#double check that it did what I think it did
print(set(training['formulaA'].unique()) - set(training_nonoble['formulaA'].unique()))
print(set(training['formulaB'].unique()) - set(training_nonoble['formulaB'].unique()))



#add columns that may be of use for the classifier
training['CovalentRadius_diff']=training['formulaA_elements_CovalentRadius']\
                                -training['formulaB_elements_CovalentRadius']
training['CovalentRadius_ratio']=training['formulaA_elements_CovalentRadius']/\
                                training['formulaB_elements_CovalentRadius']

training['MendeleevNumber_ratio']=training['formulaA_elements_MendeleevNumber']/\
                                  training['formulaB_elements_MendeleevNumber']
training['MendeleevNumber_diff']=training['formulaA_elements_MendeleevNumber']-\
                                 training['formulaB_elements_MendeleevNumber']

training['Electronegativity_diff']=training['formulaA_elements_Electronegativity']-\
                                   training['formulaB_elements_Electronegativity']
#you will have zeros if you don't remove noble gases
training['Electronegativity_ratio']=training['formulaA_elements_Electronegativity']/\
                                   training['formulaB_elements_Electronegativity']

training['FirstIonizationEnergy_ratio']=training['formulaA_elements_FirstIonizationEnergy']/\
                                        training['formulaB_elements_FirstIonizationEnergy']
training['FirstIonizationEnergy_diff']=training['formulaA_elements_FirstIonizationEnergy']-\
                                        training['formulaB_elements_FirstIonizationEnergy']

training['AtomicVolume_ratio']=training['formulaA_elements_AtomicVolume']/\
                               training['formulaB_elements_AtomicVolume']

training['BoilingT_ratio']=training['formulaA_elements_BoilingT']/\
                           training['formulaB_elements_BoilingT']

training['BulkModulus_diff']=training['formulaA_elements_BulkModulus']-\
                             training['formulaB_elements_BulkModulus']

training['Column_diff']=training['formulaA_elements_Column']-\
                        training['formulaB_elements_Column']

training['Density_ratio']=training['formulaA_elements_Density']/\
                          training['formulaB_elements_Density']

training['ElectronSurfaceDensityWS_diff']=training['formulaA_elements_ElectronSurfaceDensityWS']-\
                                          training['formulaB_elements_ElectronSurfaceDensityWS']

training['GSenergy_pa_ratio']=training['formulaA_elements_GSenergy_pa']/\
                              training['formulaB_elements_GSenergy_pa']
training['GSenergy_pa_diff']=training['formulaA_elements_CovalentRadius']\
                                -training['formulaB_elements_CovalentRadius']

#I couldn't figure out what this value represents, but from its values these should work
training['GSestBCClatcnt_ratio']=training['formulaA_elements_GSestBCClatcnt']/\
                                 training['formulaB_elements_GSestBCClatcnt']
training['GSestBCClatcnt_diff']=training['formulaA_elements_GSestBCClatcnt']\
                                -training['formulaB_elements_GSestBCClatcnt']

training['GSestFCClatcnt_ratio']=training['formulaA_elements_GSestFCClatcnt']/\
                                 training['formulaB_elements_GSestFCClatcnt']
training['GSestFCClatcnt_diff']=training['formulaA_elements_GSestFCClatcnt']\
                                -training['formulaB_elements_GSestFCClatcnt']

training['GSmagmom_diff']=training['formulaA_elements_GSmagmom']\
                          -training['formulaB_elements_GSmagmom']

training['GSvolume_pa_diff']=training['formulaA_elements_GSvolume_pa']\
                             -training['formulaB_elements_GSvolume_pa']
training['GSvolume_pa_ratio']=training['formulaA_elements_GSvolume_pa']/\
                              training['formulaB_elements_GSvolume_pa']

training['HHIr_diff']=training['formulaA_elements_HHIr']\
                      -training['formulaB_elements_HHIr']
training['HHIp_diff']=training['formulaA_elements_HHIp']\
                      -training['formulaB_elements_HHIp']

#heat capacity data appears incomplete, perhaps best not to use it?

training['ICSDVolume_ratio']=training['formulaA_elements_ICSDVolume']/\
                             training['formulaB_elements_ICSDVolume']
training['ICSDVolume_diff']=training['formulaA_elements_ICSDVolume']\
                            -training['formulaB_elements_ICSDVolume']

training['element_number_ratio']=training['formulaA_elements_Number']/\
                                 training['formulaB_elements_Number']

training['element_number_difference']=training['formulaA_elements_Number']-\
                                      training['formulaB_elements_Number']

training['polarizability_ratio']=training['formulaA_elements_Polarizability']/\
                                 training['formulaB_elements_Polarizability']
training['polarizability_difference']=training['formulaA_elements_Polarizability']-\
                                      training['formulaB_elements_Polarizability']

training['row_difference']=training['formulaA_elements_Row']-\
                           training['formulaB_elements_Row']

training['spacegroupnumber_ratio']=training['formulaA_elements_SpaceGroupNumber']/\
                                   training['formulaB_elements_SpaceGroupNumber']

training['spacegroupnumber_difference']=training['formulaA_elements_SpaceGroupNumber']-\
                                        training['formulaB_elements_SpaceGroupNumber']

training['avg_coordination_ratio']=training['avg_coordination_A']/\
                                   training['avg_coordination_B']
training['avg_coordination_difference']=training['avg_coordination_A']-\
                                        training['avg_coordination_B']

training['avg_nn_ratio']=training['avg_nearest_neighbor_distance_A']/\
                         training['avg_nearest_neighbor_distance_B']
training['avg_nn_difference']=training['avg_nearest_neighbor_distance_A']-\
                              training['avg_nearest_neighbor_distance_B']

training['meltingT_ratio']=training['formulaA_elements_MeltingT']/\
                           training['formulaB_elements_MeltingT']
training['meltingT_difference']=training['formulaA_elements_MeltingT']-\
                                training['formulaB_elements_MeltingT']

training['mradius_difference']=training['formulaA_elements_MiracleRadius']-\
                               training['formulaA_elements_MiracleRadius']



#Now for the 1/0 values for different areas of the periodic table
training['is_both_alkali']=training['formulaA_elements_IsAlkali']*\
                           training['formulaB_elements_IsAlkali']
training['alkali_difference']=abs(training['formulaA_elements_IsAlkali']-\
                                  training['formulaB_elements_IsAlkali'])

training['is_both_dblock']=training['formulaA_elements_IsDBlock']*\
                           training['formulaB_elements_IsDBlock']
training['dblock_difference']=abs(training['formulaA_elements_IsDBlock']-\
                                  training['formulaB_elements_IsDBlock'])

training['is_both_fblock']=training['formulaA_elements_IsFBlock']*\
                           training['formulaB_elements_IsFBlock']
training['fblock_difference']=abs(training['formulaA_elements_IsFBlock']-\
                                  training['formulaB_elements_IsFBlock'])

training['is_both_metal']=training['formulaA_elements_IsMetal']*\
                          training['formulaB_elements_IsMetal']
training['metal_difference']=abs(training['formulaA_elements_IsMetal']-\
                                 training['formulaB_elements_IsMetal'])

training['is_both_metalloid']=training['formulaA_elements_IsMetalloid']*\
                              training['formulaB_elements_IsMetalloid']
training['metalloid_difference']=abs(training['formulaA_elements_IsMetalloid']-\
                                     training['formulaB_elements_IsMetalloid'])

training['is_both_nonmetal']=training['formulaA_elements_IsNonmetal']*\
                             training['formulaB_elements_IsNonmetal']
training['nonmetal_difference']=abs(training['formulaA_elements_IsNonmetal']-\
                                    training['formulaB_elements_IsNonmetal'])

#Now for the valence electrons in different orbitals

#this first one generated inf values from the zeros, though I think this might still
#be a useful metric
#training['num_unfilled_ratio']=training['formulaA_elements_NUnfilled']/\
#                              training['formulaB_elements_NUnfilled']
training['num_unfilled_difference']=training['formulaA_elements_NUnfilled']-\
                                    training['formulaB_elements_NUnfilled']

training['num_valence_ratio']=training['formulaA_elements_NValance']/\
                              training['formulaB_elements_NValance']
training['num_valence_difference']=training['formulaA_elements_NValance']-\
                                   training['formulaB_elements_NValance']

training['num_dvalence_difference']=training['formulaA_elements_NdValence']-\
                                    training['formulaB_elements_NdValence']

training['num_fvalence_difference']=training['formulaA_elements_NfValence']-\
                                    training['formulaB_elements_NfValence']

training['num_pvalence_difference']=training['formulaA_elements_NpValence']-\
                                    training['formulaB_elements_NpValence']

training['num_svalence_difference']=training['formulaA_elements_NsValence']-\
                                    training['formulaB_elements_NsValence']

training['num_sunfilled_difference']=training['formulaA_elements_NsUnfilled']-\
                                     training['formulaB_elements_NsUnfilled']

training['num_punfilled_difference']=training['formulaA_elements_NpUnfilled']-\
                                     training['formulaB_elements_NpUnfilled']

training['num_dunfilled_difference']=training['formulaA_elements_NdUnfilled']-\
                                     training['formulaB_elements_NdUnfilled']

training['num_funfilled_difference']=training['formulaA_elements_NfUnfilled']-\
                                     training['formulaB_elements_NfUnfilled']





#visualize the data

#Look at any one value and compare whether it is likely to
#successfully classify the outcome for one of the vector elements
s='BoilingT'
plt.plot(training['formulaA_elements_%s' % (s)]/training['formulaB_elements_%s' % (s)],
         training['stabilityVec_total'], 'bx')
plt.show()

#same thing, but without noble gases
s='BoilingT'
plt.plot(training_nonoble['formulaA_elements_%s' % (s)]/training_nonoble['formulaB_elements_%s' % (s)],
         training_nonoble['stabilityVec_total'], 'bx')
plt.show()




#Classify the data

#make sure you only use the columns you want
col_names=list(training.columns)

col_names.remove('formulaA')
col_names.remove('formulaB')
col_names.remove('stabilityVec')
col_names.remove('stabilityVec_total')

for i in range(9):
    col_names.remove('stabilityVec%s' % i)


#split into train and test with sklearn, this example uses stabilityVec2
train,test=sklearn.model_selection.train_test_split \
    (training, test_size=.2, stratify=training['stabilityVec2'])
#noble gases removed:
#train,test=sklearn.model_selection.train_test_split \
# (training_nonoble, test_size=.2, stratify=training_nonoble['stabilityVec2'])



#set up your train and test values
train_X=train[col_names]
train_Y=train['stabilityVec2']
test_X=test[col_names]
test_Y=test['stabilityVec2']

#Set up a Random Forest
model=RandomForestClassifier(500, oob_score=True, n_jobs=-1)
fit=model.fit(train_X, train_Y)
rf_prediction=fit.predict(test_X)

#check the confusion matrix
pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, rf_prediction), \
             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])

#print Random Forest Feature Weights
print('Random Forest Feature Weights')
for i in range(0, len(test_X.keys())): print(fit.feature_importances_[i], test_X.keys()[i])



#Try Gradient Boosting
model=GradientBoostingClassifier()
fit=model.fit(train_X, train_Y)
gb_prediction=fit.predict(test_X)

#check the confusion matrix
pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, gb_prediction), \
             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])

#print Gradient Boosting Feature Weights
print('Gradient Boosting Feature Weights')
for i in range(0, len(test_X.keys())): print(fit.feature_importances_[i], test_X.keys()[i])