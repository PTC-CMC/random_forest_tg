from collections import OrderedDict
import json
import argparse

from atools_ml.dataio import df_setup
from atools_ml.descriptors import rdkit_descriptors
from atools_ml.prep import dimensionality_reduction, train_test_split
import numpy as np
import pandas as pd
import signac
import scipy
import pprint
from sklearn import ensemble, linear_model, metrics, model_selection

"""
INPUT


SMILES FOR TOP AND BOTTOM TERMINAL GROUPS

`SMILES1` corresponds to the SMILES for the terminal group of the bottom
monolayer, terminated by a hydrogen. For example, a hydroxyl group this
would simply be 'O'.

`SMILES2` corresponds to the SMILES for the terminal group of the top
monolayer, terminated by a hydrogen. For example, a methyl group this
would simply be 'C'.


RANDOM SEED

This is the random number generator seed used for test/train splits for
generating the random forest model. In the manuscript, the following
seeds were used:

    Model       Seed
    -----       ----
    1           43
    2           0
    3           1
    4           2
    5           3

As the manuscript reports on the results of Model 1, a seed of 43 is used
by default.

`path_to_data` is the relative path to the MD screening data. If the
installation instructions were followed exactly, these two directories
('terminal_group_screening' and 'terminal_groups_mixed') should be located
one above the current working directory. If these were placed elsewhere,
the `path_to_data` string should be updated accordingly.
"""

#SMILES1 = 'C(=O)N'
#SMILES2 = 'O'
#random_seed = 43
#
#path_to_data = ".."

"""

The code below will use the seed to generate the random forest model and
predict coefficient of friction and adhesion for the SMILES combination
provided.

Features are obtained using RDKit and have manually been classified into
describing either shape, size, charge distribution, or complexity. The
corresponding clusters are provided in the `feature-cluster.json` file.

"""

def predict(SMILES1, SMILES2, random_seed=None, path_to_data="../", barcode_seed=None,
        vary_descriptors=False, vary_significant=False, feature_cluster_json_location="./"):

    ch3_SMILES1 = 'C{}'.format(SMILES1)
    ch3_SMILES2 = 'C{}'.format(SMILES2)

    with open(feature_cluster_json_location + 'feature-clusters.json', 'r') as f:
        clusters = json.load(f)
    shape_features = clusters['shape']

    """
    Because Gasteiger charges are assigned, molecules aren't guarenteed to be
    charge neutral. However, the total positive and negative charge are not
    very helpful in predictive modeling, so those features are removed here.

    The "min" and "mean" indicate features describing the minimum and mean
    values between the two terminal groups respectively.
    """
    to_drop = ['pc+-mean', 'pc+-min', 'pc--mean', 'pc--min']

    # Descriptors for H-terminated SMILES
    desc_h_tg1 = rdkit_descriptors(SMILES1, vary_descriptors=vary_descriptors,
            vary_significant=vary_significant, barcode_seed=barcode_seed)
    desc_h_tg2 = rdkit_descriptors(SMILES2, vary_descriptors=vary_descriptors,
            vary_significant=vary_significant, barcode_seed=barcode_seed)

    # Descriptors for CH3-terminated SMILES
    desc_ch3_tg1 = rdkit_descriptors(ch3_SMILES1, include_h_bond=True,
                                     ch3_smiles=ch3_SMILES1,
                                     vary_descriptors=vary_descriptors,
                                     vary_significant=vary_significant,
                                     barcode_seed=barcode_seed)
    desc_ch3_tg2 = rdkit_descriptors(ch3_SMILES2, include_h_bond=True,
                                     ch3_smiles=ch3_SMILES2,
                                     vary_descriptors=vary_descriptors,
                                     vary_significant=vary_significant,
                                     barcode_seed=barcode_seed)

    desc_h_df = pd.DataFrame([desc_h_tg1, desc_h_tg2])
    desc_ch3_df = pd.DataFrame([desc_ch3_tg1, desc_ch3_tg2])

    desc_df = []
    for i, df in enumerate([desc_h_df, desc_ch3_df]):
        if i == 1:
            hbond_tb = max(df['hdonors'][0], df['hacceptors'][1]) \
                       if all((df['hdonors'][0], df['hacceptors'][1])) \
                       else 0
            hbond_bt = max(df['hdonors'][1], df['hacceptors'][0]) \
                       if all((df['hdonors'][1], df['hacceptors'][0])) \
                       else 0
            hbonds = hbond_tb + hbond_bt
            df.drop(['hdonors', 'hacceptors'], 'columns', inplace=True)
        else:
            hbonds = 0
        means = df.mean()
        mins = df.min()
        means = means.rename({label: '{}-mean'.format(label)
                              for label in means.index})
        mins = mins.rename({label: '{}-min'.format(label)
                            for label in mins.index})
        desc_tmp = pd.concat([means, mins])
        desc_tmp['hbonds'] = hbonds
        desc_tmp.drop(labels=to_drop, inplace=True)
        desc_df.append(desc_tmp)

    df_h_predict = desc_df[0]
    df_ch3_predict = desc_df[1]
    df_h_predict = pd.concat([
        df_h_predict.filter(like=feature) for feature in shape_features], axis=0)
    df_ch3_predict.drop(labels=df_h_predict.keys(), inplace=True)

    df_h_predict_mean = df_h_predict.filter(like='-mean')
    df_h_predict_min = df_h_predict.filter(like='-min')
    df_ch3_predict_mean = df_ch3_predict.filter(like='-mean')
    df_ch3_predict_min = df_ch3_predict.filter(like='-min')

    df_predict = pd.concat([df_h_predict_mean, df_h_predict_min,
                            df_ch3_predict_mean, df_ch3_predict_min,
                            df_ch3_predict[['hbonds']]])

    """
    Load data from MD screening
    """
    root_dir_same = ('{}/terminal_group_screening'.format(path_to_data))
    proj_same = signac.get_project(root=root_dir_same)

    root_dir_mixed = ('{}/terminal_groups_mixed'.format(path_to_data))
    proj_mixed = signac.get_project(root=root_dir_mixed)

    # Define chemistry identifiers and target variable
    identifiers = ['terminal_group_1', 'terminal_group_2']
    targets= ['COF', 'intercept']

    df_h = df_setup([proj_same, proj_mixed], mean=True,
                    descriptors_filename='descriptors-h.json',
                    smiles_only=True)
    df_ch3 = df_setup([proj_same, proj_mixed], mean=True,
                      descriptors_filename='descriptors-ch3.json',
                      smiles_only=True)

    to_drop = ['pc+-mean', 'pc+-diff', 'pc+-min', 'pc+-max',
               'pc--mean', 'pc--diff', 'pc--min', 'pc--max']

    df_h.drop(labels=to_drop, axis=1, inplace=True)
    df_ch3.drop(labels=to_drop, axis=1, inplace=True)

    shape_features = clusters['shape']
    df_h = pd.concat([
                df_h.filter(like=feature) for feature in shape_features],
                axis=1)
    df_ch3.drop(labels=df_h.columns, axis=1, inplace=True)

    df_h_mean = df_h.filter(like='-mean')
    df_h_min = df_h.filter(like='-min')
    df_ch3_mean = df_ch3.filter(like='-mean')
    df_ch3_min = df_ch3.filter(like='-min')

    df = pd.concat([df_h_mean, df_h_min, df_ch3_mean, df_ch3_min,
                    df_ch3[identifiers + targets + ['hbonds']]], axis=1)
    # Reduce the number of features by running them through various filters
    features = list(df.drop(identifiers + targets, axis=1))
    df_red = dimensionality_reduction(df, features, filter_missing=True,
                                      filter_var=True, filter_corr=True,
                                      missing_threshold=0.4,
                                      var_threshold=0.02,
                                      corr_threshold=0.9)
    df = df_red
    features = list(df.drop(identifiers + targets, axis=1))
    df_predict = df_predict.filter(features)
    
    prediction_results = dict()
    for target in ['COF', 'intercept']:
        train_list = list()
        test_list = list()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
                df[features], df[target], test_size=0.2,
                random_state=random_seed)

        regr = ensemble.RandomForestRegressor(n_estimators=1000,
                                              oob_score=True,
                                              random_state=random_seed)
        regr.fit(X_train, y_train)

        predicted = regr.predict(np.array(df_predict).reshape(1, -1))
        prediction_results[target] = predicted[0]
        prediction_results['R2'] = regr.score(X_train, y_train)
        #print('{} (predicted): {:.4f}'.format(target, predicted[0]))
    #prediction_results['']

    model_train_test_split_df_list = []
    test_list_test = []
    test_list_sys = []
    for ndx in X_test.index:
        test_list.append((df.at[ndx, 'terminal_group_1'],
                          df.at[ndx, 'terminal_group_2'],
                          df.at[ndx, 'COF'],
                          df.at[ndx, 'intercept']))

        systems = str(df.at[ndx, 'terminal_group_1']) + " - " + str(df.at[ndx, 'terminal_group_2'])
        test_list_test.append('Test')
        test_list_sys.append(systems)

        model_train_test_split_df_list.append({'Model {}'.format(random_seed): 'Test', 'System': systems,
                                              'Terminal Group 1': df.at[ndx, 'terminal_group_1'],
                                              'Terminal Group 2': df.at[ndx, 'terminal_group_2']})
        
        #for job in proj_mixed.find_job_documents({"COF": df.at[ndx, 'COF']}):
        #    print('job is')
        #    pprint.pprint(job)
        #    print()
    #model_train_test_split_df_list.append({'System': test_list_sys})
    #model_train_test_split_df_list.append({'Model {}'.format(random_seed): test_list_test})

    train_list_train = []
    train_list_sys = []
    for ndx in X_train.index:

        train_list.append((df.at[ndx, 'terminal_group_1'],
                           df.at[ndx, 'terminal_group_2'],
                           df.at[ndx, 'COF'],
                           df.at[ndx, 'intercept']))
        systems = str(df.at[ndx, 'terminal_group_1']) + " - " + str(df.at[ndx, 'terminal_group_2'])
        train_list_train.append('Train')
        train_list_sys.append(systems)
        model_train_test_split_df_list.append({'Model {}'.format(random_seed): 'Train', 'System': systems,
                                              'Terminal Group 1': df.at[ndx, 'terminal_group_1'],
                                              'Terminal Group 2': df.at[ndx, 'terminal_group_2']})
    
    #model_train_test_split_df_list.append({'System': *train_list_sys})
    #model_train_test_split_df_list.append({'Model {}'.format(random_seed): *train_list_train})


    prediction_results['test_data'] = test_list
    prediction_results['train_data'] = train_list
    prediction_results['Model Number'] = random_seed
    model_train_test_split_df = pd.DataFrame(model_train_test_split_df_list)
    model_train_test_split_df.sort_values(by='System').to_csv("./model_{}_summary_test_train_andrew.csv".format(random_seed))
    model_train_test_split_df.to_html("./model_{}_summary_test_train_andrew.html".format(random_seed))
    #pprint.pprint(model_train_test_split_df.sort_values(by='System'))
    return prediction_results


def main():
    parser = argparse.ArgumentParser(description='RF model for COF and'\
            'Intercept prediction.')
    parser.add_argument('--smi1', '-s1', type=str, default='C(=O)N',
            help='smiles string for one monolayer')
    parser.add_argument('--smi2', '-s2', type=str, default='O',
            help='smiles string for other monolayer')
    parser.add_argument('--model', type=int, default=43,
            help='random seed for the training split')
    parser.add_argument('--path', '-p', type=str, default='..',
            help='relative path to the data sets')
    parser.add_argument('--signac', '-sig', type=str, default=None,
            help='path to signac workspace to store results of prediction')
    parser.add_argument('--modelname', type=str, default=None,
            help='model name to associate with random seed'\
            'if none passed name will be the str(--model)')
    parser.add_argument('--barcodeseed', '-b', type=int, default=None,
            help='random seed to vary the descriptors values')
    parser.add_argument('--varydescriptors', type=bool, default=None,
            help='randomly vary the rdkit descriptors?')
    parser.add_argument('--varysignificant', type=bool, default=None,
            help='random seed to vary the descriptors values')

    args = parser.parse_args()
    
    predicted = predict(args.smi1, args.smi2, args.model, args.path,
            args.barcodeseed, args.varydescriptors, args.varysignificant)
    #print(predicted)
    #pprint.pprint({**vars(args), **predicted})
    if args.signac:
        barcode_project = signac.get_project(root=args.signac)
        if args.modelname is None:
            args.modelname = str(args.model)

        # add a statepoint for each system
        barcode_project.open_job({**vars(args), **predicted})

if __name__=="__main__":
    main()
