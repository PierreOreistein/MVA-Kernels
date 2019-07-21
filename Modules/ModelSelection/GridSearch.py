from Modules.Utils.ApplyFunctions import *
from Modules.ModelSelection.CrossValidation import CrossValidation
from Modules.DataAugmentations.DataAugmentationDefault import *
from Modules.Embeddings.EmbeddingDefault import *
from Modules.Kernels.KernelDefault import *

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm


def dictToCartesianProduct(dct):
    """Transform a dict of lists into a lists of all possible combination."""

    # Convert dict of hyperparameters as list of lists
    dct_l = [dct[key] for key in dct.keys()]

    # Cartesian product of all the possible combination of model hp
    grid_dct_l = [elt for elt in itertools.product(*dct_l)]

    return grid_dct_l


def BuildHpTuples(df_dct, hyperparameters_data_augmentation_dict,
                  hyperparameters_embeddings_dict,
                  hyperparameters_models_dict,
                  hyperparameters_kernels_dict):
    """Build all the possible combination of hyper parameters given as arg."""

    # Grid of hp for all model and kernel functions
    tuples = []

    # Loop over different embedding and data augmentation
    for data_aug_func in tqdm(hyperparameters_data_augmentation_dict):

        # Extract the hp of the kernel as a dict
        dct_data = hyperparameters_data_augmentation_dict[data_aug_func]
        grid_data_hp_l = dictToCartesianProduct(dct_data)

        for hp_data in grid_data_hp_l:
            for embedding_func in hyperparameters_embeddings_dict:

                # Extract the hp of the kernel as a dict
                dct_embedding = hyperparameters_embeddings_dict[embedding_func]
                grid_embedding_hp_l = dictToCartesianProduct(dct_embedding)

                for hp_embedding in tqdm(grid_embedding_hp_l):

                    # Convert hp of the data aug. as a dict
                    keys_list = list(dct_data.keys())
                    hp_data_aug_dct = {keys_list[i]: elt for i, elt in enumerate(hp_data)}

                    # Convert hp of the embedding as a dict
                    keys_list = list(dct_embedding.keys())
                    hp_embedding_dct = {keys_list[i]: elt for i, elt in enumerate(hp_embedding)}

                    # Definition of the embedding
                    if type(embedding_func) == type:
                        embedding = embedding_func(**hp_embedding_dct)
                    else:
                        embedding = EmbeddingDefault(embedding_func,
                                                     hp_embedding_dct)

                    # Definition of the data augmentation
                    data_aug = DataAugmentationDefault(data_aug_func,
                                                       hp_data_aug_dct)

                    # Compute the dataset for these functions
                    computed_df_dct = ApplyFunctions(df_dct, data_aug,
                                                     embedding)

                    # Loop over different models
                    for model_func in hyperparameters_models_dict.keys():

                        # Compute all possible combinations
                        dct_model = hyperparameters_models_dict[model_func]
                        grid_model_hp_l = dictToCartesianProduct(dct_model)

                        for hp_model in grid_model_hp_l:
                            for kernel_func in hyperparameters_kernels_dict.keys():

                                # Extract the hp of the kernel as a dict
                                dct_kernel = hyperparameters_kernels_dict[kernel_func]
                                grid_kernel_hp_l = dictToCartesianProduct(dct_kernel)

                                for hp_kernel in grid_kernel_hp_l:

                                    # Convert the hp of the kernel as a dict
                                    keys_list = list(dct_kernel.keys())
                                    hp_kernel_dct = {keys_list[i]: elt for i, elt in enumerate(hp_kernel)}

                                    # Convert hp of the model as a dict
                                    keys_list = list(dct_model.keys())
                                    hp_model_dct = {keys_list[i]: elt for i, elt in enumerate(hp_model)}

                                    # Definition of the kernel
                                    kernel = KernelDefault(kernel_func,
                                                           hp_kernel_dct)

                                    # Definition of the model with the current hyperparameters
                                    model = model_func(kernel, **hp_model_dct)

                                    # Add this combination ot tuples
                                    tuples.append((data_aug, hp_data_aug_dct,
                                                   embedding, hp_embedding_dct,
                                                   kernel, hp_kernel_dct,
                                                   model, hp_model_dct,
                                                   computed_df_dct))

    return tuples


def subGridSearch(df_dct, tuple_i, res_df, cv=5, n_jobs=-1):
    """Execute the CrossValidation on tuple_i of hyperparameters."""

    # Extract the relevant object from tuple_i
    [data_aug, hp_data_aug_dct,
     embedding, hp_embedding_dct,
     kernel, hp_kernel_dct,
     model, hp_model_dct, computed_df_dct] = tuple_i

    # Computation of the score trough a Cross Validation
    scores = CrossValidation(computed_df_dct, model, cv=cv, n_jobs=n_jobs,
                             return_int=False)

    # Compute the mean scores
    scores_1, scores_2, scores_3 = scores
    scores_1_mean = np.mean(scores_1)
    scores_2_mean = np.mean(scores_2)
    scores_3_mean = np.mean(scores_3)
    score = (scores_1_mean + scores_2_mean + scores_3_mean) / 3.0

    # Save the result in the dataFRame res_df
    results = {
        'scores_1': scores_1,
        'scores_2': scores_2,
        'scores_3': scores_3,
        'scores_1_mean': scores_1_mean,
        'scores_2_mean': scores_2_mean,
        'scores_3_mean': scores_3_mean,
        'score': score,

        'data_aug_type': data_aug.name,
        'data_aug_hp': hp_data_aug_dct,
        'embedding_type': embedding.name,
        'embedding_hp': hp_embedding_dct,
        'kernel_type': kernel.name,
        'kernel_hp': hp_kernel_dct,
        'model_type': model.name,
        'model_hp': hp_model_dct,
    }
    res_df = res_df.append(results, ignore_index=True)
    res_df.to_csv('./Resultats/grid_search_res.csv', sep='\t')

    # Concatenate the hyperparameters for retruning them
    best_score = score
    best_parameters_names = {"Data Augmentation": {"Function Name": data_aug.name,
                                                   "Best Parameters": hp_data_aug_dct},
                             "Embedding": {"Function Name": embedding.name,
                                           "Best Parameters": hp_embedding_dct},
                             "Kernel": {"Function Name": kernel.name,
                                        "Best Parameters": hp_kernel_dct},
                             "Model": {"Function Name": model.name,
                                       "Best Parameters": hp_model_dct}}
    best_parameters_values = {"Data Augmentation": {"Function": data_aug},
                              "Embedding": {"Function": embedding},
                              "Kernel": {"Function": kernel},
                              "Model": {"Function": model}}

    # Display score and Parameters
    print("Score: {}".format(score))
    print("Best Parameters\n--------")
    print("DataAugmentation: ", data_aug.name, ", hp: ", hp_data_aug_dct)
    print("Embedding: ", embedding.name, ", hp: ", hp_embedding_dct)
    print("Kernel: ", kernel.name, ", hp: ", hp_kernel_dct)
    print("Model: ", model.name, ", hp: ", hp_model_dct)
    print("\n\n")

    return best_score, best_parameters_names, best_parameters_values, res_df


def GridSearch(df_dct, hyperparameters_data_augmentation_dict,
               hyperparameters_embeddings_dict,
               hyperparameters_models_dict,
               hyperparameters_kernels_dict,
               cv=5, n_jobs=-1, randomise=True):
    """Launch a grid search over different value of the hps."""

    # Compute all the possible combinations of hps
    tuples_hp = BuildHpTuples(df_dct, hyperparameters_data_augmentation_dict,
                              hyperparameters_embeddings_dict,
                              hyperparameters_models_dict,
                              hyperparameters_kernels_dict)

    # Creates dataframe in which all results will be stored
    # (allows early stopping of grid search)
    pd_res_df = pd.DataFrame()

    # Executes a Cross Validation for all possible tuples
    scores_param = []

    # Randomisation of the tuples
    if randomise:
        np.random.shuffle(tuples_hp)

    for tuple_i in tqdm(tuples_hp):
        [best_score, best_params_n,
         best_params_v,  pd_res_df] = subGridSearch(df_dct, tuple_i, pd_res_df,
                                                    cv=cv, n_jobs=n_jobs)
        results = (best_score, best_params_n, best_params_v)
        scores_param.append(results)

    # Extract best scores and parameters
    maxi = 0
    best_params_names = 0
    best_params_values = 0

    for sublist in scores_param:

        if sublist[0] > maxi:
            maxi = sublist[0]
            best_params_names = sublist[1]
            best_params_values = sublist[2]

    # Return result
    return maxi, best_params_names, best_params_values
