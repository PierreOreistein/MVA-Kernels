# Math packages
import numpy as np

# Progress bar
from tqdm import tqdm

# Import different tools functions
from Modules.Utils.Dropout import *
from Modules.Utils.Transform import *
from Modules.Utils.ImportData import *
from Modules.Utils.Normalisation import *
from Modules.Utils.Preprocessing import *
from Modules.Utils.Predictions import *

# Import functions for the data augmentation
from Modules.DataAugmentations.NoAugmentation import *
from Modules.DataAugmentations.ComplementarySequences import *
from Modules.DataAugmentations.PertubedSequences import *

# Import functions for the embedding
from Modules.Embeddings.NoEmbedding import *
from Modules.Embeddings.FiguresEmbedding import *
from Modules.Embeddings.SpectrumEmbedding import *
from Modules.Embeddings.DimismatchEmbedding import *
from Modules.Embeddings.DimismatchEmbedding2 import *
from Modules.Embeddings.MotifEmbedding import *
from Modules.Embeddings.WeightedDegreeEmbedding import *
from Modules.Embeddings.OneHotEmbedding import *
# from Modules.Embeddings.HMMEmbedding import *

# Import functions for the selection of the model
from Modules.ModelSelection.CrossValidation import *
from Modules.ModelSelection.GridSearch import *

# Import functions for the kernels
from Modules.Kernels.LinearKernel import *
from Modules.Kernels.PolyKernel import *
from Modules.Kernels.DimismatchPolyKernel import *
from Modules.Kernels.GaussianKernel import *
from Modules.Kernels.HMM import *
from Modules.Kernels.SpectrumKernel import *

# Import function of model
from Modules.Models.KernelLogisticRegression import *
from Modules.Models.KernelSVM import *


if __name__ == '__main__':
    # Extraction of the dataset
    df_mat_dict = ImportData("./Data/Optionnal/", "./Data/", suffix="_mat100")
    df_dict = ImportData("./Data/", "./Data/", header=0, sep=",")

    # Hyperparameters for DataAugmentation
    hyperparameters_data_augmentation = {
        NoAugmentation: {},
        # PertubedSequences: {"n": [2], "add_compl": [False]},
        # ComplementarySequences: {}
    }

    # Hyperparameters for the embedding
    hyperparameters_embedding = {
        # NoEmbedding: {}
        SpectrumEmbedding: {
            "d_l": [[5, 7, 12]]
        }
        # FiguresEmbedding: {},
        # DimismatchEmbedding: {"d": [[5, 7, 12]]}
    }

    # Hyperparameters of the kernels
    hyperparameters_kernels = {
        # SpectrumKernel: {"d_l": [[5, 7, 12]]}
        PolyKernel: {
            "k": [10],
            "add_ones": [True]
        }
        # DimismatchPolyKernel: {
        #      "m" : [3],
        #      "k": [2],
        #      "add_ones": [True],
        #      "d_l": [[5, 6, 7]]
        #  }
        # GaussianKernelBIS: {"sigma": [None, 10000, 10]}
    }

    # Hyper-parameters of the models
    hyperparamters_models = {
        KernelLogisticRegression: {
            "lamda": [1],
            "preprocessing": [None],
            "informations": [False],
            "normalisation": [None],
            "max_iter": [15],
         },
         # KernelSVM: {
         #      "lamda": [1, 0.01],
         #      "max_iter": [10e4],
         #      "tol": [10e-6],
         #      "informations": [False],
         #      "preprocessing": [Preprocessing, None]
         # }
    }


    # GridSearch
    [best_score, best_parameters_names,
     best_parameters_values] = GridSearch(df_dict,
                                          hyperparameters_data_augmentation,
                                          hyperparameters_embedding,
                                          hyperparamters_models,
                                          hyperparameters_kernels,
                                          cv=5)

    # Display results
    print("Best Score: ", best_score)
    print("Best Parameters: ", best_parameters_names)

    # Do the Predictions
    predictions = Prediction(best_parameters_values, df_dict)

    # Save the predictions
    np.savetxt("/Resultats/Predictions_Spectrum.csv", predictions,
               fmt='%i', delimiter=",", header="Id,Bound", comments='')
