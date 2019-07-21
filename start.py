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
from Modules.Embeddings.MotifEmbedding import *
from Modules.Embeddings.WeightedDegreeEmbedding import *
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

    # Definition of the data augmentation function
    data_aug = DataAugmentationDefault(NoAugmentation, {})

    # Defintion of the embedding
    embedding = EmbeddingDefault(SpectrumEmbedding, {"d_l": [5, 7, 12]})
    #

    # Definition of the kernel
    kernel = KernelDefault(PolyKernel, {"k": 2})

    # Definition of the model
    model = KernelLogisticRegression(kernel, informations=False,  lamda=1,
                                     max_iter=15, preprocessing=None)

    # Defintion of best parameters values
    best_parameters_values = {"Data Augmentation": {"Function": data_aug},
                              "Embedding": {"Function": embedding},
                              "Kernel": {"Function": kernel},
                              "Model": {"Function": model}}

    # Computation of the predicition
    predictions = Prediction(best_parameters_values, df_dict)

    # Save the Predicitons
    np.savetxt("Yte.csv", predictions,
               fmt='%i', delimiter=",", header="Id,Bound", comments='')
