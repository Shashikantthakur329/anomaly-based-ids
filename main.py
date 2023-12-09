import sys
sys.path.insert(0,'/home/toor/miniproject_ddos/codes/learning_code/transformer_model/lib')
import os
import keras
import pandas as pd

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

from framework import NamedDatasetSpecifications, EvaluationDatasetSampling, FlowTransformer, FlowTransformerParameters, ModelInputSpecification, CategoricalFormat
from implementations.classification_heads import *
from implementations.input_encodings import *
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer
from implementations.transformers.named_transformers import *


encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    CategoricalFeatureEmbed(EmbedLayerType.Dense, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Lookup, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Projection, 16),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    LastTokenClassificationHead(),
    FlattenClassificationHead(),
    GlobalAveragePoolingClassificationHead(),
    CLSTokenClassificationHead(),
    FeaturewiseEmbedding(project=False),
    FeaturewiseEmbedding(project=True),
]

transformers = [
    BasicTransformer(2, 128, n_heads=2),
    BasicTransformer(2, 128, n_heads=2, is_decoder=True),
    GPTSmallTransformer(),
    BERTSmallTransformer()
]

flow_file_path = "../dataset/cleansed_dataset/train"
new_path = "./dataset/JIIS23-dataset-main/case1"
datasets = [
    ("CSE_CIC_IDS", os.path.join(flow_file_path, "output.csv"), NamedDatasetSpecifications.unified_flow_format,0.06, EvaluationDatasetSampling.LastRows),
    # ("UNSW_2018_IoT_Botnet_Full5pc_1.csv", os.path.join(flow_file_path, "UNSW_2018_IoT_Botnet_Full5pc_1.csv"), NamedDatasetSpecifications.nsl_kdd, 0.05, EvaluationDatasetSampling.RandomRows),
    # ("NF-UNSW-NB15-v2", os.path.join(flow_file_path, "initial_dataset.csv"), NamedDatasetSpecifications.unified_flow_format, 0.98, EvaluationDatasetSampling.LastRows)
    # ("NF-UNSW-NB15-v2", os.path.join(flow_file_path, "NF-UNSW-NB15-v2.csv"), NamedDatasetSpecifications.unified_flow_format, 0.1, EvaluationDatasetSampling.LastRows)
    ("JIIS-MAIN", os.path.join(new_path, "27_filtered_33perc_stego_original.csv"), NamedDatasetSpecifications.jiis_main, 0.98, EvaluationDatasetSampling.LastRows)
]

# NF-UNSW-NB15-v2

pre_processing = StandardPreProcessing(n_categorical_levels=32)
# categorical_format = CategoricalFormat()
# model_input_spec = ModelInputSpecification(categorical_format=CategoricalFormat)
######################################
# model_input_spec=ModelInputSpecification(categorical_format=)


# Define the transformer
ft = FlowTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[0],
                     sequential_model=transformers[1],
                     classification_head=classification_heads[0],
                     params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))#change


custom_obj = {
        "FlowTransformer" : FlowTransformer,
        "BasicTransformer" : BasicTransformer,
        "TransformerDecoderBlock" : TransformerDecoderBlock,
}


#df = pd.read_csv("../dataset/cleansed_dataset/train/NF-UNSW-NB15-v2.csv")
#print(df.head())


# Load the specific dataset
df = pd.read_csv("../dataset/cleansed_dataset/train/initial_dataset_1.csv")
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
# ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)
print("asdasdASDASDASDASD");
preprocessed_df = ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent);
print(df.head())
print("$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$")
print(preprocessed_df.columns)
print(preprocessed_df.head())




# Build the transformer model
m = ft.build_model()
#print(m.dir())
#m.summary()
"""

l_m = keras.saving.load_model("integer_encoded_model.keras",compile=True,safe_mode = False, custom_objects = custom_obj)
print(l_m.summary())
l_m.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics=['binary_accuracy'])


# eval_results: pd.DataFrame
# (train_results, eval_results, final_epoch) = ft.evaluate(l_m, batch_size=128, epochs=20, steps_per_epoch=64, early_stopping_patience=5)
# print(eval_results)

a = ft.predict(l_m, batch_size=128, epochs=20, steps_per_epoch=64, early_stopping_patience=5)

"""


"""
"""
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'], jit_compile=True)

# Get the evaluation results
eval_results: pd.DataFrame
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=20, steps_per_epoch=64, early_stopping_patience=5)
print(eval_results)

m.save('./JIIS.keras')









## Prediction results : 
# 11 inputs needed for prediction...
# 41 input : 3 predictions
# 42 input : 4 predictions
