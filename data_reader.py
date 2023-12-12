# libraries importing
from implementations.transformers.named_transformers import *
from implementations.transformers.basic_transformers import BasicTransformer
from implementations.pre_processings import StandardPreProcessing
from implementations.input_encodings import *
from implementations.classification_heads import *
from framework import NamedDatasetSpecifications, EvaluationDatasetSampling, FlowTransformer, FlowTransformerParameters, ModelInputSpecification, CategoricalFormat
import pandas as pd
import keras
import os
import pymysql
import threading
import queue
import time
import random

import sys
sys.path.insert(
    0, '~/transformer_model/lib')

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras



data_queue = queue.Queue()

db_config = {
    'host': 'localhost',
    'user': 'flow_collector',
    'password': 'mini',
    'db': 'collected_flows',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
}


def random_ftp_command_ret_code_generator():
    return 100 + random.randrange(100)


# Variable to store the last fetched id
last_fetched_id = 0

# for getting data from sql-database
def fetch_data():
    global last_fetched_id

    while True:
        try:
            connection = pymysql.connect(**db_config)

            with connection.cursor() as cursor:
                query = "SELECT * FROM flow4_flows where idx > %s LIMIT 4;"

                cursor.execute(query, (last_fetched_id,))
                rows = cursor.fetchall()
                for row in rows:
                    data_queue.put(row)
                    last_fetched_id = max(last_fetched_id, row['idx'])

            connection.close()

        except pymysql.Error as err:
            print(f"Error: {err}")


############################
    # Transformer part
############################
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
datasets = [
    ("CSE_CIC_IDS", os.path.join(flow_file_path, "output.csv"),
     NamedDatasetSpecifications.unified_flow_format, 0.06, EvaluationDatasetSampling.LastRows),
    # ("UNSW_2018_IoT_Botnet_Full5pc_1.csv", os.path.join(flow_file_path, "UNSW_2018_IoT_Botnet_Full5pc_1.csv"), NamedDatasetSpecifications.nsl_kdd, 0.05, EvaluationDatasetSampling.RandomRows),
    ("NF-UNSW-NB15-v2", os.path.join(flow_file_path, "initial_dataset_1.csv"),
     NamedDatasetSpecifications.unified_flow_format, 0.94, EvaluationDatasetSampling.LastRows)
    # ("NF-UNSW-NB15-v2", os.path.join(flow_file_path, "NF-UNSW-NB15-v2.csv"), NamedDatasetSpecifications.unified_flow_format, 0.1, EvaluationDatasetSampling.LastRows)
]

pre_processing = StandardPreProcessing(n_categorical_levels=32)


ft = FlowTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[0],
                     sequential_model=transformers[1],
                     classification_head=classification_heads[0],
                     params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))  # change


custom_obj = {
    "FlowTransformer": FlowTransformer,
    "BasicTransformer": BasicTransformer,
    "TransformerDecoderBlock": TransformerDecoderBlock,
}


l_m = keras.saving.load_model("integer_encoded_model.keras",
                              compile=True, safe_mode=False, custom_objects=custom_obj)
print(l_m.summary())
l_m.compile(optimizer="adam", loss='binary_crossentropy',
            metrics=['binary_accuracy'])


# for processing data from queue
def process_data():
    # global data_queue
    while True:

        collected_data = []

        #predicting in batches of 100
        while len(collected_data) < 100:
            if not data_queue.empty():
                data = data_queue.get(block=False)

                data.pop('idx', None)
                data.pop('IPV6_SRC_ADDR', None)
                data.pop('IPV6_DST_ADDR', None)

                # these values are not available in nProbe Pro.
                # data['DNS_QUERY_ID'] = 0
                # data['DNS_QUERY_TYPE'] = 0
                # data['DNS_TTL_ANSWER'] = 0
                # data['FTP_COMMAND_RET_CODE'] = random_ftp_command_ret_code_generator()

                # To make training and predicting df same, but Label and Attack category will not be used in prediction
                data['Label'] = 0
                data['Attack'] = 'Benign'
                collected_data.append(data)

        if len(collected_data) == 0:
            continue

        string_data_list = [
            {key: str(value) for key, value in d.items()} for d in collected_data]

        # print(string_data_list)
        df = pd.DataFrame(string_data_list)
        # print((df.shape))

        # To solve issues related to data type
        df.to_csv('./temp.csv')
        df1 = pd.read_csv("./temp.csv")

        dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[
            1]
        # ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)
        preprocessed_df = ft.load_dataset(dataset_name, df1, dataset_specification,
                                          evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

        a = ft.predict(l_m, batch_size=128)
        # time.sleep(1)
        """
        """


fetch_thread = threading.Thread(target=fetch_data)
process_thread = threading.Thread(target=process_data)

fetch_thread.start()
process_thread.start()

fetch_thread.join()
process_thread.join()
