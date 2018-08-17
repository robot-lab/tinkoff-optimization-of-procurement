import math

import pandas as pd

from parsers import parser


class LinearModelParser(parser.IParser):

    __PROPORTION = 0.7

    def __init__(self):
        self.__train_samples_num = 0
        self.__list_of_instances = []
        self.__list_of_labels = []
        self.__list_of_samples = []

    @staticmethod
    def __parse_menu(filepath_or_buffer):
        df = pd.read_csv(filepath_or_buffer)
        indices = list(df["good_id"])
        indices = set(sorted(indices))
        return indices

    @staticmethod
    def __load_data(filepath_or_buffer):
        df = pd.read_csv(filepath_or_buffer)
        result = df.groupby(['chknum', 'person_id', 'month', 'day'],
                            as_index=False).agg(list)
        list_of_instances = list(
            result.drop("good_id", axis=1).T.to_dict().values()
        )
        list_of_labels = list(result["good_id"])
        return list_of_instances, list_of_labels

    @staticmethod
    def __get_person_id(instance):
        result = instance["person_id"]
        return [result]

    def __to_sample(self, instance):  # additional_data
        return self.__get_person_id(instance)

    @staticmethod
    def __to_interim_label(label):
        return math.log(label + 1)

    @staticmethod
    def __to_final_label(interim_label):
        return math.exp(interim_label) - 1

    def parse(self, filepath_or_buffer, to_list=False, **kwargs):
        # self.__parse_menu(filepath_or_buffer)

        self.__list_of_instances, self.__list_of_labels = self.__load_data(
            filepath_or_buffer
        )
        # print(len(self.__list_of_instances), len(self.__list_of_labels))
        # print(self.__list_of_instances[:3])
        # print(self.__list_of_labels[:3])
        # print(list(map(self.__to_sample, self.__list_of_instances[:3])))

        # print(self.__to_final_label(self.__to_interim_label([42])))

        self.__list_of_samples = list(
            map(self.__to_sample, self.__list_of_instances)
        )
        # print(self.__list_of_samples[:3])

        self.__train_samples_num = int(self.__PROPORTION
                                       * len(self.__list_of_labels))

    def get_train_data(self):
        train_samples = self.__list_of_samples[:self.__train_samples_num]
        train_labels = list(
            map(self.__to_interim_label,
                self.__list_of_labels[:self.__train_samples_num])
        )
        return train_samples, train_labels

    def get_validation_data(self):
        validation_samples = self.__list_of_samples[self.__train_samples_num:]
        validation_labels = list(
            map(self.__to_interim_label,
                self.__list_of_labels[self.__train_samples_num:])
        )
        return validation_samples, validation_labels
