import math

from parsers import parser


class LinearModelParser(parser.IParser):

    __PROPORTION = 0.7

    def __init__(self):
        self.__train_samples_num = 0
        self.__list_of_instances = []
        self.__list_of_labels = []
        self.__list_of_samples = []
        self.__max = 0

    def __load_data(self, filename):
        list_of_instances = []
        list_of_labels = []

        with open(filename) as input_stream:
            header_line = input_stream.readline()
            columns = header_line.strip().split(",")
            user_id = None
            instance = None
            for line in input_stream:
                line = line.strip().split(",")
                new_instance = dict(zip(columns[:-2], line[:-2]))
                new_instance[columns[-2]] = [line[-2]]
                new_label = int(line[-1])
                new_instance[columns[-1]] = [new_label]
                new_user_id = int(line[1])
                if user_id is None:
                    user_id = new_user_id
                    instance = new_instance
                elif user_id == new_user_id:
                    instance[columns[-2]].append(line[-2])
                    instance[columns[-1]].append(new_label)
                else:
                    self.__max = max(self.__max, len(instance[columns[-1]]))
                    list_of_instances.append(instance)
                    list_of_labels.append(user_id)
                    user_id = new_user_id
                    instance = new_instance
        # print(list_of_instances[:3])
        # print(list_of_labels[:3])
        return list_of_instances, list_of_labels

    def __get_good_id(self, instance):
        result = instance["good_id"]
        if len(result) < self.__max:
            result += [0] * (self.__max - len(result))
        return result

    def __to_sample(self, instance):  # additional_data
        return self.__get_good_id(instance)

    @staticmethod
    def __to_interim_label(label):
        return math.log(label + 1)

    @staticmethod
    def __to_final_label(interim_label):
        return math.exp(interim_label) - 1

    def parse(self, filepath_or_buffer, to_list=False, **kwargs):
        self.__list_of_instances, self.__list_of_labels = self.__load_data(
            filepath_or_buffer
        )
        # print(len(self.__list_of_instances), len(self.__list_of_labels))
        # print(self.__list_of_instances[:3])
        # print(self.__list_of_labels[:3])
        # print(list(map(self.__to_sample, self.__list_of_instances[:3])))

        # print(self.__to_final_label(self.__to_interim_label(42)))

        self.__list_of_samples = list(
            map(self.__to_sample, self.__list_of_instances)
        )

        self.__train_samples_num = int(self.__PROPORTION
                                       * len(self.__list_of_labels))
        # print(self.__train_samples_num)

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
