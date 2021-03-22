import boto3
import json
import pandas as pd
import numpy as np
import random
import re
import os

from global_variables import API_PARAMETERS_FILE
from global_variables import print_green, print_yellow, print_red
from global_variables import service_dict
from global_variables import default_feature_list


def read_api_calls_file():
    with open(API_PARAMETERS_FILE, "r") as log_file:
        data = json.load(log_file)

    return data


def call_api_with_auto_generated_parameters(api, method, debug=True):
    try:
        client = boto3.client(api)
    except ValueError as e:
        print_red(str(e))

    api_calls = read_api_calls_file()
    #  normal api calls
    api_methods = json.loads(api_calls[api])

    if debug:
        print_green("Print list of methods belong to {0}".format(api))
        for method in api_methods:
            print(method)

    response = None
    try:
        parameters = api_methods[method]
        valued_params = {}
        for parm in parameters:
            valued_params[parm] = "random-value"

        callable_api = getattr(client, method)
        print_yellow(valued_params)

        response = callable_api(**valued_params)

        if debug:
            print_yellow("Response after calling {0}.{1}:{2}".format(api,method,response))

    except ValueError as e:
        print_red(str(e))

    return response


def assign_random_user():
    """
    75% user with temporary credential
    25% root
    """
    rand = random.randint(4)
    if rand >= 1:
        user = "AssumedRole"
    else:
        user = "Root"

    return user


def generate_api_calls_based_on_aws_cloudtrail_logs(log_file, number_of_replication, produce_random_sample=False):
    """
    NOT TO USE THIS FUNCTION
    """
    #  read the log file
    collected_logs = pd.read_csv(log_file)
    columns = collected_logs.columns.tolist()
    # first col: method, second api:request parameters
    for log in collected_logs:
        print(log)

        for i in range(number_of_replication):
            user = assign_random_user()
            #  region = assign_random_region()
            for feature in columns:
                call_api_with_auto_generated_parameters()


class AWSLogAnalyzer(object):

    def __init__(self, log_file, log_dir, read_all_file=False):
        self.log_file = log_dir + log_file
        if read_all_file:
            self.log_df = pd.read_csv(log_dir + log_file)
            # call encoder once
            if not os.path.exists("encoding.py"):
                self.encode_event_names()
            if not os.path.exists("vocab_file"):
                self.fill_vocab_file_for_bert_tokenizer()

    def fill_out_logs_encoder_dict(self):
        if self.log_df is None:
            print("Cannot generate logs encoder; log df is empty")
        else:
            unique_services = self.get_feature_from_dataset("eventsource")
            if len(unique_services) >= 100:
                print_red("There are more than a 100 service. YOU MIGHT HAVE A PROBLEM IN ENCODING")
                return 0
            # a list of services involved in the log file (index of the list shows the encoding)
            service_list = []
            for index, value in enumerate(unique_services):
                if index < 10:
                    service_dict[value] = str(index).zfill(1)
                else:
                    service_dict[value] = str(index)

            with open("../encoding.py", 'w') as encoding_file:
                encoding_file.write("service_dict = " + json.dumps(service_dict))

    def encode_event_names(self):
        """Fit label encoder
            Parameters
            ----------
            y : array-like of shape=(n_samples,)
                Target values
            Returns
            -------
            self : LabelEncoder
                returns an instance of self
            """
        # Get unique values
        y = self.get_feature_from_dataset("eventname")
        y = np.unique(y)

        # Encode labels
        encoding = dict()
        encoding = {x: i+len(encoding) for i, x in enumerate(set(y) - set(encoding))}

        # generate the encoding dict
        with open("../encoding.py", 'w') as encoding_file:
            encoding_file.write("events_encoding = " + json.dumps(encoding))
            encoding_file.write("\n")

        # generate the decoding dict
        inverse_encoding = {v: k for k, v in encoding.items()}
        with open("../encoding.py", 'a') as encoding_file:
            encoding_file.write("events_decoding = " + json.dumps(inverse_encoding))

            encoding_file.close()

    def fill_vocab_file_for_bert_tokenizer(self):
        # Get unique values
        y = self.get_feature_from_dataset("eventname")
        y = np.unique(y)
        with open("vocab_file", "w") as vocabs:
            for item in y:
                vocabs.write(item + "\n")
        vocabs.close()

    def convert_aws_region_to_int(self, logs_data_frame):
        #  TODO: complete the list of regions
        region_dict = {
            "us-east-1": 1,
            "us-east-2": 2,
            "us-west-1": 3,
            "us-west-2": 4,
        }
        logs_data_frame.replace({"awsregion":region_dict}, inplace=True)
        return logs_data_frame

    def extract_user_identity_features(self):
        self.log_df["principalid"] = self.log_df["useridentity"].apply(lambda x: re.search('principalid=(.+?),', x).group(1))
        self.log_df["accountid"] = self.log_df["useridentity"].apply(lambda x: re.search('accountid=(.+?),', x).group(1))
        self.log_df["invokedby"] = self.log_df["useridentity"].apply(lambda x: re.search('invokedby=(.+?),', x).group(1))

        self.log_df.drop(columns=["useridentity"], inplace=True)

    def extract_request_parameters(self, debug=True):
        self.log_df.requestparameters.fillna("", inplace=True)
        self.log_df["requestparameters"] = self.log_df['requestparameters'].map(lambda x: json.loads(x) if (x) else {})

        req_parameter_col_expanded = pd.json_normalize(self.log_df['requestparameters'])

        #logs_df = logs_df.join(req_parameter_col_expanded, on=['index'])
        self.log_df = pd.concat([self.log_df, req_parameter_col_expanded], axis=1)

        if debug:
            print_yellow(self.log_df.head(5))
            print_yellow(self.log_df.columns)

        self.log_df.drop(columns=["requestparameters"], inplace=True)

    def pre_process_logs_for_encoder(self, dst_dir, dst_file,
                                     feature_list=default_feature_list, debug=True):
        """
        pre-process cloud trail logs in order to:
        - convert aws regions to integer
        - convert event source to integer
        - extract principal ID, account ID and invokedBy from "useridentity" feature of each log
        - extract parameters from requestparameters (json) to individual columns
        """
        self.log_df = pd.read_csv(self.log_file, usecols=feature_list)[feature_list]
        #collected_logs = convert_aws_region_to_int(collected_logs)
        #collected_logs.replace({"eventsource": service_dict}, inplace=True)
        self.extract_user_identity_features()
        try:
            self.extract_request_parameters()
        except AttributeError:
            pass

        if debug:
            print_yellow(self.log_df.columns)
            print_yellow(self.log_df.head(10))

        self.log_df.to_csv(dst_dir+dst_file, sep=' ', header=False, index=False)
        return dst_file

    def get_feature_from_dataset(self, feature, debug=False):
        """
        Query over JSON is not easy in Athena, instead this function
        helps me to get some statistics from the dataset
        """
        #unique_user_ids = (collected_logs["useridentity"].apply(lambda x: re.search('type=(.+?),', x).group(1))).unique()
        unique_feature_list = self.log_df[feature].unique()
        if debug:
            print_green([item for item in unique_feature_list])

        return unique_feature_list

    def get_unique_event_type_called_by_user(self, user_identity_type, debug=True):
        logs_by_this_user = self.log_df.loc(self.log_df['useridentity'].str.findall(user_identity_type))

        unique_event_type = logs_by_this_user["eventtype"].unique()

        if debug:
            print_green([event for event in unique_event_type])

    def get_event_info(self, eventname):
        event_logs = self.log_df.loc[self.log_df["eventname"] == eventname]
        return event_logs

    def get_sequence_of_events(self, source_service):
        # # filter out not readonly events
        # #ro_logs = log_df.loc[log_df['readOnly']]
        #
        # not_ecommerce_event_source = ['cloudtrail.amazonaws.com', 'health..amazonaws.com', 'config.amazonaws.com']
        #
        # sequence_of_events = ()
        # for log in self.log_df.rows():
        #     print_yellow(log['eventSource'])
        #     if log['eventSource'] not in not_ecommerce_event_source:
        #         if "logs" in log['eventSource']:
        #             username = log["useridentity"].apply(
        #                 lambda x: re.search('userName=(.+?),', x).group(1))
        #
        #             print_yellow("Found a log from AWS logs: {0}".format(username))
        #
        #         sequence_of_events.append(log)
        #
        # tuple(sequence_of_events)
        # return sequence_of_events
        return self.log_df.loc[self.log_df["eventsource"] == source_service]

    def get_column_from_dataset(self, column):
        return self.log_df[column]

    def get_all_events_by_src_ip(self, src_ip):
        return list(self.log_df.loc[self.log_df["sourceipaddress"] == src_ip]["eventname"])


def main():
    CLOUD_TRAIL_LOGS_TRAIN = "../dataset/amplify_may_trail_logs.csv"
    CLOUD_TRAIL_LOGS_TRAIN = "../../dataset/awsamplify_ecommerce_sample_1.csv"

    PROCESSED_LOGS_TRAIN = "../dataset/normalized_amplify_may_trail_logs.csv"

    ECOMMERCE_LOGS = "../../dataset/ecommerce_app/6:630-event_history.csv"
    my_session = boto3.Session()
    service_list = my_session.get_available_services()

    aws_parser = AWSLogAnalyzer()
    # index = 1
    # service_dict = {}
    # for item in service_list:
    #     service_dict[item+".amazonaws.com"] = index
    #     index += 1
    # with open("service_list.txt",'w') as file:
    #     file.write(str(service_dict))

    #print(len(service_list))
    #call_api_with_auto_generated_parameters("ec2", "create_model")

    #pre_process_logs_for_encoder(CLOUD_TRAIL_LOGS_TRAIN, PROCESSED_LOGS_TRAIN)

    # unique_user_list = get_feature_from_dataset(CLOUD_TRAIL_LOGS_TRAIN, "useridentity")
    #
    # unique_event_name = get_feature_from_dataset(CLOUD_TRAIL_LOGS_TRAIN, "eventname")
    # print_green("There is {0} unique API calls in the log file".format(len(unique_event_name)))

    # for user in unique_user_list:
    #     print_green("List of event type sources by: {0}".format(user))
    #     get_unique_event_type_called_by_user(CLOUD_TRAIL_LOGS_TRAIN, user)
    # user = "Root"
    # get_unique_event_type_called_by_user(CLOUD_TRAIL_LOGS_TRAIN, user)

    #get_sequence_of_events(ECOMMERCE_LOGS).
    aws_parser.get_feature_from_dataset(CLOUD_TRAIL_LOGS_TRAIN, "eventtype")


if __name__ == "__main__":
    main()
