import sys
import hashlib
import torch
import re
import pickle
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


from logparser import Spell
from aws_scripts.process_aws_api import AWSLogAnalyzer
from global_variables import default_feature_list, feature_list, aws_logs_to_sentence_format
from encoding import events_encoding

sys.path.append('../')


class Parser(object):
    """
    Object for parsing an unstructured log file
    """
    def __init__(self):
        self.log_formats = {"HDFS": "<Date> <Time> <Pid> <Level> <Component>: <Content>",

                            "OpenStack": "<SRC> <Date> <Time> <Instance_ID> <Level> <Generator> <Component>: <Content>",

                            "AWS": "<eventTime> <eventSource> <eventName> <awsRegion> <principalid> <Component>: <Content>",
                            }

    def parse_logs(self, log_file, log_format):
        """
        Parse the logs using Spell method

        log_file: The input log file name
        """
        input_dir = '../logs/'+log_format+"/"  # The input directory of log file
        output_dir = 'Spell_result/'  # The output directory of parsing results
        tau = 0.5  # Message type threshold (default: 0.5)
        regex = []  # Regular expression list for optional preprocessing (default: [])

        if log_format == "AWS":
            aws_agent = AWSLogAnalyzer(log_file, input_dir, read_all_file=False)
            #  convert AWS CloudTrail logs into a "logparser" readable format
            log_file = aws_agent.pre_process_logs_for_encoder(dst_dir=input_dir, dst_file="aws_preprocessed.log", debug=False)

        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=self.log_formats[log_format], tau=tau, rex=regex)
        parsed_logs = parser.parse(log_file)
        return parsed_logs

    def parse_logs_into_english_text(self, log_dir, log_file, log_format=feature_list, sentence_format=aws_logs_to_sentence_format):
        """
        read an unstructured log file and convert it to a meaningful English text
        """

        # read logs from file and extract fields according to the log_format
        pre_processed_logs = pd.read_csv(log_dir+log_file, usecols=log_format)[log_format]

        # convert formatted logs into a english essay using sentence_format
        with open(log_dir+"english_logs.txt", 'w') as file:
            for index, log in pre_processed_logs.iterrows():
                log = re.sub(r'<<({})>>'.format("|".join(sorted(feature_list, reverse=True))), lambda x: log[x.group(1)], sentence_format)
                file.write(log+"\n")

        file.close()

        return log_dir+"english_logs.txt"

    def parse_logs_into_source_based_sequence_events(self, raw_log_file):
        """
        Prase AWS logs into sequence-based events such that
        each line contains unique events called by the same source (happened in a row)

        output: a list of parsed-log files per event source
        """
        input_dir = '../logs/AWS/'  # The input directory of log file

        aws_agent = AWSLogAnalyzer(raw_log_file, input_dir, read_all_file=True)
        unique_sources = aws_agent.get_feature_from_dataset("eventsource")

        for service in unique_sources:
            output_file = input_dir + service
            event_list = aws_agent.get_sequence_of_events(source_service=service)

            events_per_line = []
            for index, log in event_list.iterrows():
                event = str(events_encoding[log["eventname"]])
                #event_code = int(hashlib.sha256(event.encode('utf-8')).hexdigest(), 16) % 10 ** 8
                if event not in events_per_line:
                    events_per_line.append(event)
                else:
                    with open(output_file, 'a') as output:
                        #pickle.dump(events_per_line, output)
                        output.write(' '.join(events_per_line)+"\n")
                        output.close()

                    events_per_line.clear()
                    events_per_line.append(event)

            # in case that there is only one event from a service
            if events_per_line:
                with open(output_file, 'a') as output:
                        #pickle.dump(events_per_line, output)
                        output.write(' '.join(events_per_line)+"\n")
                        output.close()

        return unique_sources

    def parse_logs_into_unique_seq_events(self, raw_log_file, input_dir=None, bert_format=False):
        """
               Prase AWS logs into a sequence-based events such that
               each line contains unique events happened in a row

               output: a list of parsed-log files
               """
        if input_dir is None:
            input_dir = '../logs/AWS/'  # The input directory of log file

        aws_agent = AWSLogAnalyzer(raw_log_file, input_dir, read_all_file=True)
        event_names_list = aws_agent.get_column_from_dataset("eventname")

        output_file = input_dir + "event_list"

        events_per_line = []
        for event in event_names_list:
            if not bert_format:
                event = str(events_encoding[event])
            if event not in events_per_line:
                events_per_line.append(event)
            else:
                with open(output_file, 'a') as output:
                    # pickle.dump(events_per_line, output)
                    output.write(' '.join(events_per_line) + "\n")
                    output.close()

                events_per_line.clear()
                events_per_line.append(event)

        return output_file

    # def digitize_string_logs(self, log_df):
    #     """
    #     Convert string-based logs into a sequence of integers (to feed into LSTM/BERT)
    #     """
    #     def convert_to_int(x):
    #         return int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16) % 10 ** 8
    #
    #     return log_df.applymap(hash)

class BertPreprocessor(object):
    """
    Object for parsing the structured logs and returs torch tensors of tokenized input
    """
    def __init__(self, input_log_file,):
        with open(input_log_file,'r') as input_logs:
            self.sentences = input_logs.readlines()
        self.input_ids = []
        self.attention_mask = []

    def tokenize(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenized_text = [tokenizer.tokenize(sentence) for sentence in self.sentences]
        print("Samples of tokenized sentences {0}".format(tokenized_text[:5]))
        self.input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
        self.input_ids = pad_sequences(self.input_ids, maxlen=128,dtype="float", truncating="post", padding="post")

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in self.input_ids:
            seq_mask = [float(i > 0) for i in seq]
            self.attention_mask.append(seq_mask)

    def train_test_splitter(self):
        self.tokenize()
        input_train, input_test = train_test_split(self.input_ids, test_size=0.1, random_state=1372)

        mask_train, mask_test = train_test_split(self.attention_mask, test_size=0.1, random_state=1372)

        input_train = torch.tensor(input_train)
        input_test = torch.tensor(input_test)
        mask_train = torch.tensor(mask_train)
        mask_test = torch.tensor(mask_test)

        return input_train, input_test, mask_train, mask_test


