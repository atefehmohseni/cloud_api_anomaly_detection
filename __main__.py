import os
import torch

from preprocessing import Parser
from transformer import Transformer

from global_variables import print_red
from encoding import events_decoding
from aws_scripts.process_aws_api import AWSLogAnalyzer

from transformer import BertTransformer


def new_data_set_parameter_extraction(aws_logs):
    # call this once to generate 'encoding' script including unique events in the log file and their code!
    aws_agent = AWSLogAnalyzer(aws_logs, '../logs/AWS/', read_all_file=True)


def generate_fake_event_list(aws_logs, n=10):
    for i in range(n):
        l_parser = Parser()
        all_events_file = l_parser.parse_logs_into_unique_seq_events(aws_logs)


def manual_analysis(aws_logs, offline=False):
    l_parser = Parser()
    all_events_file = l_parser.parse_logs_into_unique_seq_events(aws_logs)
    # generate list of unique events per line called by the same source (~ AWS service)
    list_of_services = l_parser.parse_logs_into_source_based_sequence_events(aws_logs)

    my_transformer = Transformer()
    all_event_corpus = my_transformer.fill_corpus(all_events_file, bert_format=False)

    train, test = my_transformer.test_train_split(all_event_corpus, train_ratio=0.8)

    # get the best k for each cluster (run this part once)
    # for service in list_of_services:
    #     my_transformer.get_the_best_k_for_clustering(service, k_max=13)

    if offline:
        workflows = []
        if os.path.exists("clusters.py"):
            from clusters import clusters
            for service in clusters:
                for event_list in clusters[service]:
                    workflows.append(' '.join(event_list["nodes"]))
    else:
        workflows = my_transformer.cluster_events_and_group_similar_sequences(list_of_services, train)

    my_transformer.evaluate_coverage(workflows, test, similarity_ratio=0.92)


def bert_analysis(aws_logs):
    l_parser = Parser()

    bert_transformer = BertTransformer()

    all_events_file = l_parser.parse_logs_into_unique_seq_events(aws_logs, bert_format=True)
    # all_events_corpus = my_transformer.fill_corpus(all_events_file, bert_format=False)

    all_events_tokenizer = bert_transformer.set_tokenizer()

    dataset = bert_transformer.load_and_tokenize(all_events_file, all_events_tokenizer)
    print("After load the tokens into a dataset")
    ##dataset = bert_transformer.tokenize_logs(all_events_tokenizer, all_events_file)
    print("BEfore split")
    train, test = bert_transformer.test_train_split(dataset, train_ratio=0.8)
    print("After split")

    training_args = bert_transformer.set_training_arguments(output_dir="./models/")
    print("After set_training_arguments")

    trainer = bert_transformer.set_trainer(training_args, train, test, device="cpu",tokenizer=all_events_tokenizer)
    print("After set_trainer")

    bert_transformer.start_training_and_save_model(trainer, "./models/", dataset)


def bert_performance(aws_logs):
    bert_transformer = BertTransformer()
    l_parser = Parser()

    all_events_file = l_parser.parse_logs_into_unique_seq_events(aws_logs, bert_format=True)

    all_events_tokenizer = bert_transformer.set_tokenizer()
    dataset = bert_transformer.load_and_tokenize(all_events_file, all_events_tokenizer)
    train, test = bert_transformer.test_train_split(dataset, train_ratio=0.6)

    bert_transformer.evaluate_model(model="./models/base_bert_v3", train=train, test=test)


def bert_prediction():
    bert_transformer = BertTransformer()
    all_events_tokenizer = bert_transformer.set_tokenizer()
    bert_transformer.fill_mask_per_event(model="./models/base_bert_v1_gpu", event_tokenizer=all_events_tokenizer)


def user_analysis(aws_logs, dst_log_file):
    aws_agent = AWSLogAnalyzer(aws_logs, '../logs/AWS/', read_all_file=True)
    unique_assume_role = aws_agent.get_event_info("AssumeRoleWithWebIdentity")
    #unique_assume_role = aws_agent.get_event_info("SignUp")

    init_auth_counter = aws_agent.get_event_info("initiateauth")

    print("Number of assume-role logs: {0}\nNumber of init-auth logs: {1}".format(len(unique_assume_role), len(init_auth_counter)))

    print("number of all unique IP address:{0}".format(len(unique_assume_role["sourceipaddress"].unique())))

    assume_role_ips = unique_assume_role["sourceipaddress"].unique()

    ip_list = aws_agent.get_feature_from_dataset("sourceipaddress")
    #print(ip_list)
    print("number of all IP address:{0}".format(len(ip_list)))

    workflows = []
    for ip in ip_list:
        events = aws_agent.get_all_events_by_src_ip(ip)
        if events not in workflows:
            #events.insert(0, ip)
            workflows.append(events)

    events_per_line = []
    for flow in workflows:
        print(flow)
        print("*******")
        for event in flow:
            if event not in events_per_line:
                events_per_line.append(event)
            else:
                with open(dst_log_file, 'a') as output:
                    # pickle.dump(events_per_line, output)
                    output.write(' '.join(events_per_line) + "\n")
                    output.close()

                events_per_line.clear()
                events_per_line.append(event)
    return dst_log_file


def bert_analysis_with_gpu(aws_log_file):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # specify which GPU(s) to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    bert_transformer = BertTransformer()
    all_events_tokenizer = bert_transformer.set_tokenizer()
    dataset = bert_transformer.load_and_tokenize(aws_log_file, all_events_tokenizer)
    train, test = bert_transformer.test_train_split(dataset, train_ratio=0.6)

    training_args = bert_transformer.set_training_arguments(output_dir="./models/")
    print("After set_training_arguments")

    trainer = bert_transformer.set_trainer(training_args, train, test,
                                           device=device, tokenizer=all_events_tokenizer)
    print("After set_trainer")

    bert_transformer.start_training_and_save_model(trainer, "./models/", model_name="base_bert_v1_gpu")

    #bert_transformer.evaluate_model(model="./models/base_bert_v3", train=train, test=test)


def main():
    #aws_logs = "++signup_auto_gen_july21_22_athena.csv"
    aws_logs = "+aws-ecomm-signup-logs-july-31-100users.csv"

    aws_logs = "aws-auto-signup-200plus.csv"

    #new_data_set_parameter_extraction(aws_logs)

    #manual_analysis(aws_logs, offline=True)

    #bert_analysis(aws_logs="manually_grouped_logs")
    #generate_fake_event_list(aws_logs, n=100)

    #bert_performance(aws_logs)

    # for i in range(20):
    #     user_analysis(aws_logs, "manually_grouped_logs")

    #bert_analysis_with_gpu("manually_grouped_logs")
    bert_prediction()


if __name__ == "__main__":
    main()
