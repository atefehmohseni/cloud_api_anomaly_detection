
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import scipy.spatial

import json
import os
import torch

from sentence_transformers import SentenceTransformer, util

from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments, AutoModelForMaskedLM
from transformers import pipeline
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer, DistilBertTokenizer, RobertaForMaskedLM, RobertaTokenizerFast, BertModel


from torch.utils.data import DataLoader


from global_variables import service_list_phase_one, print_red,print_green
from encoding import events_decoding, events_encoding


class Transformer(object):
    def __init__(self):
        self.embedder = SentenceTransformer("bert-base-nli-mean-tokens")

    def fill_corpus(self, log_file, log_dir=None, bert_format=False):
        """
        read a log file (each line has a sequence of unique event names called by a single source,
        i.e, one log file per one source in AWS )
        and convert it to a list of unique sequence of logs
        """
        if log_dir is not None: log_file = log_dir+log_file

        corpus = []
        try:
            with open(log_file, 'r') as input_file:
                # read the file and remove repeated lines!
                if bert_format:
                    for line in input_file.readlines():
                        corpus.append(torch.tensor(list(int(num) for num in line.split(' ')), dtype=torch.int))
                else:
                    corpus = set([line.rstrip("\n") for line in input_file.readlines()])
        except Exception as e:
            print("Problem with the input log file due to {0}".format(str(e)))

        return corpus

    # def train_embedder(self, train_data, test_data, evaluator, output_dir):
    #     train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    #
    #     train_objectives = [(train_dataloader, train_loss)],
    #     evaluator = evaluator,
    #     epochs = num_epochs,
    #     evaluation_steps = 1000,
    #     warmup_steps = warmup_steps,
    #     output_path = model_save_path

    def clustering(self, num_clusters, corpus):
        corpus_embeddings = self.embedder.encode(corpus)

        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignments = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignments):
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        for i, cluster in enumerate(clustered_sentences):
            print("Cluster ", i + 1)
            print(cluster)
            print("")

        return clustered_sentences, clustering_model.inertia_

    def plot_square_dist_for_clustering(self, k_range, sum_of_squared_distances, plot_title):
        plt.plot(k_range, sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k SERVICE:{0}'.format(plot_title))
        plt.show()

    def get_the_best_k_for_clustering(self, service, k_max):
        corpus = self.fill_corpus(log_file="../logs/AWS/" + service)
        print("CLUSTERS FOR SERVICE: {}".format(service))

        sum_of_square_dist = []
        k_max = min(k_max, len(corpus))

        for k in range(1, k_max):
            clustered_logs, square_dist = self.clustering(k, list(corpus))
            sum_of_square_dist.append(square_dist)

        self.plot_square_dist_for_clustering(k_range=range(1, k_max),
                                             sum_of_squared_distances=sum_of_square_dist, plot_title=service)

    def semantic_search(self, corpus, query_list, n_similarities=5):
        corpus_embeddings = self.embedder.encode(corpus)
        query_embeddings = self.embedder.encode(query_list)

        if query_list is None:
            return None

        for query, query_embedding in zip(query_list, query_embeddings):
            dist = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(dist)), dist)
            results = sorted(results, key=lambda x: x[1])

            # print("\n\n======================\n\n")
            # print("Query:", query)
            # print("\nTop {0} most similar sentences in corpus:".format(n_similarities))
            #
            # for idx, distance in results[0:n_similarities]:
            #     print(corpus[idx].strip(), "(Score: %.4f)" % (1 - distance))

        return results[:n_similarities]

    def cluster_events_and_group_similar_sequences(self, list_of_services, all_events_corpus):
        cluster_dict = {}
        workflows = []
        for service in list_of_services:
            if service in service_list_phase_one.keys():
                corpus = self.fill_corpus(log_file="../logs/AWS/" + service)

                print_green("CLUSTERS FOR SERVICE: {}".format(service))
                cluster_dict[service] = []
                clustered_logs, square_dist = self.clustering(service_list_phase_one[service], list(corpus))

                for cluster in clustered_logs:
                    query = cluster
                    similarity_results = self.semantic_search(all_events_corpus, query, n_similarities=1)

                    for idx, distance in similarity_results[0:1]:
                        if (1 - distance) >= 0.85:
                            flow = [event for event in list(all_events_corpus)[idx].split(' ')]
                            workflows.append(flow)
                            cluster_dict[service].append(
                                {"nodes":flow ,
                                 "Score": "%.4f" % (1 - distance)})
                            print([events_decoding[event] for event in list(all_events_corpus)[idx].split(' ')],
                                  "(Score: %.4f)" % (1 - distance))

        print_green(json.dumps(cluster_dict))
        dst_file = "clusters.py"
        with open(dst_file, "w") as cluster_file:
            cluster_file.write("clusters=" + json.dumps(cluster_dict))

        print_red("\n\n Done with clustering!")

        return workflows

    def test_train_split(self, input_dataset, train_ratio):
        train_size = int(train_ratio * len(input_dataset))
        test_size = len(input_dataset) - train_size
        train, test = torch.utils.data.random_split(input_dataset, [train_size, test_size])

        train_dataset = []
        for idx, my_list in enumerate(train.dataset):
            if idx in list(train.indices):
                train_dataset.append(my_list)
        test_dataset = []
        for idx, my_list in enumerate(test.dataset):
            if idx in list(test.indices):
                test_dataset.append(my_list)

        return train_dataset, test_dataset

    def evaluate_coverage(self, workflows, test_data, similarity_ratio=0.90):
        """
        this function evaluates the coverage of detected workflows (from logs) on the sample test dataset
        """
        tot_test_case = len(test_data)
        covered = 0
        for query in test_data:
            similarity_results = self.semantic_search(workflows, query, n_similarities=1)
            print("\n\n======================\n\n")
            print("test data: {0}".format([events_decoding[event] for event in query.split(' ')]))
            for idx, distance in similarity_results[0:1]:
                if (1 - distance) >= similarity_ratio:
                    covered = covered + 1
                    print([events_decoding[event] for event in list(workflows)[idx].split(' ')],
                          "(Score: %.4f)" % (1 - distance))

        print("Coverage: %{0} \n covered:{1}/{2}".format((covered/tot_test_case)*100, covered, tot_test_case))
        return covered, tot_test_case


class BertTransformer(object):
    def __init_(self):
        pass

    def load_and_tokenize(self, file_path, tokenizer):
        """
        load a sequential text dataset and tokenize it line by line
        sample of file:
        event1, event2, event3
        event5, event3, event7
        ...
        """
        if not os.path.exists(file_path):
            print_red("File {0} not exists. Cannot find the input dataset file to tokenize.".format(file_path))
            return None

        tokenized_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=10)
        return tokenized_dataset

    def tokenize_logs(self, tokenizer, input_file):
        if not isinstance(tokenizer, BertTokenizer):
            return None
        with open(input_file, "r") as logs:
            tokenized_logs = tokenizer([line.strip() for line in logs.readlines()], padding=True, truncation=True,
                                       return_tensors="pt")

        return tokenized_logs

    def set_tokenizer(self, vocab_file="vocab_file", do_lower_case=True, do_basic_tokenize=False):
        print("current directory: {0}".format(os.getcwd()))
        with open(vocab_file, 'r') as vocabs:
            words_list = list(vocabs.readlines())
            my_tokenizer = BertTokenizer(vocab_file, do_lower_case=do_lower_case, do_basic_tokenize=do_basic_tokenize,
                                     never_split=words_list, unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]',
                                     cls_token='[CLS]', mask_token='[MASK]', tokenize_chinese_chars=False,
                                     add_special_tokens=False,)#special_tokens=["[CLS]", "[UNK]", "[PAD]", "[MASK]"]

        # pretrained_weights = 'roberta-base'
        # my_tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_weights)

        # pretrained_weights = "bert-base-uncased"
        # my_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        return my_tokenizer

    def set_training_arguments(self, output_dir, overwrite_output_dir=True, train_epochs=5):
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            num_train_epochs=train_epochs,
            per_device_train_batch_size=64,
            # save_steps=10_000,
            # save_total_limit=2,
        )
        return training_args

    def set_trainer(self, training_args, train_data_set, test_data_set, device, tokenizer=events_decoding):  # model,
        #model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        #model = AutoModelForMaskedLM.from_pretrained("roberta-base")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

        model.to(device)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_data_set,
            eval_dataset=test_data_set,
            prediction_loss_only=True,
        )
        return trainer

    def start_training_and_save_model(self, trainer, output_dir, model_name):
        if isinstance(trainer, Trainer) is False:
            print_red("The input trainer is not a valid Trainer object!")
            return False
        trainer.train()
        eval_res = trainer.evaluate()
        trainer.save_model(output_dir + model_name)
        print_green("EVALUATION RESULT:\n{0}".format(eval_res))
        return True

    def evaluate_model(self, model, train, test):
        model = BertModel.from_pretrained(model)
        print(model.eval())
        eval_params = model.evaluate()
        return eval_params

    def fill_mask_per_event(self, model, event_tokenizer):
        nlp_fill = pipeline('fill-mask', model=model, tokenizer=event_tokenizer)
        #res = nlp_fill('GetTrailStatus DescribeEventAggregates [MASK] DescribeTrails')#.format({nlp_fill.tokenizer.mask_token}))

        res = nlp_fill("SignUp [MASK]")#ConfirmSignUp
        print("Predicted events:")
        for item in res:
            print(item)

    def test_bert_auto_tokenizer(self):
        from transformers import AutoModelWithLMHead, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
        sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."
        input = tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
        token_logits = model(input)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

        for token in top_5_tokens:
            print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

    def test_train_split(self, input_dataset, train_ratio):
        train_size = int(train_ratio * len(input_dataset))
        test_size = len(input_dataset) - train_size
        train, test = torch.utils.data.random_split(input_dataset, [train_size, test_size])
        return train, test
        # train_dataset = []
        # test_dataset = []
        #
        # for idx, my_list in enumerate(train.dataset):
        #     if idx in list(train.indices):
        #         train_dataset.append(my_list)
        #     if idx in list(test.indices):
        #         test_dataset.append(my_list)
        # print("test")
        #return torch.index_select(train.dataset, 0, train.indices), torch.index_select(test.dataset, 0, test.indices)
