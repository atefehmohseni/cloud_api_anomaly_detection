import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ast import literal_eval

import json
import pandas as pd
import numpy as np
import re
import hashlib

from global_variables import API_PARAMETERS_FILE
from global_variables import print_green, print_yellow, print_red
from global_variables import service_dict

from beautifultable import BeautifulTable


def get_logs_per_service(log_df, service_name, debug=False):
    """
    Query over JSON is not easy in Athena, instead this function
    helps me to get some statistics from the dataset
    """
    if debug:
        print_yellow(log_df.columns.tolist())

    unique_events_per_service = (log_df.loc[log_df["eventsource"].isin(service_name)])#.unique()

    return unique_events_per_service


def get_logs_per_time(log_file, start_time=None, end_time=None, debug=False):
    collected_logs = pd.read_csv(log_file)

    if debug:
        print_green("Original Schema from Cloudtrail: {0}".format(collected_logs.info()))

    if start_time and end_time:
        collected_logs = collected_logs.astype({'eventtime': 'datetime64'}).where((collected_logs["eventtime"] >= start_time) & (collected_logs["eventtime"] <= end_time))
        if debug:
            print_yellow("Schema of selected time-based logs: {0}".format(collected_logs.info()))
    else:
        collected_logs = collected_logs.astype({'eventtime': 'datetime64'})

    return collected_logs


def plot_time_based_events(service_logs_df, dst_file, debug=False):
    if debug:
        print_yellow(service_logs_df.columns)
    fig, ax = plt.subplots()
    fig.autofmt_xdate()

    xfmt = mdates.DateFormatter('%D:%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    x_axes = service_logs_df["eventtime"]#.astype('int')

    get_involved_services = service_logs_df["eventsource"].unique()

    y_axes = []
    for service in get_involved_services:
        event_list = service_logs_df.eventname.where(service_logs_df.eventsource == service).values.tolist()
        y_axes.append({"service": service, "events": event_list})

    for item in y_axes:
        ax.plot(x_axes, item["events"], "*", label=item["service"])

    ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.xlabel("Time")
    plt.ylabel("API call")

    duration = get_duration_of_log_file(service_logs_df)
    plt.savefig(dst_file+str(duration)+".pdf", bbox_inches='tight')
    plt.show()


def plot_corr(corr_matrix, df, threshold=0.5):
    f = plt.figure(figsize=(20, 15))
    #f, axes = plt.subplots(len(thresholds), sharex=True, sharey=True)
    # mask = np.tri(corr_matrix.shape[0],k=-1)
    # new_corr_matrix = np.ma.array(corr_matrix, mask=mask)
    #for i, ax in enumerate(axes):
    plt.matshow(corr_matrix[corr_matrix >= threshold], interpolation='nearest')
    plt.xticks(range(df.shape[1]), df.columns, fontsize=12, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    plt.xlabel('Highly correlated features', fontsize=16,)

    plt.show()

    # sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #        .stack()
    #        .sort_values(ascending=False))


def get_shared_parameters_statistics_per_service(log_df, txt_stat_file, service_name, debug=False):
    unique_events = log_df.eventname.unique()
    shared_parameters = {}
    for event in unique_events:
        try:
            req_params = log_df.requestparameters.where(log_df.eventname == event).dropna().iloc[0]
        except IndexError as e:
            print_yellow("Event {0} has no request parameters. details exception: {1}".format(event,str(e)))
        try:
            if req_params:
                request_parameters = json.loads(req_params)
                for param in request_parameters:
                    if param not in shared_parameters.keys():
                        shared_parameters[param] = {"count": 1}
                    else:
                        shared_parameters[param]["count"] += 1
                        shared_parameters[param]["type"] = type(request_parameters[param])
                        shared_parameters[param]["sample_value"] = request_parameters[param]
        except Exception as e:
            print_red("An exception occurred due to: {0}".format(str(e)))

    with open(txt_stat_file, "a") as stat_file:
        stat_file.write("service\tshared parameter's name\tpercent\tdescription\tparameter's type\tparam's sample value\n")

        for param in shared_parameters.keys():
            if shared_parameters[param]["count"] > 1:
                shared_percent = (shared_parameters[param]["count"]/len(unique_events))*100
                stat_file.write("{0}\t{1}\t{2}\t{3}out of{4}\t{5}\t{6}\n".format(service_name,param, shared_percent, shared_parameters[param]["count"],
                        len(unique_events), shared_parameters[param]["type"], shared_parameters[param]["sample_value"]))


def create_table_from_file(txt_file):
    with open(txt_file,'r') as raw_data:
        records = raw_data.readlines()
        table = BeautifulTable()
        for i,record in enumerate(records):
            if i == 0:
                table.column_headers = record.split("\t")
            else:
                table.append_row(record.split("\t"))

    print_green(table)


def get_statistics_among_diff_services(log_df, parameter, debug=False):
    #  get the number of non-NaN values per parameter
    tot_num_records = log_df[parameter].count()

    unique_parameters_value = log_df[parameter].unique()

    if debug:
        print_green(unique_parameters_value)

    stat = []
    for item in unique_parameters_value:
        item_records_count = log_df[parameter].where(log_df[parameter] == item).count()
        stat.append({
            "parameter-value": item,
            "percentage": item_records_count/tot_num_records*100,
            "parameter-counter": item_records_count,
            "total": tot_num_records
        })

    return stat


def get_user_identity_stat_among_diff_services(log_df, attribute, debug=True):
    tot_user_ids = log_df.useridentity.count()
    user_types = (log_df["useridentity"].apply(lambda x: re.search(attribute + '=(.+?),', x).group(1))).dropna().value_counts()

    for index, value in user_types.items():
        print_green("{0}: {1}\t{2} out of {3}\tpercentage: {4}".format(attribute, index,
                                                                       value, tot_user_ids,
                                                                       (value/tot_user_ids)*100))

    return user_types


def analyse_anonymous_principal_id(log_df, service_names):
    #service_logs = log_df.where((log_df.useridentity.str.contains("[+<principalid>(?=)?<null>?<Anonymous>]", regex=True))).dropna(how='all')

    for service_name in service_names:
        service_logs = log_df.where((log_df.eventsource == service_name) &(log_df.useridentity.str.contains("Anonymous"))).dropna(how='all')

        event_list = np.unique(service_logs.eventname)
        print_yellow("List of events for the service {0} with anonymous principal-id:\n{1}\nlength:{2}".format(service_name, event_list,len(event_list)))


def analyse_unknown_useridentity_type(log_df, service_names):
    for service_name in service_names:
        service_logs = log_df.where((log_df.eventsource == service_name) &(log_df.useridentity.str.contains("Unknown"))).dropna(how='all')

        event_list = np.unique(service_logs.eventname)
        print_yellow("List of events for the service {0} with Unknown user type:\n{1}\nlength:{2}".format(service_name, event_list,len(event_list)))


def extract_user_identity_json_inside(src_df):
    def extract_user_identity(x):
        new_json = {}
        params = x.split(",")
        for item in params:
            variables = item.split("=")
            new_json[variables[0].strip()] = variables[1].strip()
        return new_json

    src_df.useridentity = src_df.useridentity.apply(extract_user_identity)
    return src_df


def convert_user_id_string_parameters_to_int(logs_data_frame):
    def convert_to_int(x):
        return int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16) % 10 ** 8

    # need_normalization_types = [str, object]
    # allowed_data_types = [float, int]
    # mask = logs_data_frame.dtypes.isin(allowed_data_types)
    # logs_data_frame = logs_data_frame.loc[:, mask].apply(convert_to_int)
    #logs_data_frame.apply(pd.to_numeric, errors="ignore", downcast="integer")

    return logs_data_frame.applymap(hash)


def get_correlation_matrix(df, debug=False):
    df = extract_user_identity_json_inside(df)
    new_user_identity = pd.json_normalize(df['useridentity'])
    df.drop("useridentity", axis=1, inplace=True)

    preprocessed_df = pd.concat([df, new_user_identity], axis=1)
    if debug:
        print_green("Data frame Schema BEFORE normalization:\n{0}".format(preprocessed_df.dtypes))

    normalized_df = convert_user_id_string_parameters_to_int(preprocessed_df)

    if debug:
        print_green("Data frame Schema AFTER normalization:\n{0}".format(normalized_df.dtypes))

    cor_max = normalized_df[normalized_df.columns].corr(method="pearson")#.abs() #spearman kendall pearson
    if debug:
        print_green("The correlation matrix:\n{0}".format(cor_max))

    return cor_max, normalized_df


def get_duration_of_log_file(log_df):
    time_records = log_df.eventtime
    # print_yellow(time_records.max())
    # print_yellow(time_records.min())
    duration = time_records.max()-time_records.min()
    print_green("Duration of the log report: {0}".format(duration))
    return duration


def pretty_print(input_list):
    """
    get a list of dictionaries and print them into a table format
    """
    table = BeautifulTable()
    header = input_list[0].keys()
    np_array = np.array(input_list)
    for item in input_list:
        print(item)
        #table.insert_column(np_array[item])

    print_green(table)


def main():
    log_file = "../dataset/ecommerce_app/seclab_all_ecommerce_trail_logs_athena.csv"
    log_file = "../dataset/ecommerce_app/ecomm_all_may27_jun15.csv"
    stat_file = "../dataset/results/shared_param_statistics.txt"
    plot_folder = "../dataset/results/"
    # log in no pinpoint (appearntly)
    #log_df = get_logs_per_time(log_file, start_time="2020-05-30T01:01:47Z", end_time="2020-05-30T01:09:12Z")

    # log in + pinpoint
    #log_df = get_logs_per_time(log_file, start_time="2020-05-29T19:39:49Z", end_time="2020-05-29T21:28:31Z")

    ecommerce_services = ["apigateway.amazonaws.com", "dynamodb.amazonaws.com", "pinpoint.amazonaws.com", "s3.amazonaws.com","cognito.amazonaws.com",
                          "cognito-idp.amazonaws.com", "cognito-identity.amazonaws.com", "logs.amazonaws.com","amplify.amazonaws.com","lambda.amazonaws.com","glue.amazonaws.com"]

    cognito_events = ["cognito.amazonaws.com", "cognito-idp.amazonaws.com", "cognito-identity.amazonaws.com"]

    service_name = ["s3.amazonaws.com", ]

    log_df = get_logs_per_time(log_file)

    #get_duration_of_log_file(log_df)

    # for service in ecommerce_services:
    #     service_logs = get_logs_per_service(log_df, [service,])
    #     get_shared_parameters_statistics_per_service(service_logs, stat_file, service_name=service)

    #create_table_from_file(stat_file)

    service_logs = get_logs_per_service(log_df, service_name, debug=True)
    # for item in service_logs:
    #     print(item)

    #plot_time_based_events(service_logs, plot_folder, debug=False)

    # analyse_anonymous_principal_id(service_logs, ecommerce_services)
    # analyse_unknown_useridentity_type(service_logs, ecommerce_services)

    corr_matrix, service_logs = get_correlation_matrix(service_logs)
    plot_corr(corr_matrix, service_logs, threshold=0.5)
    plot_corr(corr_matrix, service_logs, threshold=0.75)
    plot_corr(corr_matrix, service_logs, threshold=1)

    #
    # parameters_list = ["useragent", "sourceipaddress", "eventsource", "readonly", "eventtype"]
    #
    # for par in parameters_list:
    #     result = get_statistics_among_diff_services(service_logs, parameter=par)
    #     print_red(par)
    #     pretty_print(result)
    #
    # user_identity_attributes = ["type","accesskeyid",]
    # for att in user_identity_attributes:
    #     get_user_identity_stat_among_diff_services(service_logs,att)


if __name__ == "__main__":
    main()
