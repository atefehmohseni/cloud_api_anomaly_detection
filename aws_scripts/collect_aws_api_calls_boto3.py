import boto3
import re
import json
import inspect

from botocore.exceptions import ParamValidationError
from inspect import signature
from inspect import getcallargs
from timeout import timeout

from global_variables import API_PARAMETERS_FILE
from global_variables import print_red, print_green,print_yellow

def _get_args_dict(fn, args, kwargs):
    args_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    return {**dict(zip(args_names, args)), **kwargs}


def test_samples(service_name, method):
    """
    To run one single method from aws cli available services
    """
    client = boto3.client(service_name)
    func = getattr(client, method)
    sig = signature(func)
    print_green(inspect.getfullargspec(func))
    print_green(getcallargs(func))
    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            print(param)
    print_green(sig.parameters["kwargs"].annotation)
    print(sig)

    #res = _get_args_dict(func,sig.parameters["args"],sig.parameters["kwargs"])
    #print(res)
    #
    # for par in sig.parameters.values():
    #     print(sig.parameters["kwargs"])
    try:
        call_method_with_time_out(func)
        #func()
    except Exception as e:
        print(e)


@timeout(10)
def call_method_with_time_out(method):
    """
    Note: some of the methods doesnt get any response back (not response nor error/exception)
    to prevent the program stuck at these function, there is a timeout defined
    """
    required_params = []
    try:
        method()
    except TimeoutError as e:
        print_green("Time out: " + str(method.__name__))
        required_params = ["timeout"]
    except ParamValidationError as e:
        print_green(str(e))
        required_params = re.findall(r'\"(.+?)\"', str(e))
    except Exception as e:
        # Can't handle methods with required arguments.
        required_params = ["N/P"]
    return required_params


def get_available_services_parameters(service_list):
    """
    Collect the list of available services (according to this session and boto3)
    plus the required parameters to methods
    Output: Json contains available services(e.g, ec2) as keys and actions (methods) as a value
    each method has a list of required parameters.
    """
    #banned_fucntions = ["describe_domains","list_domain_names","delete_report_definition","describe_report_definitions"]#cloudserach,,cur,
    #banned_services = ["alexaforbusiness"]
    api_list = {}
    method_description = {}
    for li in service_list:
        print_red(li)
        rsc = boto3.client(li)
        attrs = (getattr(rsc, name) for name in dir(rsc) if not name.startswith("_") and not name.startswith("__"))
        methods = filter(inspect.ismethod, attrs)

        for method in methods:
            print_yellow(method.__name__)
            required_params = []
            required_params = call_method_with_time_out(method)
            print_green(required_params)
            method_description[method.__name__] = required_params

        api_list[li] = json.dumps(method_description)
        method_description = {}

    with open(API_PARAMETERS_FILE, 'w') as file:
        json.dump(api_list, file)


def read_api_calls_file():
    with open(API_PARAMETERS_FILE, "r") as log_file:
        data = json.load(log_file)

    return data


def get_method_parameters(api, method):
    api_calls = read_api_calls_file()

    api_methods = json.loads(api_calls[api])
    parameters = api_methods[method]
    print_yellow("Method {0}'s parameters: {1}'".format(method,parameters))

    return parameters


def get_number_of_non_defined_params():
    api_calls = read_api_calls_file()
    failed_counter = 0
    total_methods = 0
    for api in api_calls.keys():
        api_methods = json.loads(api_calls[api])
        for method in api_methods.keys():
            total_methods += 1
            if "N/P" in api_methods[method]:
                failed_counter += 1
                print_yellow("Service {0}: method: {1}".format(api, method))
    return total_methods, failed_counter


def call_service_with_parameters(api, method=None, parameters=None):
    client = boto3.client(api)

    #func = getattr(client, method)

    response = client.create_model(
    restApiId='7u9p0rr810',
    name='testapi',
    description='string',
    schema="",
    contentType='application/json'
    )

    response = client.get_method(
    restApiId='7u9p0rr8b7',
    resourceId='xn1gzwrdel',
    httpMethod='POST'
    )

    return response


def main():
    mysession = boto3.Session()
    service_list = mysession.get_available_services()

    #test_samples("apigateway","create_model")
    #get_available_services_parameters(service_list)

    #get_method_parameters("alexaforbusiness", "associate_device_with_room")

    #get_method_parameters("apigateway",)

    tot_methods, failed_params_retrive = get_number_of_non_defined_params()
    print_red("Total {0}, Failed attempts: {1}".format(tot_methods,failed_params_retrive))

    # res = call_service_with_parameters(api="apigateway")
    # print(res)


if __name__ == "__main__":
    main()
