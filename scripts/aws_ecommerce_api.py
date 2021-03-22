import requests
requests.adapters.DEFAULT_RETRIES = 2

import json
import re
import time
import subprocess
import random
import argparse
from argformat import StructuredFormatter

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

global email_handler

from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.keys import Keys
from pythonping import ping

from global_variables import print_red, print_green, print_yellow
from scripts.MailHandler import EMailHandler

main_url = "https://cognito-idp.us-east-1.amazonaws.com/"
client_id = "3t8fc6j18fg00i7tqtbdp0187d"
browser_url = "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com/"

# create an email handler instance for verification part

mailinator_authorization_token = "fe418749652740b5a448a9af2000093e" # old (@ucsb) "81d9ca35ed0c4426aee5cb249154ad53"

from lxml.html import fromstring
from itertools import cycle


def is_open(address):
    res = ping(address, count=2)
    return res.success()


def get_proxies(n=10):
    # url = 'https://free-proxy-list.net/'
    # response = requests.get(url)
    # parser = fromstring(response.text)
    # proxies = set()
    # for i in parser.xpath('//tbody/tr')[:n]:
    #     if i.xpath('.//td[7][contains(text(),"yes")]'):
    #         proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
    #         proxies.add("http://"+proxy)

    proxies = [
                "http://201.8.210.156:8080",
                "http://130.61.95.193:3128",
                "http://95.79.55.196:53281",

                "http://188.191.123.69:58170", "http://158.69.62.249:3128",
              ]
    not_working = ["http://193.169.20.98:8536", "http://185.232.66.123:5836",
                   "http://185.232.66.124:5836",  "http://185.232.66.127:5836", "http://185.232.66.126:5836",
                   "http://61.7.145.190:8080", "http://188.242.63.131:8080", "http://96.9.67.84:8080",
                   "http://103.21.161.105:6666", "http://188.168.27.71:36733","http://212.19.6.91:8888",
                   "http://195.225.172.208:41273",  "http://118.172.51.84:43147", "http://110.76.128.53:42670"
                   ]
    return proxies


def generate_random_user_data(user_id, user_email):
    #client_id = str(hash("test_user"+str(user_id)))
    #user_email = "80702892-1922-4b0a-8c26-86987ca93ed5@mailslurp.com"
    return {
            "ClientId": client_id,
            "Password":"test1234",
            "UserAttributes": [
                {"Name": "email", "Value": user_email},
                {"Name": "given_name", "Value": "test_user_"+str(user_id)},
                {"Name": "family_name", "Value": "test_user_"+str(user_id)},
                {"Name": "custom:street", "Value": "tt2"},
                {"Name": "custom:city", "Value": "tt2"},
                {"Name": "custom:state", "Value": "tt2"},
                {"Name": "custom:postcode", "Value": "tt2"},
                {"Name": "custom:country", "Value": "tt2"}],
                "Username": user_email,
                "ValidationData": None}, client_id


def send_sign_up_request(url, user_json, proxy):
    request_header = {
        "Host": "cognito-idp.us-east-1.amazonaws.com",
        "Accept": "*/*",
        "Referer": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com/",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.SignUp",
        "X-Amz-User-Agent": "aws-amplify/0.1.x js",
        "Origin":  "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com",
        "Content-Type": "application/x-amz-json-1.1",
        "Connection": "keep-alive",
       # "DNT": 1,
    }
    resp = requests.post(url, data=json.dumps(user_json), headers=request_header, verify=False, proxies={"http": proxy, "https": proxy})

    # r = ProxyRequests(url)
    # r.set_headers(request_header)
    # r.post_with_headers(user_json)
    # print(r.get_status_code())
    # print(r.get_json())

    if resp.status_code != 200:# r.get_status_code() != 200:
        print("SignUp request Failed: status code {0}, due to {1}".format(resp.status_code, resp.content))
        return False

    return True


def send_confirm_sign_up_request(email, confirm_code, proxy):
    """
    For some reason sending POST request doesnt work
    use aws cli instead
    """
    payload = {"ClientId": client_id, "ConfirmationCode": confirm_code, "Username": email, "ForceAliasCreation": True}
    headers = {
        "Host": "cognito-idp.us-east-1.amazonaws.com",
        "Content-Type": "application/x-amz-json-1.1",
        "Accept": "*/*",
        "Referer": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com/",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.ConfirmSignUp",
        "X-Amz-User-Agent": "aws-amplify/0.1.x js",
        "Origin": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com",
        "TE": "Trailers",
        "Connection": "keep-alive",
        "cache-control": "no-cache",
    }
    resp = requests.post(main_url, data=json.dumps(payload), headers=headers, verify=False, proxies={"http": proxy, "https": proxy})
    if resp.status_code != 200:
        print("Confirm SignUp request Failed: status code {0}, due to {1}".format(resp.status_code, resp.content))
        return False

    return True

    # output = subprocess.run(["aws", "cognito-idp", "confirm-sign-up", "--client-id", client_id,
    #                          "--username={0}".format(email), "--confirmation-code", confirm_code, "--user-context-dat={0}".format(proxy)],
    #                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    # if output.stderr:
    #     print("Confirm SignUp request Failed due to \n{0}".format(output.stderr))
    #     return False
    #
    # return True

    # output = subprocess.run(["curl", "-X", "POST", "https://cognito-idp.us-east-1.amazonaws.com/", "-H", "Content-Type: application/x-amz-json-1.1",
    #                         "-H", 'cache-control: no-cache', "-H", 'x-amz-target: AWSCognitoIdentityProviderService.ConfirmSignUp',
    #                         "-d", json.dumps(payload), "-x", proxy],
    #                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    #                         )
    #
    # if output.stderr:
    #     print("Confirm SignUp request Failed due to \n{0}".format(output.stderr))
    #     return False
    #
    # return True


def reset_password_request(email_address, proxy):
    options_header_req = {
        "Host": "cognito-idp.us-east-1.amazonaws.com",
        "Accept": "*/*",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-amz-target,x-amz-user-agent",
        "Referer": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com/",
        "Origin": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com",
        "Connection": "keep-alive",
        }
    options_resp = requests.options(url="https://cognito-idp.us-east-1.amazonaws.com/",
                                    headers=options_header_req, proxies={"http": proxy, "https": proxy})
    if options_resp.status_code != 200:
        print("Couldn't get available options before request for resetting the password {0}".format(options_resp.status_code))
    # if "POST" not in options_resp: #["headers"]["Access-Control-Allow-Methods"]:
    #     print("POST is not supported for the requested URL")
    #     return False
    reset_pass_header = {
        "Host": "cognito-idp.us-east-1.amazonaws.com",
        "Accept": "*/*",
        "Referer": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com/",
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.ForgotPassword",
        "X-Amz-User-Agent": "aws-amplify/0.1.x js",
        "Origin": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com",
        "Connection": "keep-alive",
        "TE": "Trailers"
    }

    reset_pass_payload = {"ClientId": client_id, "Username": email_address}
    reset_pass_resp = requests.post(url, data=json.dumps(reset_pass_payload), headers=reset_pass_header)
    if reset_pass_resp.status_code == 200:
        return True
    else:
        print("Reset password request failed {0},\n due to {1}".format(reset_pass_resp.status_code,
                                                                        reset_pass_resp.text))


def set_new_password(user_name, new_password, code, proxy):
    new_pass_header = {
        "Host": "cognito-idp.us-east-1.amazonaws.com",
        "Accept": "*/*",
        "Referer": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com/",
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.ConfirmForgotPassword",
        "X-Amz-User-Agent": "aws-amplify/0.1.x js",
        "Origin": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com",
        "Connection": "keep-alive",
    }

    new_pass_payload = {"ClientId": client_id, "Username": user_name, "ConfirmationCode": code, "Password": new_password}

    resp = requests.post(main_url, headers=new_pass_header, data=json.dumps(new_pass_payload), proxies={"http": proxy, "https": proxy})
    if resp.status_code == 200:
        return True
    else:
        print("Couldn't set the new password {0},\n due to {1}".format(resp.status_code, resp.text))
        return False


def resend_confirm_code(email):
    header = {
        "Host": "cognito-idp.us-east-1.amazonaws.com",
        "Accept": "*/*",
        "Referer": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com/",
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.ResendConfirmationCode",
        "X-Amz-User-Agent": "aws-amplify/0.1.x js",
        "Origin": "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com",
        "Connection": "keep-alive",
        "TE": "Trailers"
    }

    payload = {"ClientId": client_id, "Username": email}

    requests.post(main_url, data=json.dumps(payload), headers=header)


def sign_in_from_browser(url, user_name, password):
    binary_path = "/home/at/phD/projects/Cloudsecurity/tools/geckodriver-v0.26.0-linux64/geckodriver"

    driver = webdriver.Firefox(executable_path=binary_path)
    driver.get(url)

    email_elem = driver.find_element_by_name("email")
    pswd_elem = driver.find_element_by_name("password")

    email_elem.clear()
    pswd_elem.clear()

    email_elem.send_keys(user_name)
    pswd_elem.send_keys(password)

    driver.find_element_by_xpath("//*[@data-test='sign-in-sign-in-button']").click()
    time.sleep(4)
    if "All things Alexa" in driver.page_source:
        # success log in
        print_green("{0} successfully logged in!".format(user_name))
        driver.close()
        return True
    # elif "User does not exist" in driver.page_source:
    #     print("{0} does not exist".format(user_name))
    # elif "Incorrect username or password" in driver.page_source:
    #     print("Incorrect username ({0}) or password ({1})".format(user_name, password))
    elif "Sign in to your account" in driver.page_source:
        print_red("Couldn't sign in here is the page source")# \n{0}".format(driver.page_source))
    else:
        print_red(driver.page_source)

    time.sleep(3)
    driver.close()
    return False


def create_user(email_address, inbox_id, user_index, proxy_info):
    sample_user_json, user_id = generate_random_user_data(user_index, email_address)
    success_create_account = send_sign_up_request(main_url, user_json=sample_user_json, proxy=proxy_info)
    if success_create_account:
        # confirm sign up
        time.sleep(5)
        res = email_handler.check_verification_code(inbox_id)
        if res:
            resp = send_confirm_sign_up_request(email=email_address, confirm_code=res, proxy=proxy_info)
            if resp:
                print_green("HAYYY: test_user{0} signed up and Confirmed SUCCESSFULLY!".format(user_index))
                return True
        else:
            print_red("Couldn't Get the response code, email not found!")

    return False


def forgot_password(email, inbox_id, new_password, proxy_info):
    res = reset_password_request(email, proxy_info)
    if res:
        time.sleep(5)
        #verification_code = check_verification_code(email)
        verification_code = email_handler.check_verification_code(inbox_id)
        if verification_code:
            res = set_new_password(email, new_password, verification_code, proxy_info)
            if res:
                print_green("Successfully set the new password for {0}".format(email))
                return True
            else:
                print_red("Couldn't set the password!")

    return False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(prog="aws_ecommerce_api.py", formatter_class=StructuredFormatter)

    group_input = parser.add_argument_group("Input parameters")
    group_input.add_argument('token', help='MailSlurp API Token')
    group_input.add_argument('start', help='start user index')
    group_input.add_argument('end', help='end user index')
    group_input.add_argument('proxy_num', help='number of proxy servers')

    args = parser.parse_args()

    # print(args.token)
    # print(type(args.token))

    global email_handler
    email_handler = EMailHandler(api_key=args.token)

    # proxy settings
    proxies = get_proxies(n=int(args.proxy_num))
    proxy_pool = cycle(proxies)

    # create a sample user
    success_sign_in = 0
    success_forgot_pwd = 0
    success_log_in = 0

    counter = 0
    for i in range(int(args.start), int(args.end)):
        proxy = next(proxy_pool)

        # while not is_open(proxy.split(':')[0]):
        #     print("Not open let's try another one!")
        #     proxy = next(proxy_pool)

        email_info = email_handler.create_inbox()
        email = email_info["email"]
        inbox_id = email_info["id"]

        # test email
        # email = "68cc4417-7307-49d2-ab69-368acf79592a@mailslurp.com"
        # inbox_id = "68cc4417-7307-49d2-ab69-368acf79592a"
        res = create_user(email, inbox_id=inbox_id, user_index=i, proxy_info=proxy)

        if res:
            success_sign_in += 1
            rand = random.randint(1, 10)
            # ~100% of the time do the log in right after signed in
            if rand <= 9 or True:
                time.sleep(4)
                res = sign_in_from_browser(browser_url, email, "test1234")
                if res:
                    success_log_in += 1
                    time.sleep(2)

            if rand <= 3 or True:
                # ~20% of times do the 'forgot password'
                res = forgot_password(email, inbox_id=inbox_id, new_password="test4321", proxy_info=proxy)
                if res:
                    success_forgot_pwd += 1

                new_rand = random.randint(1, 10)
                if new_rand <= 9 or True:
                    time.sleep(4)
                    res = sign_in_from_browser(browser_url, email, "test4321")
                    if res:
                        success_log_in += 1
                        time.sleep(2)

        #email_handler.delete_inbox(inbox_id)

        counter += 1
        rand_time = random.randint(10, 15)
        time.sleep(rand_time)

    print_yellow("Total : {0}".format(counter))
    print_yellow("Success Log In: {0}".format(success_log_in))
    print_yellow("Success Forgot password: {0}".format(success_forgot_pwd))
    print_yellow("Success Sign In: {0}".format(success_sign_in))


if __name__ == "__main__":
    # res = reset_password_request(sample_user_id, sample_user_email)

    # test mailinator
    #res = check_verification_code("test_user")

    #resp = send_confirm_sign_up_request(email="test_user4@mailinator.com", confirm_code="430918")

    #sign_in_from_browser(main_url, sample_user_email, password="test1235")

    #create_user("00")

    #forgot_password("test_user7@mailinator.com", "test12345")

    #main()

    for proxy in get_proxies():
        try:
            send_confirm_sign_up_request("atefehmohseni72@yahoo.com", "880519", proxy)
            print_green("--------------")
        except Exception as e:
            print_red("Damn")