import requests
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

## mailslurp config
from mailinator.client import MailinatorClient

from global_variables import print_red, print_green, print_yellow
from scripts.MailHandler import EMailHandler
from scripts.BrowserHandler import BrowserHandler


def create_user_data(user_id, user_email):
    return {
        "email": user_email,
        "password": "test1234",
        "given_name": "test_user_" + str(user_id),
        "family_name": "test_user_" + str(user_id),
        "street": "test_street",
        "city": "test_city",
        "state": "test_state",
        "postcode": "test_postcode",
        "country": "test_country"
    }


def create_user(email_address, inbox_id, user_index, browser_handler):
    sample_user_json = create_user_data(user_index, email_address)
    success_create_account = browser_handler.create_user_browser(sample_user_json)
    if success_create_account:
        # confirm sign up
        time.sleep(5)
        res = email_handler.check_verification_code(inbox_id)
        if res:
            resp = browser_handler.confirm_signup(email=email_address, confirm_code=res)
            if resp:
                print_green("HAYYY: test_user{0} signed up and Confirmed SUCCESSFULLY!".format(user_index))
                return True
        else:
            print_red("Couldn't Get the response code, email not found!")

    return False


def forgot_password(email, inbox_id, new_password, browser_handler):
    res = browser_handler.forgot_password(email)
    if res:
        time.sleep(5)
        verification_code = email_handler.check_verification_code(inbox_id)
        if verification_code:
            res = browser_handler.confirm_forgot_password(verification_code, new_password)
            if res:
                print_green("Successfully set the new PASSWORD for {0}".format(email))
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

    args = parser.parse_args()

    # print(args.token)
    # print(type(args.token))

    global email_handler
    email_handler = EMailHandler(api_key=args.token)

    # create a sample user
    success_sign_in = 0
    success_forgot_pwd = 0
    success_log_in = 0

    counter = 0
    for i in range(int(args.start), int(args.end)):
        browser = BrowserHandler()
        email_info = email_handler.create_inbox()
        email = email_info["email"]
        inbox_id = email_info["id"]

        # test email
        # email = "68cc4417-7307-49d2-ab69-368acf79592a@mailslurp.com"
        # inbox_id = "68cc4417-7307-49d2-ab69-368acf79592a"
        res = create_user(email, inbox_id=inbox_id, user_index=i, browser_handler=browser)

        if res:
            success_sign_in += 1
            rand = random.randint(1, 10)
            # ~100% of the time do the log in right after signed in
            if rand <= 9 or True:
                time.sleep(4)
                res = browser.sign_in(email, "test1234")
                if res:
                    success_log_in += 1
                    time.sleep(2)

            if rand <= 3 or True:
                # ~20% of times do the 'forgot password'
                res = forgot_password(email, inbox_id=inbox_id, new_password="test4321", browser_handler=browser)
                if res:
                    success_forgot_pwd += 1

                new_rand = random.randint(1, 10)
                if new_rand <= 9 or True:
                    time.sleep(4)
                    res = browser.sign_in(email, "test4321")
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
    main()
