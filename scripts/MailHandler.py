import mailslurp_client
from mailslurp_client.rest import ApiException
import re

from global_variables import print_red, print_green, print_yellow

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class EMailHandler(object):
    def __init__(self, api_key):
        configuration = mailslurp_client.Configuration()
        # Configure API key authorization: API_KEY
        configuration.api_key['x-api-key'] = api_key
        print("SET THE TOKEN!!")
        with mailslurp_client.ApiClient(configuration) as api_client:
            self.api_client = api_client

    def create_inbox(self):
        inbox_inof = {}
        api_instance = mailslurp_client.InboxControllerApi(self.api_client)
        try:
            # Create an Inbox (email address)
            inbox_instance = api_instance.create_inbox()
            #print_yellow(inbox_instance)
            inbox_info = {"email": inbox_instance.email_address, "id": inbox_instance.id}
        except ApiException as e:
            print("Exception when calling InboxControllerApi->create_inbox: %s\n" % e)

        return inbox_info

    def get_email(self, inbox_id):
        email_id = None
        api_instance = mailslurp_client.InboxControllerApi(self.api_client)
        limit = 30
        # Minimum acceptable email count. Will cause request to hang
        # (and retry) until minCount is satisfied or retryTimeout is reached
        min_count = 1
        retry_timeout = 1000
        try:
            # Get emails in an Inbox
            emails = api_instance.get_emails(inbox_id, limit=limit, min_count=min_count,
                                             retry_timeout=retry_timeout)
            for item in emails:
                if "Your verification code" in item.subject:
                    email_id = item.id
            #print_yellow(emails)
        except ApiException as e:
            print("Exception when calling InboxControllerApi->get_emails: %s\n" % e)

        return email_id

    def get_email_body(self, email_id):
        email_body = None
        api_instance = mailslurp_client.EmailControllerApi(self.api_client)
        decode = True

        try:
            # Get email content
            api_response = api_instance.get_email(email_id, decode=decode)
            email_body = api_response.body
            #print_yellow(api_response)
        except ApiException as e:
            print("Exception when calling EmailControllerApi->get_email: %s\n" % e)
        return email_body

    def delete_email(self, email_id):
        api_instance = mailslurp_client.EmailControllerApi(self.api_client)
        try:
            # Delete an email
            api_instance.delete_email(email_id)
        except ApiException as e:
            print("Exception when calling EmailControllerApi->delete_email: %s\n" % e)

    def check_verification_code(self, inbox_id):
        verification_code = None
        email_id = self.get_email(inbox_id)
        if email_id:
            email_body = self.get_email_body(email_id)
            if email_body:
                verification_code = re.search("Your verification code is (\d+)", email_body)

                if verification_code:
                    verification_code = verification_code.group(1)
                    print_red("Verification Code is : {0}".format(verification_code))
            #self.delete_email(email_id)

        return verification_code

    def delete_inbox(self, inbox_id):
        api_instance = mailslurp_client.InboxControllerApi(self.api_client)
        try:
            # Delete inbox
            api_instance.delete_inbox(inbox_id)
        except ApiException as e:
            print("Exception when calling InboxControllerApi->delete_inbox: %s\n" % e)


# if __name__ == "__main__":
#     mail_handler = EMailHandler()
#     # inbox_info = mail_handler.create_inbox()
#     # print(inbox_info)
#     #mail_handler.check_verification_code("ec35edb7-c2d1-4424-9a55-1365ef957308")
#     mail_handler.delete_inbox("ec35edb7-c2d1-4424-9a55-1365ef957308")
