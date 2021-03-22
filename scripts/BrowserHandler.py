import time

from selenium import webdriver

from global_variables import print_red, print_green, print_yellow
global email_handler


class BrowserHandler(object):
    def __init__(self):
        binary_path = "/home/at/phD/projects/Cloudsecurity/tools/geckodriver-v0.26.0-linux64/geckodriver"

        self.driver = webdriver.Firefox(executable_path=binary_path)

        self.url = "http://awsamplifyecommerce-20200713180028-hostingbucket-dev.s3-website-us-east-1.amazonaws.com/"

        self.driver.get(self.url)

    def create_user_browser(self, user_data):
        link = self.driver.find_element_by_link_text('Create account')
        link.click()
        time.sleep(2)

        for key, value in user_data.items():
            element = self.driver.find_element_by_name(key)
            element.send_keys(value)

        self.driver.find_element_by_xpath("//*[@data-test='sign-up-create-account-button']").click()
        time.sleep(3)

        if "Confirm Sign Up" in self.driver.page_source:
            # success create user
            print_green("{0} successfully created!".format(user_data["given_name"]))
            return True

        return False

    def confirm_signup(self, username, verification_code):
        email = self.driver.find_element_by_name("username")
        code = self.driver.find_element_by_name("code")

        email.send_keys(username)
        code.send_keys(verification_code)

        self.driver.find_element_by_xpath("//*[@data-test='confirm-sign-up-confirm-button']").click()

        if "Sign in to your account" in self.driver.page_source:
            # success create user
            print_green("{0} successfully created!".format(username))
            return True

        return False

    def forgot_password(self, email):
        link = self.driver.find_element_by_link_text('Reset password')
        link.click()
        time.sleep(2)

        email_elem = self.driver.find_element_by_name("email")
        email_elem.send_keys(email)
        self.driver.find_element_by_xpath("//*[@data-test='forgot-password-send-code-button']").click()
        time.sleep(3)
        if "Reset your password" in self.driver.page_source:
            return True

        return False

    def confirm_forgot_password(self, code, new_pass):
        code_elem = self.driver.find_element_by_name("code")
        password = self.driver.find_element_by_name("password")

        code_elem.send_keys(code)
        password.send_keys(new_pass)

        self.driver.find_element_by_xpath("//*[@data-test='forgot-password-submit-button']").click()

        if "Sign in to your account" in self.driver.page_source:
            return True

        return False

    def sign_in(self, user_name, password):
        email_elem = self.driver.find_element_by_name("email")
        pswd_elem = self.driver.find_element_by_name("password")

        email_elem.clear()
        pswd_elem.clear()

        email_elem.send_keys(user_name)
        pswd_elem.send_keys(password)

        self.driver.find_element_by_xpath("//*[@data-test='sign-in-sign-in-button']").click()
        time.sleep(4)
        if "All things Alexa" in self.driver.page_source:
            # success log in
            print_green("{0} successfully logged in!".format(user_name))
            self.driver.close()
            return True

        elif "Sign in to your account" in self.driver.page_source:
            print_red("Couldn't sign in here is the page source")  # \n{0}".format(driver.page_source))
        else:
            print_red(self.driver.page_source)

        time.sleep(3)
        self.driver.close()
        return False


# if __name__ == "__main__":
#     sample_user = {
#         "email": "atefeh@mailinator.com",
#         "password": "test1234",
#         "given_name": "test_user_1",
#         "family_name": "test_user_1",
#         "street": "test_street",
#         "city": "test_city",
#         "state": "test_state",
#         "postcode": "test_postcode",
#         "country": "test_country"
#     }
#     browser = BrowserHandler()
#     browser.create_user_browser(sample_user)
