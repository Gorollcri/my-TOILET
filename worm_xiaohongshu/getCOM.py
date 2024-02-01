import csv
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

with open("informa.json", "r", encoding='utf-8') as f:
    data=json.load(f)

phone_number = data.get("phone_number")
keyword = data. get("keyword")

options=webdriver.ChromeOptions()
options.add_argument(r'--user-data-dir=C:\\Users\\86152\\AppData\\Local\\Google\\Chrome\\User Data')
driver=webdriver.Chrome(options=options)


driver.get("https://www.xiaohongshu.com")


phone_input = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/div/div[1]/div[3]/div[2]/form/label[1]/input")  
phone_input.send_keys(phone_number)
login_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/div/div[1]/div[3]/div[2]/form/label[2]/span")
login_button.click()    #获得验证码

verification_code = input()

code_input = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/div/div[1]/div[3]/div[2]/form/label[2]/input")  
code_input.send_keys(verification_code)
login_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/div/div[1]/div[3]/div[3]/span")  
login_button.click()    #确认协议
login_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/div/div[1]/div[3]/div[2]/form/button")  
login_button.click()     #登录

search_input = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/div[1]/header/div[2]/input")  
search_input.send_keys(keyword)
search_input.send_keys(Keys.RETURN)    #查找关键词


wait = WebDriverWait(driver, 10)
search_results = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "search-result")))     

comments = []
for result in search_results:
    result.click()
    comment_elements = driver.find_elements(By.CLASS_NAME, "comment")
    for comment_element in comment_elements:  
        comment_text = comment_element.text
        comments.append(comment_text)
#挨个点击搜索结果，获取评论

with open("comments.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Comment"])
    writer.writerows(comments)
#写入csv文件

driver.quit()
