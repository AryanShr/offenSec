import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from collections import defaultdict
import json
import re
import os

driver = webdriver.Chrome(executable_path=r"D://chromedriver.exe")
main_event = ["Misc","Process","Registry","Files","Mutex","Network"]
pattern = r'offset=[0-9]+$'

# Yahan pe URL me se id dal dena bas

id = "231016-l73tmsdd2v"

if not os.path.exists(id):
    os.makedirs(id)
for eve in main_event:
    if os.path.exists(f'{id}/{eve}.json'):
        print(f"Skipping {eve} path already exist")
        continue
    # run karne se pahle yeh bhi check kar lena
    url = f"https://tria.ge/{id}/behavioral1/analog?main_event={eve}"
    driver.get(url)
    time.sleep(10)

    try:
        pagination = driver.find_element(By.CLASS_NAME, "pagination")
        totalClicks = pagination.text.split(" ")[-2]
        totalClicks = int(totalClicks)
    except:
        totalClicks = 1
    data = defaultdict(list)

    for _ in range(totalClicks):
        # find elements by their class name
        elements = driver.find_element(By.CLASS_NAME, "list").find_elements(By.TAG_NAME, "li")

        for ele in elements:
            event = ele.find_element(By.TAG_NAME, "b").text
            # print(event)
            detail = ele.find_elements_by_css_selector(".key-value.rows > div")
            meta = {}
            for i in detail:
                content = i.find_elements(By.TAG_NAME, "div")
                meta[content[0].text] = content[1].text
            data[event].append(meta)

        # try to click the "Next" button and go to the next page
        try:
            pagination = driver.find_element(By.CLASS_NAME, "pagination")
            nextButton = pagination.find_elements(By.TAG_NAME, "a") 
            match = re.search(pattern, nextButton[-2].get_attribute("href"))
            if match:
                found = match.group()
                nextButton[-2].click()
                print("Navigating to Next Page ", found)
            else:
                print("Pattern not found in the URL.")
                break
        except Exception as e:
            print(e)
            print("Reached the end of pages.")
            break
    print(f"Finished scraping {eve}")
    file_path = f'{id}/{eve}.json'
    with open(file_path, 'w') as fp:
        json.dump(data, fp)
    