import csv
import json
import os
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.edge.service import Service
from urllib.parse import urlencode
from selenium.webdriver.edge.options import Options
import time
import re
import logging
import pandas as pd

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def findstr(pattern, string):
    # 正则表达式匹配字符串
    ans = re.search(pattern, string)
    if ans:
        span = ans.span()
        return string[span[0]: span[1]]
    return ""


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("scraping.log"),
                            logging.StreamHandler()
                        ])


def fetch_url(url, retries=3):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response
        else:
            logging.warning(f"Failed to fetch {url}: Status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        if retries > 0:
            time.sleep(2)  # Backoff before retry
            return fetch_url(url, retries - 1)
    return None

def sanitize_filename(filename):
    # 通过正则表达式移除文件名中的非法字符
    return re.sub(r'[\\/*?:"<>|]', '', filename)

def download_pdf(url, save_folder):
    try:
        response = fetch_url(url)
        if response:
            # 提取并清理文件名
            file_name = url.split('/')[-1].split('&')[0]
            file_name = sanitize_filename(file_name)  # 调用函数移除非法字符
            file_path = os.path.join(save_folder, file_name)

            # 写入文件
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f'File downloaded to {file_path}')
        else:
            logging.error(f'Failed to download: {url}')
    except IOError as e:
        logging.error(f'File operation error: {e}')
    except Exception as e:
        logging.error(f'Unknown error: {e}')


def main():
    setup_logging()
    logging.info("Starting the scraper...")

    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)

    try:
        logging.info("Navigating to the main page...")
        url = "https://data.eastmoney.com/report/industry.jshtml"
        driver.get(url)

        res = driver.page_source
        soup = BeautifulSoup(res, 'lxml')
        nav_tags = soup.find('div', class_='catemark').find_all('a')

        passage_href_lst = []
        articles = []

        # 导航遍历
        for each_nav in nav_tags:
            try:
                nav_url = 'https://data.eastmoney.com/report/' + each_nav.get('href')
                logging.info(f"Processing {nav_url}")
                driver.get(nav_url)

                for _ in range(1):  # Adjust page limit
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'table-model'))
                    )
                    res_nav = driver.page_source
                    nav_soup = BeautifulSoup(res_nav, 'lxml')
                    table = nav_soup.find('table', class_='table-model')

                    if table:
                        rows = table.find('tbody').find_all('tr')
                        for each in rows:
                            temp_each = each.find_all('a')[5]
                            passage_href_lst.append(temp_each)
                    else:
                        logging.warning("Table not found on the page.")
                        break

                    # 翻页
                    try:
                        nextpage_label = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.LINK_TEXT, '下一页'))
                        )
                        nextpage_label.click()
                    except TimeoutException:
                        logging.info("No more pages to load.")
                        break
            except Exception as e:
                logging.error(f"Error processing navigation {each_nav.get('href')}: {e}")

        save_folder = os.path.abspath('..\东方财富爬虫')
        os.makedirs(save_folder, exist_ok=True)

        # 提取pdf
        article_text = []
        for each_passage in passage_href_lst:
            try:
                driver.get('https://data.eastmoney.com/' + each_passage.get('href'))
                res_passage = driver.page_source
                soup_passage = BeautifulSoup(res_passage, 'lxml')
                passage_href = soup_passage.find('a', class_='pdf-link')

                if passage_href:
                    pdf_url = passage_href.get('href')
                    article_text.append(pdf_url)
                    download_pdf(pdf_url, save_folder)
                else:
                    logging.warning(f"No PDF link found for {each_passage.get('href')}")
            except Exception as e:
                logging.error(f"Error processing article {each_passage}: {e}")

    finally:
        driver.quit()
        logging.info("Scraper finished.")


if __name__ == "__main__":
    main()
