from bs4 import BeautifulSoup
import requests
from selenium import webdriver
import time

url = "https://universe.leagueoflegends.com/pt_BR/explore/everything/newest/"
webdriver = webdriver.Chrome()
webdriver.get(url)
time.sleep(20)
webdriver.implicitly_wait(50)
webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

soup = BeautifulSoup(webdriver.page_source, "html.parser")

print(soup.prettify())

#link_story = soup.find_all(
#    nome = 'a',
#    attrs={'class': 'Card_CCcI Result_2bn_'}
#)

#print(link_story)