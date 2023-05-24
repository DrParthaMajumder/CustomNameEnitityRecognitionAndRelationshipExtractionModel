#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:54:37 2023

@author: partha
"""


# PART====>1 : Train NER model
import os
os.chdir("/home/partha/EnvParthaWin/ProjectsExplotion/VCC_DATA/CustomNER")

import json
import spacy
from spacy.util import filter_spans
import requests
from bs4 import BeautifulSoup
from spacy.tokens import DocBin
from tqdm import tqdm
nlp = spacy.load("en_core_web_lg")
doc_bin = DocBin()


#=====>NER MODEL<===========================
# !python -m spacy convert TrainIOB.tsv ./ -t json -n 1 -c iob
# !python -m spacy convert TestIOB.tsv ./ -t json -n 1 -c iob
# !python -m spacy convert TrainIOB.json ./ -t spacy 
# !python -m spacy convert TestIOB.json ./ -t spacy
# !python -m spacy init fill-config base_config.cfg config.cfg
# !python -m spacy train config.cfg --output ./ --paths.train ./TrainIOB.spacy --paths.dev ./TestIOB.spacy
# !spacy evaluate ./model-best ./TestIOB.spacy 


#=====>NER MODEL<===========================
!python -m spacy convert TestIOB35_IOB.tsv ./ -t json -n 1 -c iob
!python -m spacy convert Train_IOB_40.tsv ./ -t json -n 1 -c iob

!python -m spacy convert TestIOB35_IOB.json ./ -t spacy 
!python -m spacy convert Train_IOB_40.json ./ -t spacy

!python -m spacy convert Testkk.json ./ -t spacy


!python -m spacy init fill-config base_config.cfg config.cfg

!python -m spacy train config.cfg --output ./ --paths.train ./Train_IOB_40.spacy --paths.dev ./TestIOB.spacy



!spacy evaluate ./model-best ./TestIOB.spacy 






# PART=====>2 : Test NER model
import os
import requests
url = "https://www.vccircle.com//temasek-sequoia-others-invest-226-mn-in-fashion-startup-zilingo"
colors = {"STARTUP": "#F67DE3", "VALUATION": "#7DF6D9", "VC FIRM":"#0B5DF0", "FUNDING AMOUNT": "#0BF02E"}
options = {"colors": colors} 
article_page = requests.get(url)
soup = BeautifulSoup(article_page.content, "html.parser")
article_element = soup.find("div", class_ = "articleDetail_article-content__GPSys")
date_element = soup.find("ul", class_ = "articleDetail_date__MgWtb")
all_paragraphs = article_element.find_all("p")
c = list()
news=list()
for paragraph in all_paragraphs:
    news.append(paragraph.text)


import spacy
nlp = spacy.load("model-best")
nlp.add_pipe('sentencizer')
Text=news
for doc in nlp.pipe(Text, disable=["tagger"]):
   print(f"spans: {[(e.start, e.text, e.label_) for e in doc.ents]}")




# PART=====>3 : Train Relationship model
os.chdir("/home/partha/EnvParthaWin/ProjectsExplotion/VCC_DATA/tutorials/rel_component")

!spacy project run train_cpu


# Part =====>4: Test Relationship model

import os
os.chdir("/home/partha/EnvParthaWin/ProjectsExplotion/VCC_DATA/CustomNER")
import spacy
nlp = spacy.load("model-best")
nlp.add_pipe('sentencizer')
#nlp.add_pipe(nlp.create_pipe('sentencizer'))
Text=['''Southeast Asian fashion startup Zilingo said it has raised $226 million in its latest funding round from existing backers such as venture capital firm Sequoia Capital, with Singapore’s Temasek Holdings joining as a new investor.', 'The Series D financing follows a $54 million fundraising last year, taking the total capital raised by the company to $308 million. The company declined to provide valuation.', 'The latest round included Singapore investment fund EDBI and previous investors Burda Principal Investments, a division of Germany’s Hubert Burda Media, and Belgian investment firm Sofina, Zilingo said in a statement on Tuesday.', 'The Singapore-headquartered company plans to use the funds to invest in infrastructure and technology to further integrate and digitize the fashion and beauty supply chain, it said.', 'Zilingo, whose main market for the consumer business is Indonesia, is expanding in countries such as Australia in 2019.', 'The company, which started as a fashion marketplace, has been rapidly growing its business-to-business (B2B) tools and platforms, which include providing value-added services to its merchants as well as a marketplace to help them source efficiently from manufacturers.', 'The company, which also acts as an affiliate for companies to provide financing to small firms on its platform, is now earning the bulk of its revenue from its B2B business.', 'Zilingo has grown its revenues by four times in the last 12 months, it said, but did not provide specific numbers.', '“We are pretty close to profitability and have a clear path to it,” Ankiti Bose, the company’s co-founder and CEO, told Reuters. Bose founded the company in 2015 with Dhruv Kapoor, its chief technology officer.', 'In Southeast Asia, local fashion e-commerce players such as Zalora, ‘Love, Bonito’ and JD.com-backed fashion retailer Pomelo compete with global platforms like ASOS. A study by Google and Temasek has forecast that e-commerce in Southeast Asia will exceed $100 billion in gross merchandise value by 2025 from over $23 billion in 2018.', 'Bose said purely selling to consumers would mean that the only way to win is through price wars and discounting.', '“Instead what we are trying to do is trying to lower the cost of procurement for these merchants and add services on that layer there, which is basically before it even gets to the merchant and they try to sell it online,” she added.', 'Share article on''']
for doc in nlp.pipe(Text, disable=["tagger"]):
   print(f"spans: {[(e.start, e.text, e.label_) for e in doc.ents]}")



os.chdir("/home/partha/EnvParthaWin/ProjectsExplotion/VCC_DATA/tutorials/rel_component/scripts")


import random
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from rel_pipe import make_relation_extractor, score_relations
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
# We load the relation extraction (REL) model
nlp2 = spacy.load("/home/partha/EnvParthaWin/ProjectsExplotion/VCC_DATA/tutorials/rel_component/training/model-best")
for name, proc in nlp2.pipeline:
  doc = proc(doc)




for value, rel_dict in doc._.rel.items():
  for sent in doc.sents:
    for e in sent.ents:
      for b in sent.ents:
        if e.start == value[0] and b.start == value[1]:
          if rel_dict['STARTUP=>FUNDING_AMOUNT'] >=0.03:
            print(f" entities: {e.text, b.text} --> predicted relation: {rel_dict}")


