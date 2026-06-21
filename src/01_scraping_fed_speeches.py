'''
 Scraping Fed Speeches 
 In this script we define and implement the functions to scrape the federal reserve website
 The speeches are stored following the structure (date, speaker, title, link, speech)
 The results are stored in a pickle file (to continue with NPL) and a .xlsx file (for visualization purposes)
'''

#######################################################
### Function that creates a list of all links neccesary 
def create_url_list(start_year, end_year, prefix, suffix):
    annual_htm_list = []
    for x in range(start_year, end_year+1):
        if x <=2010:
            this_suffix = 'speech.htm'
            mid_str=str(x)
            annual_htm_list.append(prefix + mid_str + this_suffix)
        else:
            mid_str = str(x)
            annual_htm_list.append(prefix + mid_str + suffix)
    return annual_htm_list

#######################################################
### Function that creates 4 lists, containing the dates, speakers, titles and links for every speech in a particular year 
### based on the links created in the previous function
def find_speeches_by_year(host, this_url, print_test=False):
    conn = HTTPSConnection(host = host)
    conn.request(method='GET', url = this_url)
    resp = conn.getresponse()
    body = resp.read()
    # check that we received the correct response code
    if resp.status != 200:
        print('Error from Web Site! Response code: ', resp.status)
    else:
        soup=BeautifulSoup(body, 'html.parser')
        event_list = soup.find('div', class_='row eventlist')
        # creating the list of dates, titles, speakers and html articles from web page
        date_lst =[]
        title_lst = []
        speaker_lst = []
        link_lst = []

        for row in event_list.find_all('div', class_='row'):
            tmp_date= [x.text for x in row.find_all('time')]
            date_lst.append(tmp_date)
        
            tmp_speaker = [x.text for x in row.find_all('p', class_='news__speaker')]
            speaker_lst.append(tmp_speaker)
        
            tmp_title = [x.text for x in row.find_all('em')]
            title_lst.append(tmp_title)

        # some of the links include video with the transcript. We are deleteing these here
        for link in event_list.find_all('a', href=True, class_ = lambda x: x != 'watchLive'):
            link_lst.append(link['href'])
        
        if print_test:
            print('length of dates: ', len(date_lst))
            print('length of speakers: ', len(speaker_lst))
            print('length of titles: ', len(title_lst))
            print('length of href: ', len(link_lst))

        return date_lst, speaker_lst, title_lst, link_lst

#######################################################
### This function creates a dataframe from the 4 lists created in the previous function, as well as an empty column 
### to store the speeches when scraping them
def create_speech_df(host, annual_htm_list):
    all_dates = []
    all_speakers = []
    all_titles = []
    all_links = []
    for item in annual_htm_list:
        date_lst, speaker_lst, title_lst, link_lst =find_speeches_by_year(host, 
                                                    item, print_test=False)
        all_dates = all_dates + date_lst
        all_speakers = all_speakers + speaker_lst
        all_titles = all_titles + title_lst
        all_links = all_links + link_lst
    
    dict1 = {'date': all_dates, 'speaker':all_speakers,
            'title': all_titles, 'link':all_links}
    df = pd.DataFrame.from_dict(dict1)
    
    #Cleaning up some of the dateframe elements to remove brackets
    df['date']=df['date'].str[0]
    df['date'] = pd.to_datetime(df['date'])
    df['speaker']=df['speaker'].str[0]
    df['title']=df['title'].str[0]
    
    # creating empty column for documents
    df['text'] = ""

    # removing items that are not speeches. These contain a link that starts with '/pubs/feds'
    delete_these = df[df['link'].str.match('/pubs/feds')].index
    df = df.drop(delete_these)
    df = df.reset_index(drop=True)
    return df

#######################################################
### This function creates a loop to iterate the following function, which actually does the scraping, 
### and stores the scraped speech into the dataframe
def retrieve_docs(host, df):
    for index, row in df.iterrows():
        this_item = df['link'][index]
        print('Scraping text for documents #: ', index)
        doc = get_one_doc(host, this_item)
        df.loc[index, 'text'] = doc
    return df

#######################################################
### This function returns a string with the speech from a particular link
### It incorporates various filtering to automatically remove noise and clean unwanted parts
def get_one_doc(host, this_url):
    import re
    temp_url = 'https://' + host + this_url
    response = requests.get(temp_url)
    sp = BeautifulSoup(response.text, 'html.parser')
    article = sp.find('div', class_='col-xs-12 col-sm-8 col-md-8')

    if article is None:
        article = sp.find('div', class_='col-md-8')

    if article is None:
        article = sp.find('div', id='content')

    if article is None:
        print("No article found")
        return ""

    # Now extract clean paragraphs
    doc = []

    for p in article.find_all('p'):
        text = p.get_text(strip=True)

        if not text:
            continue

        lower_text = text.lower()

        # Detect section headers
        if re.search(r'\b(references|footnotes|notes)\b', lower_text):
            break

        # Detect typical footnote patterns (start of line)
        if re.match(r'^\d+[\.\)]\s', text):  # "1. ..." or "1) ..."
            break

        doc.append(text)
    
    full_text = '\n'.join(doc)

    # Remove video instruction block at the beginning
    if full_text.startswith("Accessible Keys for Video"):
        full_text = re.sub(
            r'^Accessible Keys for Video.*?caption on/off\.\s*',
            '',
            full_text,
            flags=re.DOTALL
        )
    
    # Remove any leftover "Return to text"
    full_text = re.sub(r'Return to text', '', full_text)

    return full_text.strip()

#######################################################
### MAIN

if __name__ == '__main__':
    # import functions
    import pandas as pd 
    import numpy as np
    from bs4 import BeautifulSoup
    from http.client import HTTPSConnection
    import pickle
    from urllib.request import urlopen
    import requests

    host = 'www.federalreserve.gov'
    prefix = '/newsevents/speech/'
    suffix = '-speeches.htm'
    start_year = 2006
    end_year = 2026

    # create list of web site containing annual speech links
    annual_htm_list =create_url_list(start_year, end_year, prefix, suffix)        
    print('Below is the annual_htm_list')
    print(annual_htm_list)
    
    # create dataframe containing speech information (not yet the text)
    df = create_speech_df(host, annual_htm_list)
    
    # scrape the text from every speech in the dataframe
    df = retrieve_docs(host, df)
    print(df.info())
    df = df.reset_index(drop=True)

    # create a copy of the dataframe that will be manipulated for better visualization
    df_export = df.copy()

    # normalize spaces
    df['text'] = df['text'].str.replace(r'\n+', '\n', regex=True)

    # saving the df to a pickle file
    pickle_out = open('all_fed_speeches', 'wb')
    pickle.dump(df, pickle_out)
    pickle_out.close()

    # Remove excessive whitespace (on the export df)
    df_export['text'] = df_export['text'].str.replace(r'\s+', ' ', regex=True)

    # create a visual appealing df
    df_export.to_excel("fed_speeches.xlsx", index=False)
