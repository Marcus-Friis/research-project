import requests
import io
import PyPDF2
import time
import json
import os
import xml.etree.ElementTree as ET

class PDFToStringConverter:
    def __init__(self, pdf_url):
        self.pdf_url = pdf_url

    def download_pdf(self):
        try:
            response = requests.get(self.pdf_url)
            if response.status_code == 200:
                return io.BytesIO(response.content)
            else:
                raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download PDF: {e}")

    def convert_to_string(self):
        pdf_file = self.download_pdf()
        pdf_text = ""
        
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                pdf_text += page.extract_text()
            return pdf_text
        except Exception as e:
            raise Exception(f"Failed to convert PDF to text: {e}")


ARXIV_JSON_FILENAME = 'arxiv.json'

def _init_json():
    if os.path.isfile(ARXIV_JSON_FILENAME):
        with open(ARXIV_JSON_FILENAME, 'r') as f:
            return json.load(f)
    
    data = {}
    with open(ARXIV_JSON_FILENAME, 'w') as f:
        json.dump(data, f, indent=4)
    return data

def _get_body(arxiv_id: str) -> str:
    url = f'https://arxiv.org/pdf/hep-ph/{arxiv_id}.pdf'
    pdf = PDFToStringConverter(url)
    return pdf.convert_to_string()


def _parse_arx_xml(root):
    element = ET.fromstring(root)
    result = {}
    for child in element:
        tag = child.tag.split('}')[-1]
        child_dict = _parse_arx_xml(child)

        if tag in result:
            if isinstance(result[tag], list):
                result[tag].append(child_dict)
            else:
                result[tag] = [result[tag], child_dict]
        else:
            result[tag] = child_dict

    # Handle the case where the element has no sub-elements and contains attributes
    if not result:
        if element.text:
            return element.text
        elif element.attrib:
            return element.attrib
        else:
            return None

    return result

def _get_metadata(arxiv_id: str) -> str:
    api_url = f'https://export.arxiv.org/api/query?id_list=hep-ph/{arxiv_id}&max_results=1'
    response = requests.get(api_url)
#    return _parse_arx_xml(response.text)

def _retry(func):
    def wrapper(*args, **kwargs):
        for interval in [5, 10, 20, 60]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                time.sleep(interval)
                continue
        raise e
    return wrapper

#@_retry
def _visit_article(arxiv_id: str) -> str:
    metadata = _get_metadata(arxiv_id)
    body = _get_body(arxiv_id)
    data = {
        'metadata': metadata,
        'body': body
    }
    return data

def wipe_cache():
    with open(ARXIV_JSON_FILENAME, 'w') as f:
        json.dump({}, f, indent=4)

def scrape(arxiv_list: list, sleep_interval: int = 15) -> dict:
    data = _init_json()
    for arxiv_id in arxiv_list:
        if arxiv_id not in data:
            arxiv_content = _visit_article(arxiv_id)
            data[arxiv_id] = arxiv_content
            with open(ARXIV_JSON_FILENAME, 'w') as f:
                json.dump(data, f, indent=4)
            time.sleep(sleep_interval)
    return data






if __name__ == '__main__':
    arxiv_list = ['0203079']
    print(_visit_article(arxiv_list[0]))


    ## OFFICIAL ARXIV API STUFF
    # import urllib, urllib.request
    # url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'
    # data = urllib.request.urlopen(url)
    # print(data.read().decode('utf-8'))