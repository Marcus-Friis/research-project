import requests
import io
import PyPDF2
import time
import json
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

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
            # print(f"Failed to convert PDF to text: {e}")
            # return f"Failed to convert PDF to text: {e}"


ARXIV_JSON_FILENAME = 'arxiv.json'

def _init_json():
    if os.path.isfile(ARXIV_JSON_FILENAME):
        with open(ARXIV_JSON_FILENAME, 'r') as f:
            return json.load(f)

    reset_cache()
    return {}

def _get_body_from_id(arxiv_id: str) -> str:
    url = f'https://arxiv.org/pdf/hep-ph/{arxiv_id}.pdf'
    pdf = PDFToStringConverter(url)
    return pdf.convert_to_string()

def _get_body_from_url(url: str) -> str:
    pdf = PDFToStringConverter(url)
    return pdf.convert_to_string()

def _parse_arx_xml(xml):
    element = ET.fromstring(xml)
    result = {}
    for child in element:
        tag = child.tag.split('}')[-1]
        child_dict = _parse_arx_xml(ET.tostring(child, encoding='utf-8').decode('utf-8'))

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
    if response.status_code == 200:
        return _parse_arx_xml(response.text)
    else:
        return None  # Handle the case where the request fails

def _retry(func):
    def wrapper(*args, **kwargs):
        for interval in [5, 10, 20, 60]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = e
                print(f'Error {e} occured, retrying in {interval} seconds.')
                time.sleep(interval)
                continue
        raise error
    return wrapper

# @_retry
def _visit_article(arxiv_id: str) -> str:
    metadata = _get_metadata(arxiv_id)
    if metadata is None:
        raise Exception('Unable to get metadata')
    
    # extract version number from the api
    pdf_url = metadata['entry']['id']
    pdf_url = f'http://arxiv.org/pdf/hep-ph/{pdf_url.split("/")[-1]}.pdf'

    body = _get_body_from_url(pdf_url)
    data = {
        'metadata': metadata,
        'body': body
    }
    return data

def _pad_id(arxiv_id: str) -> str:
    id_length = 7
    n = len(arxiv_id)
    return '0'*(id_length - n) + arxiv_id

def reset_cache():
    with open(ARXIV_JSON_FILENAME, 'w') as f:
        json.dump({}, f, indent=4)

def scrape(arxiv_list: list, sleep_interval: int = 15) -> dict:
    data = _init_json()
    for arxiv_id in tqdm(arxiv_list):
        if arxiv_id not in data:
            padded_id = _pad_id(arxiv_id)
            arxiv_content = _visit_article(padded_id)
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
