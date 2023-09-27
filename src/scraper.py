import requests
import io
import PyPDF2
import time
import json
import os


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


class ArxivScraper:
    filename = 'arxiv.json'
    
    def _init_json(self):
        if os.path.isfile(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        
        data = {}
        j = json.dumps(data, indent=4)
        with open(self.filename, 'w') as f:
            f.write(j)
        return data
    
    def __init__(self) -> None:
        self.data = self._init_json()    
    
    def _get_body(self, arxiv_id: str) -> str:
        url = f'https://arxiv.org/pdf/hep-ph/{arxiv_id}.pdf'
        pdf = PDFToStringConverter(url)
        return pdf.convert_to_string()
    
    def _get_metadata(self, arxiv_id: str) -> str:
        api_url = f'https://export.arxiv.org/api/query?id_list=hep-ph/{arxiv_id}&max_results=1'
        response = requests.get(api_url)
        return response.text
        
    @staticmethod
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
        
    @_retry
    def _visit_article(self, arxiv_id: str) -> str:
        metadata = self._get_metadata(arxiv_id)
        body = self._get_body(arxiv_id)
        data = {
            'metadata': metadata,
            'body': body
        }
        return data
    
    def wipe_chache(self):
        with open(self.filename, 'w') as f:
            json.dump({}, f)
        self.data = {}

    def scrape(self, arxiv_list: list, sleep_interval: int = 15) -> dict:
        for arxiv_id in arxiv_list:
            if arxiv_id not in self.data.keys():
                arxiv_content = self._visit_article(arxiv_id)
                self.data[arxiv_id] = arxiv_content
                j = json.dumps(self.data, indent=4)
                with open(self.filename, 'w') as f:
                    f.write(j)
                time.sleep(sleep_interval)
        return self.data


if __name__ == '__main__':
    arxiv_list = ['0203079']
    arx = ArxivScraper()
    print(arx._visit_article(arxiv_list[0]))


    ## OFFICIAL ARXIV API STUFF
    # import urllib, urllib.request
    # url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'
    # data = urllib.request.urlopen(url)
    # print(data.read().decode('utf-8'))