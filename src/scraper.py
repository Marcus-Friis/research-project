import requests
import io
import PyPDF2


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
    def __init__(self, arxiv_list: list) -> None:
        self.arxiv_list = arxiv_list
    
    
    def _get_body(self, arxiv_id: str) -> str:
        url = f'https://arxiv.org/pdf/hep-ph/{arxiv_id}.pdf'
        pdf = PDFToStringConverter(url)
        return pdf.convert_to_string()
    
    def _get_metadata(self, arxiv_id: str) -> str:
        api_url = f'https://export.arxiv.org/api/query?id_list=hep-ph/{arxiv_id}&max_results=1'
        response = requests.get(api_url)
        return response.content
        
    def _visit_article(self, arxiv_id: str) -> str:
        metadata = self._get_metadata(arxiv_id)
        body = self._get_body(arxiv_id)
        data = {
            'metadata': metadata,
            'body': body
        }
        return data

    def scrape(self):
        data = {}
        for arxiv_id in self.arxiv_list:
            arxiv_content = self._visit_article(arxiv_id)
            data[arxiv_id] = arxiv_content
        return data


if __name__ == '__main__':
    arxiv_list = ['0203079']
    arx = ArxivScraper(arxiv_list)
    print(arx._visit_article(arxiv_list[0]))


    ## OFFICIAL ARXIV API STUFF
    # import urllib, urllib.request
    # url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'
    # data = urllib.request.urlopen(url)
    # print(data.read().decode('utf-8'))