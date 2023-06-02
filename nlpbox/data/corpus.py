import requests
import tarfile
import zipfile
from io import BytesIO
from nlpbox.data.resources import get as get_resource


class Assin:
    url = 'http://nilc.icmc.usp.br/assin/assin.tar.gz'
    path = get_resource('assin')

    def download(self) -> None:
        if any(self.path.iterdir()):
            return
        with (requests.get(self.url, stream=True, timeout=10) as request,
              tarfile.open(fileobj=request.raw, mode="r:gz") as file):
            file.extractall(self.path, members=filter(lambda x: x.name.startswith('assin'), file))


class CoLA:
    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
    path = get_resource('cola')

    def download(self) -> None:
        if any(self.path.iterdir()):
            return
        with (requests.get(self.url, stream=True, timeout=10) as request,
              zipfile.ZipFile(file=BytesIO(request.content)) as file):
            file.extractall(self.path)
