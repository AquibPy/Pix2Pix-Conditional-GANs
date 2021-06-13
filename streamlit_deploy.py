from streamlit import cli as stcli
import streamlit
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io
import sys


def main():
    streamlit.title('Pix2Pix GAN Model')

    # fastapi endpoint
    url = 'http://127.0.0.1:8000'
    endpoint = '/uploadfile/'

    streamlit.write('''Pix2Pix implemented in PyTorch.
            This streamlit example uses a FastAPI service as backend.
            Visit this URL at `:8000/docs` for FastAPI documentation.''') # description and instructions

    image = streamlit.file_uploader('insert image')  # image upload widget


    def process(image, server_url: str):

        m = MultipartEncoder(
            fields={'file': ('filename.jpg', image, 'image/jpeg')}
            )

        r = requests.post(server_url,
                        data=m,
                        headers={'Content-Type': m.content_type},
                        timeout=8000)

        return r


    if streamlit.button('Get generated'):

        if image == None:
            streamlit.write("Insert an image!")  # handle case with no image
        else:
            segments = process(image, url+endpoint)
            print(segments)
            segmented_image = Image.open(io.BytesIO(segments.content)).convert('RGB')
            print(segmented_image)
            streamlit.image([image, segmented_image], width=300)  # output dyptich


if __name__ == '__main__':
    if streamlit._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())