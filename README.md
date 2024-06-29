it is a text-image generation using stable difussion
***
### requirements 
- python 3.8 or latter
- install python from pthon page [https://www.python.org/downloads/]
***
#### installations
- cd src

- pip install -r requirements.txt
***

copy .env.example .env
***
uvicorn app:app --reload
***
download ngrok from  agrok website
- then open command prompt
- write: ngrok http 8000 
- you will take a url use it while uvicorn app:app --reload run
-  
