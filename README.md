it is a text-image generation using stable difussion
***
### requirements 
- python 3.8 or latter
- install python from pthon page [https://www.python.org/downloads/]
***
#### installations
- cd src
pip install -r requirements.txt
- 
***


- uvicorn main:app --reload



=======
copy .env.example .env
***
uvicorn app:app --reload
***
important note <<very important>>
if you run the code and test the url and gives you "torch doesnt compile with cuda"
download cuda toklit from cuda website and choose the cuda that support your operating system 
after that download pytorch that compile with cuda

***
download ngrok from  agrok website
- then open command prompt
- write: ngrok http 8000 
- you will take a url use it while uvicorn app:app --reload run
-  
