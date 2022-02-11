from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel


import app.hf_pegasus as hf_pegasus

app = FastAPI()



class Text(BaseModel):
    text : str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/summarizeTextTest/")
async def summarize_text_test(input_text: Text):
  text_dict = input_text.dict()
  if text_dict:  # not null
     text_to_summarize = text_dict['text']
     doc = {
       'summary': text_to_summarize
     }
     return JSONResponse(content=jsonable_encoder(doc))
  return '', 204


@app.post("/summarizeText/")
async def summarize_text(input_text: Text):
  text_dict = input_text.dict()
  if text_dict:  # not null
     text_to_summarize = text_dict['text']
     summarized_text = hf_pegasus.summarize(text_to_summarize)
     doc = {
       'summary': summarized_text
     }
     return JSONResponse(content=jsonable_encoder(doc))
  return '', 204

@app.post("/summarizeTextLegal/")
async def summarize_text_legal(input_text: Text):
  text_dict = input_text.dict()
  if text_dict:  # not null
     text_to_summarize = text_dict['text']
     summarized_text = hf_pegasus.summarize_legal(text_to_summarize)
     doc = {
       'summary': summarized_text
     }
     return JSONResponse(content=jsonable_encoder(doc))
  return '', 204



