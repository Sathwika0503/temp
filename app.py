from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the conditions list
with open('model/conditions_list.json', 'r') as f:
    conditions_list = json.load(f)

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('model/clinicalbert_model1')
tokenizer = AutoTokenizer.from_pretrained('model/clinicalbert_tokenizer1')

# Load the dataset for drug suggestions
df = pd.read_csv('data/drugsComTrain.tsv', sep='\t')

# Function to extract top drugs
def top_drugs_extractor(condition):
    criteria = [
        (9, 100),
        (9, 50),
        (8, 100),
        (8, 50)
    ]
    
    drug_lst = []
    
    for rating_threshold, useful_count_threshold in criteria:
        if len(drug_lst) < 3:
            df_filtered = df[(df['rating'] >= rating_threshold) & (df['usefulCount'] >= useful_count_threshold)]
            df_filtered = df_filtered[df_filtered['condition'] == condition]
            df_filtered = df_filtered.sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
            
            for drug in df_filtered['drugName'].tolist():
                if drug not in drug_lst:
                    drug_lst.append(drug)
                    if len(drug_lst) == 3:
                        break
    
    return drug_lst

# Function to predict condition and get top drugs
def predict_and_suggest_drugs(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    predicted_condition = conditions_list[predicted_class]
    top_drugs = top_drugs_extractor(predicted_condition)
    return predicted_condition, top_drugs

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == "admin@gmail.com" and password == "admin":
        response = RedirectResponse(url="/prediction", status_code=302)
        response.set_cookie(key="username", value=username)
        return response
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})

@app.get("/prediction", response_class=HTMLResponse)
async def get_prediction_page(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request})

@app.post("/prediction", response_class=HTMLResponse)
async def get_prediction(request: Request, review: str = Form(...)):
    predicted_condition, top_drugs = predict_and_suggest_drugs(review)
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "predicted_condition": predicted_condition,
        "top_drugs": top_drugs
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)