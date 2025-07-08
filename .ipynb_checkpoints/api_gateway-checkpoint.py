from fastapi import FastAPI, Request
import httpx

app = FastAPI()

# روابط الخدمات الخلفية
SERVICE1_URL = "http://127.0.0.1:8000"
SERVICE2_URL = "http://127.0.0.1:8001"

@app.post("/api/query")
async def query_gateway(request: Request):
    data = await request.json()
    method = data.get("method")  # 'service1' أو 'service2'

    if method == "service1":
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{SERVICE1_URL}/query", json=data)
        return response.json()

    elif method == "service2":
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{SERVICE2_URL}/query", json=data)
        return response.json()

    else:
        return {"error": "طريقة غير مدعومة. الرجاء استخدام 'service1' أو 'service2'."}
