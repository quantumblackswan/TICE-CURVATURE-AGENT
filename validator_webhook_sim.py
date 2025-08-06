from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()

class ValidatorPayload(BaseModel):
    validator_id: str
    tau: float
    delta_psi: float
    eta: float

@app.post("/api/validator_ping")
async def ping(payload: ValidatorPayload):
    Λ = (payload.delta_psi ** 2 * payload.tau) / payload.eta
    XP = round(Λ * payload.tau * 0.91, 3)
    print(f"[ΞΛ Webhook] Validator: {payload.validator_id} | Λ: {Λ:.3f} | XP: {XP}")
    
    return {
        "validator_id": payload.validator_id,
        "Λ": Λ,
        "XP_earned": XP,
        "timestamp": time.time(),
        "status": "Validator curvature attested"
    }

@app.get("/")
def root():
    return {"message": "ΞΛ Validator Webhook Running", "status": "OK"}

if __name__ == "__main__":
    uvicorn.run("validator_webhook_sim:app", host="0.0.0.0", port=8080, reload=True)
