from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():

    return {
        "message": "Smart Water RL API Running"
    }


@app.get("/status")
def status():

    return {
        "status": "active",
        "model": "Q-Learning"
    }