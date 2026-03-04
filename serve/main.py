from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import analysis, training, inference

app = FastAPI(title="AutoApex TimeSeries Analysis Platform")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(analysis.router)
app.include_router(training.router)
app.include_router(inference.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)