import uvicorn
import os
from api import app

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )