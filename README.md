# TASTI Zoo-Simian Web Platform Integration

This project provides a modular structure for integrating machine learning models into a web API, with optional connectivity to the **Simian Web GUI**. It is designed to run the Zoo inference API in a Docker container, while Simian GUI operates in a separate container or environment.

##  Project Structure

```
tasti-web-platform-zoo-smian-integration/
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── generation/
│   │   ├── api/                  # API routes for model access
│   │   ├── model/                # Pydantic models for request/response schemas
│   │   └── ml_interface.py       # Simian Web integration interface (should be externalized)
```

##  About `ml_interface.py`

- Located in `src/generation/ml_interface.py`, this module manages the interaction between Zoo models and the Simian Web GUI.
- Since Simian runs its GUI in a separate container, this interface code **must be moved outside** the Zoo container to avoid conflicts.
- The API is designed to be consumed externally by Simian via endpoint URLs like: `http://localhost:8000/api`.

##  Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (if running locally without Docker)
- Simian Web GUI (setup separately)

### Run the Zoo API

```bash
docker-compose up --build
```

> **Note:** This container only includes the Zoo API. The Simian GUI must be deployed separately.

## Integration with Simian

- After launching the Simian Web GUI, configure its API endpoint to point to the Zoo container, for example:
  ```
  http://localhost:8000/api
  ```
- Ensure `ml_interface.py` or related interface logic is moved into the Simian workspace if needed.

## Tech Stack

- **Python** – FastAPI-based API
- **Pydantic** – For model schemas
- **Docker** – Containerized deployment
- **Simian Web GUI** – External user interface

## Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

© 2025 TASTI Web Platform – All rights reserved.
