from pydantic import BaseModel


class HealthResponse(BaseModel):
    # Response status, for example ok or error.
    status: str
    # Text description for the status.
    message: str


class HelloRequest(BaseModel):
    # The name passed in by the user.
    name: str


class HelloResponse(BaseModel):
    # The greeting message returned by the service.
    message: str
