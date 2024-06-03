from pydantic import Field
from pydantic_settings import BaseSettings


# TODO: refactor

# class ModelSettings(BaseSettings):
#     """Model settings used to load a Llama model."""
#
#     model: str = Field(
#         description="The path to the model to use for generating completions."
#     )
#
#
# class ServerSettings(BaseSettings):
#     """Server settings used to configure the FastAPI and Uvicorn server."""
#
#     host: str = Field(default="localhost", description="Listen address")
#     port: int = Field(default=8000, description="Listen port")
#

class Settings(BaseSettings):
    model: str = Field(
        description="The path to the model to use for generating completions."
    )
