from fastapi import APIRouter
from api.main import search

router = APIRouter()

router.add_api_route("/search", search, methods=["POST"])
