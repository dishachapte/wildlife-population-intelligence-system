from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from tortoise import Tortoise

from app.core.config import settings
from app.models.monitoring import MonitoringSite


async def init_db():
    # PostgreSQL
    await Tortoise.init(
        db_url=settings.POSTGRES_DSN,
        modules={"models": ["app.models.user"]},
    )

    await Tortoise.generate_schemas()

    # MongoDB
    client = AsyncIOMotorClient(settings.MONGO_URI)

    await init_beanie(
        database=client[settings.MONGO_DB_NAME],
        document_models=[MonitoringSite],
    )


async def close_db():
    await Tortoise.close_connections()

