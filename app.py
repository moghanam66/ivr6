import sys
import traceback
from datetime import datetime

from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import (BotFrameworkAdapter, BotFrameworkAdapterSettings,
                             TurnContext)
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.schema import Activity, ActivityTypes

from bot import MyBot
from config import DefaultConfig

CONFIG = DefaultConfig()

SETTINGS = BotFrameworkAdapterSettings(CONFIG.APP_ID, CONFIG.APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)

# ee
async def on_error(context: TurnContext, error: Exception):

    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity(
        "To continue to run this bot, please fix the bot source code."
    )

    if context.activity.channel_id == "emulator":

        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
        
        await context.send_activity(trace_activity)

ADAPTER.on_turn_error = on_error

BOT = MyBot()

async def messages(req: Request) -> Response:

    if "application/json" in req.headers["Content-Type"]:
        body = await req.json()
    else:
        return Response(status=415)

    activity = Activity().deserialize(body)
    auth_header = req.headers["Authorization"] if "Authorization" in req.headers else ""

    response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
    if response:
        return json_response(data=response.body, status=response.status)
    return Response(status=201)

# Modifying the below code

def init_func(argv):
    APP = web.Application(middlewares=[aiohttp_error_middleware])
    APP.router.add_post("/api/messages", messages)
    return APP
if __name__ == "__main__":
    APP = init_func(None)

    try:
        web.run_app( APP , host="0.0.0.0", port=CONFIG.PORT)
    except Exception as error:
        raise error
