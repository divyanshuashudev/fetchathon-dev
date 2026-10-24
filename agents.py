import os
from datetime import datetime
from uuid import uuid4

from openai import OpenAI
from uagents import Context, Protocol, Agent
from uagents.experimental.chat_agent.protocol import build_llm_message_history
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)

SYSTEM_PROMPT = (
    "You are Travent, an expert AI travel planner and budget concierge. "
    "When a user provides a trip query (origin, destination, budget, duration), break down their budget into: "
    "Transit (35%), Accommodation (35%), Local Rides (15%), and Dining (15%). "
    "Provide 3 distinct packages: Balanced Smart Saver, Luxury Stay Focus, and Backpacker/Foodie. "
    "Always provide booking recommendations for IRCTC/RedBus, MakeMyTrip/OYO, and Ola/Uber. "
    "If details are missing, politely ask the user for them."
)

client = OpenAI(
    base_url=os.environ.get("ASI1_BASE_URL", "https://api.asi1.ai/v1"),
    api_key=os.environ.get("ASI1_API_KEY", "sk_7026343497154a28a39be90fea47bd771559905953d34bda9e3f6629ae712ce1"),
)

agent = Agent()

protocol = Protocol(spec=chat_protocol_spec)

@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    text = msg.text()
    if not text:
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *build_llm_message_history(ctx),
    ]

    try:
        r = client.chat.completions.create(
            model="asi1",
            messages=messages,
            max_tokens=2048,
        )
        response = str(r.choices[0].message.content)
    except Exception as e:
        ctx.logger.exception('Error querying model')
        response = f"An error occurred while processing the request. Please try again later. {e}"

    await ctx.send(sender, create_text_chat(response))

@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass

agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
