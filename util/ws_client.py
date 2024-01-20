import asyncio
import websockets
import json

async def connect_to_websocket():
    uri = "ws://localhost:8089/chat" 

    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")

        while True:
            message = input("Enter message (or 'exit' to quit): ")

            if message.lower() == 'exit':
                break

            await websocket.send(message)
            print(f"Sent: {message}")

            response = await websocket.recv()
            print(f"Received: {json.dumps(response)}")
            

asyncio.get_event_loop().run_until_complete(connect_to_websocket())
