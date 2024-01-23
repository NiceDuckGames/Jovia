import lmql
import os
import json
import asyncio
import chromadb

from lmql.runtime.bopenai import get_stats
from aiohttp import web

from server.output import ChatMessageOutputWriter
from server.output import *
from queries.queries import *
from util.prompts import godot_assistant_system_prompt
import util.vector_ops as vops

commands_db_client = chromadb.PersistentClient(path="../commands-index/")
commands_collection = commands_db_client.get_collection(name="commands")

def parse_command(user_prompt):
     # Split the user input into command and message parts
    parts = user_prompt.split(': ', 1)
    
    # If the input doesn't contain a ': ', consider it as a chat message (default case)
    if len(parts) == 1:
        command = 'chat'
        message = parts[0]
    else:
        command, message = parts

    return command, message

async def command_handler(user_prompt):
    # We need consistency here. An approach to doing that is to force the user to use commands to interact
    command, user_message = parse_command(user_prompt)

    if command == 'task':
        print("HEEY")
        subtasks = await task_decomposition("", user_message)
        summary = await task_summarization("", subtasks)
        print(subtasks)
        print(summary)

        query_results = vops.retrieve_command_sequences(commands_collection, json.loads(subtasks))
        print(query_results)
    elif command == 'goal':
        "{:assistant} Unsupported command: {command}"
    elif command == 'docs':
        "{:assistant} Unsupported command: {command}"
    elif command == 'chat':
        "{:assistant} Unsupported command: {command}"
    else:
        "{:assistant} Unsupported command: {command}"

class ChatServer:
    """
    A minimal WebSocket-based chat server that serves a provided LMQL file 
    as a chat application, including a simple graphical user interface.

    All required web resources are located in the chat_assets/ subfolder.

    Parameters

    query: str or callable
        Either a path to an LMQL file or a callable that returns an LMQL query function. If a path is 
        provided, the file is read and parsed on each new chat connection.
    port: int
        The port to serve the chat application on. Default: 8089
    host: str
        The host to serve the chat application on. Default: 'localhost'

    """
    def __init__(self, port=8089, host='localhost'):
        self.port = port
        self.host = host

        # Holds async executors that map to each connection
        self.executors = []

    def make_query(self):
        '''
        Make a query from a LMQL file 
        '''
        if callable(self.query):
            # first check if we already have a query function
            return self.query
        else:
            # read and parse query function from self.file
            with open(self.query) as f:
                source = f.read()
                return lmql.query(source)

    async def handle_websocket_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print("Client connected!")
        try:
            chatbot = self.make_query()
        except Exception as e:
            import traceback
            await ws.send_str(json.dumps({
                "type": "error",
                "message": str(e) + "\n\n" + traceback.format_exc()
            }))
            await ws.close()
            return ws

        chat_executor = websocket_executor(ws)
        self.executors.append(chat_executor)
        
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                if msg.data == 'close':
                    await ws.close()
                else:
                    data = json.loads(msg.data)
                    print(data)
                    if data["type"] == "input":
                        # Handle the user input, this will invoke the correct query for the input
                        print("Got input, invoking LMQL query.")  
                        command_handler(data["text"])
                        #we got a user input message
                        # if chat_executor.user_input_fut is not None:
                            # fut = chat_executor.user_input_fut
                            # chat_executor.user_input_fut = None
                            # fut.set_result(data["text"])
                        # else:
                            # print("warning: got input but query is not waiting for input", flush=True)
            elif msg.type == web.WSMsgType.ERROR:
                print('ws connection closed with exception %s' %
                    ws.exception())
        chat_executor.chatbot_task.cancel()

        if chat_executor in self.executors:
            self.executors.remove(chat_executor)
        print("websocket connection closed", len(self.executors), "executors left", flush=True)
        return ws

    async def main(self):
        app = web.Application()
        
        # host chatbot query
        # websocket connection
        app.add_routes([web.get('/chat', self.handle_websocket_chat)])

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print('🤖 Your chatbot API is waiting for you at ws://{}:{}'.format(self.host, self.port), flush=True)
        
        # idle to keep the server running
        while True:
            await asyncio.sleep(3600)

    def run(self):
        asyncio.run(self.main())


class websocket_executor(ChatMessageOutputWriter):
    """
    Encapsulates the continuous execution of an LMQL query function and
    streams I/O via a provided WebSocket connection.
    """
    def __init__(self, ws):
        self.user_input_fut = None
        self.ws = ws
        # self.chatbot_task = asyncio.create_task(self.error_handling_query_call(query), name='chatbot query')
        self.message_id = 0
        
        self.current_message = ""

        self.list_of_subtasks = []

    def begin_message(self, variable):
        self.message_id += 1
        self.current_message = ""

    def stream_message(self, message):
        self.current_message = message

    def complete_message(self, message):
        self.current_message = ""

    async def error_handling_query_call(self, query):
        try:
            await query(output_writer=self)
        except Exception as e:
            print("error in chatbot query", flush=True)
            import traceback
            traceback.print_exc()
            await self.ws.send_str(json.dumps({
                "type": "error",
                "message": str(e)
            }))
            await self.ws.close()

    async def add_interpreter_head_state(self, variable, head, prompt, where, trace, is_valid, is_final, mask, num_tokens, program_variables): 

        python_scope = program_variables.python_scope
        # print(python_scope.keys())
        # print(python_scope.get("subtasks"))
    
        chunk = {
            "type": "response",
            "message_id": self.message_id,
            "data": {
                # always send the full prompt (you probably want to disable this for production use)
                #'prompt': prompt, 
                "variables": {
                    # client expects all message output to be stored in the "ANSWER" variable (query may use different variable names)
                    "ANSWER": self.current_message,
                    "summary": python_scope.get("summary"),
                    "subtasks": python_scope.get("subtasks"),
                },
            }
        }

        # Add a LMQL context variable if we have it
        variable_value = program_variables.get(variable, None)

        if variable_value is not None:
            chunk["data"]["variables"][variable] = variable_value 

        chunk_json = json.dumps(chunk)

        #await self.ws.send_str(chunk_json)

    async def input(self, *args):
        if self.user_input_fut is not None:
            return await self.user_input_fut
        else:
            self.user_input_fut = asyncio.get_event_loop().create_future()
            self.message_id += 1
            return await self.user_input_fut