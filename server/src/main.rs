use actix::{Actor, StreamHandler};
use actix_web::{web, App, Error, HttpRequest, HttpResponse, HttpServer};
use actix_web_actors::ws;
use anyhow::Error as E;

// Inference stuff
use inference::{text_generation::TextGeneration, *};

/// Define HTTP actor
struct WSActor {
    text_pipeline: TextGeneration,
}

impl WSActor {
    fn text_generation(&mut self, prompt: &str, sample_len: usize) -> Result<(), E> {
        // Need to update the text generation code to return the generated text
        self.text_pipeline.run(prompt, sample_len)
    }
}

impl Actor for WSActor {
    type Context = ws::WebsocketContext<Self>;
}

/// Handler for ws::Message message
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WSActor {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        // Initialize the text generation pipeline
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => {
                // Extract the prompt from the request
                let prompt: &str = std::str::from_utf8(text.as_bytes()).unwrap();
                // TODO: Extract sample_len from reqeust?
                let r = self.text_generation(prompt, 256); // Hardcode 256 sample_len for the time being
                match r {
                    Ok(_) => (),
                    Err(e) => (), // TODO implement error handling
                }
                ctx.text(format!("Here is your input: {}", text))
            }
            Ok(ws::Message::Binary(bin)) => ctx.binary(bin),
            _ => (),
        }
    }
}

async fn index(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, Error> {
    let actor = WSActor {
        text_pipeline: TextGeneration::new(None, None, 299792458, None, None, 1.1, 64, false)
            .unwrap(),
    };
    let resp = ws::start(actor, &req, stream);
    println!("{:?}", resp);
    resp
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/chat", web::get().to(index)))
        .bind(("127.0.0.1", 8089))?
        .run()
        .await
}

