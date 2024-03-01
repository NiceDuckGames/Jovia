# ✨Jovia - Open Source AI Framework for Godot 🎮

Welcome to Jovia! Jovia is an open-source AI framework designed to allow Godot developers to integrate local AI models into their games. Jovia, also provides a game design assistant named... Jovia of course, capable of executing instructions within the Godot engine.

## Feature Roadmap 🚀

- ☑️ **Interactive Assistant**: Interact with a game development assistant using natural language commands.
- ☑️ **Godot Integration**: Execute model inference from GDScript.
- ☑️ **Local Models**: Utilize locally hosted AI models for enhanced privacy and offline functionality.
- ☑️ **Text Generation**: Our first goal is to enable text generation via LLMs from the Godot game engine in an efficient manner.
- **Image Generation**: We plan to include image generation capabilities to aid in texture generation, texture repeating, among other things directly from Godot.
- **Mesh Generation**: Generate 3D meshes from text prompts straight from within the Godot game engine.
- **Future Hosted Option**: The project aims to offer a hosted option in the future for convenience and scalability.

## Getting Started 🛠️

To get started with Jovia, follow these steps:

- [ ] TODO

## Usage 🕹️

1. Install the Jovia Godot addon (available soon)
2. Interact with the assistant using natural language commands.
3. Use the Jovia API from GDscript to integrate local text-generation, image generation, and more into your game.
4. See the demo project under examples/ to get started (TODO)

### Building From Source

- Clone this repo `git clone https://github.com/NiceDuckGames/Jovia.git`
- Build the Godot Bindings `cargo build -p binding --release`
- TODO: hook into godot project

#### Cross-compilation
##### Windows

To cross compile for windows from Linux you will need to install a few additional things.

- *Windows build target* `rustup target add x86_64-pc-windows-gnu`
- *Mingw comilation toolchain* `sudo apt install mingw-w64`
- *Build the bindings* `cargo build -p bindings --release --target x86_64-pc-windows-gnu`

## Contributing 🤝

We are not currently accepting pull-requests on this repository from people outside of our team, though we probably will once the project reaches a certain level of stability and usefulness.

Instead if you would like to contribute, please open an issue for discussion first or create a [discussion](https://github.com/NiceDuckGames/jovia/discussions).

An additional way to contribute to the project is to contribute to the datasets used to create the custom models.

### Contributing to the Model Dataset
#### Command Sequence Data
To contribute your own command sequence data to the model dataset see the [Submitting Your Own Data](https://github.com/NiceDuckGames/jovia-command-dataset?tab=readme-ov-file#submitting-training-data) section of the ducky-command-dataset repository.

#### GDScript Data
If you wish to contribute GDScript source code for use in training the jovia codegen model see the [ducky-ai-codegen-optins](https://github.com/NiceDuckGames/jovia-codegen-optins) repository for instruction on how to do that.

## Sponsorship and Support 💡

Jovia is a passion-driven project, and your support is greatly appreciated. You can support the development of Jovia through the following platforms:

- **GitHub Sponsorship**: Become a sponsor on GitHub to support ongoing development and maintenance of Jovia. [Sponsor Jovia](https://github.com/sponsors/NiceDuckGames)

- **Patreon**: Join our Patreon community to get exclusive access to behind-the-scenes updates, early releases, and more. [Support Jovia on Patreon](https://patreon.com/niceduckgames)

Your support helps us continue to improve Jovia and provide valuable resources to the community.

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements 🙏

- Special thanks to [Godot Engine](https://godotengine.org/) for being awesome and empowering game developers!
- Shout out to the HuggingFace crew for making [Candle](https://github.com/huggingface/candle) which is an awesome ML framework for Rust.

## Support 📧

For support or inquiries, please contact us at support@niceduck.games.
