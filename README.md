# Scratch

A simple drawing program.

## Examples

To start the program, run `cargo run`.

To export a drawing, run `cargo run -e <format> -n <name>`.
`<format>` can be "jpg", "png", or "none" (no export).
`<name>` is the name of the file to be exported.

## Installation

1. Clone the repository.
```bash
git clone https://github.com/your-username/scratch.git
```
2. Navigate to the cloned directory.
```bash
cd scratch
```
3. Build the project.
```bash
cargo build
```

## Usage

Once the project is built, you can run it with the following command:

```bash
cargo run
```

This will launch the Scratch application. You can draw with the left mouse button, and erase with the left mouse button while holding the 'E' key.

To save your drawing, run:

```bash
cargo run -e <format> -n <name>
```

This will save your drawing to a file named `<name>.<format>`.

For example, to save your drawing as a JPEG file named "my_drawing", run:

```bash
cargo run -e jpg -n my_drawing
```

The exported image will be saved to the directory `.local/scratch/notes/` in your home directory.
