use std::error::Error;
use std::ffi::{CStr, CString};
use std::num::NonZeroU32;
use std::ops::Deref;

use image::{ImageBuffer, Rgb};

use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::Window;

use glutin::config::{Config, ConfigTemplateBuilder};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentContext, PossiblyCurrentContext, Version,
};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{Surface, SwapInterval, WindowSurface};

use glutin_winit::{DisplayBuilder, GlWindow};

pub mod gl {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));

    pub use Gles2 as Gl;
}

enum ExportType {
    None,
    JPG,
    PNG,
}

struct Flags {
    pub export_type: ExportType,
    pub name: String,
}

fn man() {}

fn parse_flags() -> Result<Flags, std::io::Error> {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = Flags {
        export_type: ExportType::JPG,
        name: "".to_string(),
    };

    if args.len() < 1 {
        panic!("Usage: {} [-sef]", args[0]);
    }

    for arg in args.iter().skip(1) {
        if arg.starts_with("-") && !arg.starts_with("--") {
            for (i, c) in arg.chars().enumerate().skip(1) {
                match c {
                    'e' => {
                        if i + 1 < args.len() {
                            flags.export_type = match args[i + 1].as_str() {
                                "jpg" => ExportType::JPG,
                                "png" => ExportType::PNG,
                                "none" => ExportType::None,
                                _ => {
                                    return Err(std::io::Error::new(
                                        std::io::ErrorKind::InvalidInput,
                                        "Invalid argument for flag -e",
                                    ));
                                }
                            };
                        } else {
                            man();
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "Missing argument for flag -e",
                            ));
                        }
                    }
                    'n' => {
                        if i + 1 < args.len() {
                            flags.name = args[i + 1].clone();
                        } else {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "Missing argument for flag -n",
                            ));
                        }
                    }
                    _ => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("Invalid flag: {}", c),
                        ));
                    }
                }
            }
        }
    }

    Ok(flags)
}

fn main() -> Result<(), Box<dyn Error>> {
    let flags = parse_flags()?;
    let event_loop = winit::event_loop::EventLoop::new().unwrap();

    let window_attributes = Window::default_attributes()
        .with_transparent(true)
        .with_title("Scratch");

    let template = ConfigTemplateBuilder::new()
        .with_alpha_size(8)
        .with_transparency(cfg!(cgl_backend));

    let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_attributes));

    let mut app = App::new(template, display_builder);
    event_loop.run_app(&mut app)?;

    let save_as = if flags.name.is_empty() {
        chrono::Local::now().timestamp_micros().to_string()
    } else {
        flags.name
    };

    let home_dir = match std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .or_else(|_| {
            std::env::var("HOMEDRIVE").and_then(|homedrive| {
                std::env::var("HOMEPATH").map(|homepath| format!("{}{}", homedrive, homepath))
            })
        }) {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => panic!("Failed to get home directory"),
    };

    let save_dir = home_dir.join(".local/scratch/notes/");
    std::fs::create_dir_all(&save_dir)?;

    let format = match flags.export_type {
        ExportType::JPG => "jpg",
        ExportType::PNG => "png",
        ExportType::None => "none",
    };

    let save_path = save_dir.join(format!("{}.{}", save_as, format));

    let (width, height, pixels) = app.final_pixels.unwrap();
    let image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width as u32, height as u32, pixels)
        .expect("Failed to create image buffer");

    image.save(save_path.clone())?;
    println!("Saved to {}", save_path.display());

    app.exit_state
}

fn window_coords_to_ndc(x: f64, y: f64, width: u32, height: u32) -> [f32; 2] {
    let x = (2.0 * x as f32 / width as f32) - 1.0;
    let y = 1.0 - (2.0 * y as f32 / height as f32);

    [x, y]
}

fn ndc_origin() -> [f32; 2] {
    [-1.0, 1.0]
}

fn distance(p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    (dx * dx + dy * dy).sqrt()
}

fn get_default_cursor(input_mode: InputMode) -> winit::window::CursorIcon {
    match input_mode {
        InputMode::Draw => winit::window::CursorIcon::Crosshair,
        InputMode::Select => winit::window::CursorIcon::Default,
        InputMode::Erase => winit::window::CursorIcon::Cell,
        InputMode::Pan => winit::window::CursorIcon::Grab,
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let (mut window, gl_config) = match self.display_builder.clone().build(
            event_loop,
            self.template.clone(),
            gl_config_picker,
        ) {
            Ok(ok) => ok,
            Err(e) => {
                self.exit_state = Err(e);
                event_loop.exit();
                return;
            }
        };

        println!("Picked a config with {} samples", gl_config.num_samples());

        let raw_window_handle = window
            .as_ref()
            .and_then(|window| window.window_handle().ok())
            .map(|handle| handle.as_raw());

        // XXX The display could be obtained from any object created by it, so we can
        // query it from the config.
        let gl_display = gl_config.display();

        // The context creation part.
        let context_attributes = ContextAttributesBuilder::new().build(raw_window_handle);

        // Since glutin by default tries to create OpenGL core context, which may not be
        // present we should try gles.
        let fallback_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(None))
            .build(raw_window_handle);

        // There are also some old devices that support neither modern OpenGL nor GLES.
        // To support these we can try and create a 2.1 context.
        let legacy_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
            .build(raw_window_handle);

        // Reuse the uncurrented context from a suspended() call if it exists, otherwise
        // this is the first time resumed() is called, where the context still
        // has to be created.
        let not_current_gl_context = self
            .not_current_gl_context
            .take()
            .unwrap_or_else(|| unsafe {
                gl_display
                    .create_context(&gl_config, &context_attributes)
                    .unwrap_or_else(|_| {
                        gl_display
                            .create_context(&gl_config, &fallback_context_attributes)
                            .unwrap_or_else(|_| {
                                gl_display
                                    .create_context(&gl_config, &legacy_context_attributes)
                                    .expect("failed to create context")
                            })
                    })
            });

        let window = window.take().unwrap_or_else(|| {
            let window_attributes = Window::default_attributes()
                .with_transparent(true)
                .with_title("Glutin triangle gradient example (press Escape to exit)");
            glutin_winit::finalize_window(event_loop, window_attributes, &gl_config).unwrap()
        });

        window.set_cursor(get_default_cursor(InputMode::Draw));

        let attrs = window
            .build_surface_attributes(Default::default())
            .expect("Failed to build surface attributes");
        let gl_surface = unsafe {
            gl_config
                .display()
                .create_window_surface(&gl_config, &attrs)
                .unwrap()
        };

        // Make it current.
        let gl_context = not_current_gl_context.make_current(&gl_surface).unwrap();

        // The context needs to be current for the Renderer to set up shaders and
        // buffers. It also performs function loading, which needs a current context on
        // WGL.
        self.renderer
            .get_or_insert_with(|| Renderer::new(&gl_display));

        // Try setting vsync.
        if let Err(res) = gl_surface
            .set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
        {
            eprintln!("Error setting vsync: {res:?}");
        }

        assert!(self
            .state
            .replace(AppState {
                gl_context,
                gl_surface,
                window,
                cursor_position: (0.0, 0.0),
                strokes: Vec::new(),
                current_stroke: None,
                input_mode: InputMode::Draw,
                clear_on_next_draw: false,
                mouse_down: false,
                pan_state: PanState {
                    current_pan_start: None,
                    current_pan_position: None,
                },
            })
            .is_none());
    }

    fn suspended(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // This event is only raised on Android, where the backing NativeWindow for a GL
        // Surface can appear and disappear at any moment.
        println!("Android window removed");

        // Destroy the GL Surface and un-current the GL Context before ndk-glue releases
        // the window back to the system.
        let gl_context = self.state.take().unwrap().gl_context;
        assert!(self
            .not_current_gl_context
            .replace(gl_context.make_not_current().unwrap())
            .is_none());
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) if size.width != 0 && size.height != 0 => {
                if let Some(AppState {
                    gl_context,
                    gl_surface,
                    ..
                }) = self.state.as_ref()
                {
                    gl_surface.resize(
                        gl_context,
                        NonZeroU32::new(size.width).unwrap(),
                        NonZeroU32::new(size.height).unwrap(),
                    );
                    let renderer = self.renderer.as_ref().unwrap();
                    renderer.resize(size.width as i32, size.height as i32);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                if self.final_pixels.is_none() {
                    self.final_pixels = Some(
                        self.renderer
                            .as_mut()
                            .unwrap()
                            .draw_to_pixels(self.state.as_ref().unwrap().strokes.clone()),
                    );
                }

                event_loop.exit()
            }
            // i'm positive there's a better way for this but i'm not familiar enough with rust
            // syntax
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyD),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(AppState { window, .. }) = self.state.as_ref() {
                    window.set_cursor(get_default_cursor(InputMode::Draw));
                    if let Some(AppState { input_mode, .. }) = self.state.as_mut() {
                        *input_mode = InputMode::Draw;
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyE),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(AppState { window, .. }) = self.state.as_ref() {
                    window.set_cursor(get_default_cursor(InputMode::Erase));
                    if let Some(AppState { input_mode, .. }) = self.state.as_mut() {
                        *input_mode = InputMode::Erase;
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyS),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(AppState { window, .. }) = self.state.as_ref() {
                    window.set_cursor(get_default_cursor(InputMode::Select));
                    if let Some(AppState { input_mode, .. }) = self.state.as_mut() {
                        *input_mode = InputMode::Select;
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyH),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(AppState { window, .. }) = self.state.as_ref() {
                    window.set_cursor(get_default_cursor(InputMode::Pan));
                    if let Some(AppState { input_mode, .. }) = self.state.as_mut() {
                        *input_mode = InputMode::Pan;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(AppState {
                    gl_context,
                    gl_surface,
                    window,
                    current_stroke,
                    strokes,
                    clear_on_next_draw,
                    pan_state,
                    ..
                }) = self.state.as_mut()
                {
                    let mut latest = Vec::new();
                    let strokes_to_render = if *clear_on_next_draw {
                        strokes
                    } else {
                        if let Some(stroke) = current_stroke {
                            latest.push(stroke.clone());
                        } else if !strokes.is_empty() {
                            latest.push(strokes.last().unwrap().clone());
                        }

                        &mut latest
                    };

                    if strokes_to_render.is_empty() && !*clear_on_next_draw {
                        return;
                    }

                    let (width, height): (u32, u32) = window.inner_size().into();
                    let origin = ndc_origin();
                    let strokes_to_render = strokes_to_render
                        .into_iter()
                        .map(|s| {
                            let mut points = s.points.clone();
                            for point in points.iter_mut() {
                                if let Some(current_pan_start) = pan_state.current_pan_start {
                                    if let Some(current_pan_position) =
                                        pan_state.current_pan_position
                                    {
                                        let dxdy = window_coords_to_ndc(
                                            current_pan_position.0 - current_pan_start.0,
                                            current_pan_position.1 - current_pan_start.1,
                                            width,
                                            height,
                                        );

                                        let (dx, dy) = (
                                            dxdy[0] as f32 - origin[0],
                                            dxdy[1] as f32 - origin[1],
                                        );

                                        point[0] += dx as f32;
                                        point[1] += dy as f32;
                                    }
                                }
                            }

                            Stroke {
                                points,
                                color: s.color,
                            }
                        })
                        .collect::<Vec<_>>();

                    let renderer = self.renderer.as_mut().unwrap();
                    renderer.draw(&strokes_to_render, *clear_on_next_draw);
                    window.request_redraw();

                    gl_surface.swap_buffers(gl_context).unwrap();

                    *clear_on_next_draw = false;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(AppState {
                    cursor_position,
                    window,
                    current_stroke,
                    strokes,
                    input_mode,
                    clear_on_next_draw,
                    mouse_down,
                    pan_state,
                    ..
                }) = self.state.as_mut()
                {
                    *cursor_position = (position.x, position.y);
                    let (width, height): (u32, u32) = window.inner_size().into();
                    if let Some(stroke) = current_stroke {
                        stroke
                            .points
                            .push(window_coords_to_ndc(position.x, position.y, width, height));
                    }

                    // current method of checking if the cursor is over a stroke
                    // probably will need reworked
                    let converted_cursor_position =
                        window_coords_to_ndc(position.x, position.y, width, height);
                    if input_mode == &InputMode::Select {
                        for stroke in strokes.iter() {
                            for point in stroke.points.iter() {
                                if distance(*point, converted_cursor_position) < 0.02 {
                                    window.set_cursor(winit::window::CursorIcon::Grab);
                                    return;
                                }
                            }
                        }
                    } else if input_mode == &InputMode::Erase && *mouse_down && !*clear_on_next_draw
                    {
                        let mut index_to_remove = 0;
                        for (i, stroke) in strokes.iter().enumerate() {
                            for point in stroke.points.iter() {
                                if distance(*point, converted_cursor_position) < 0.02 {
                                    *clear_on_next_draw = true;
                                    index_to_remove = i;
                                    break;
                                }
                            }

                            if *clear_on_next_draw {
                                break;
                            }
                        }

                        if *clear_on_next_draw {
                            strokes.remove(index_to_remove);
                        }
                    } else if input_mode == &InputMode::Pan && *mouse_down {
                        if pan_state.current_pan_start.is_some() {
                            pan_state.current_pan_position = Some(*cursor_position);
                        }

                        *clear_on_next_draw = true;
                    }

                    window.set_cursor(get_default_cursor(input_mode.clone()));
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // begin a new stroke
                if let Some(AppState {
                    window,
                    cursor_position,
                    current_stroke,
                    input_mode,
                    strokes,
                    clear_on_next_draw,
                    mouse_down,
                    pan_state,
                    ..
                }) = self.state.as_mut()
                {
                    let (x, y) = cursor_position;
                    let (width, height): (u32, u32) = window.inner_size().into();
                    if input_mode == &InputMode::Draw {
                        let stroke = Stroke {
                            points: vec![window_coords_to_ndc(*x, *y, width, height)],
                            color: [1.0, 1.0, 1.0],
                        };

                        *current_stroke = Some(stroke);
                    } else if input_mode == &InputMode::Erase {
                        let converted_cursor_position = window_coords_to_ndc(*x, *y, width, height);
                        let mut index_to_remove = 0;
                        for (i, stroke) in strokes.iter().enumerate() {
                            for point in stroke.points.iter() {
                                if distance(*point, converted_cursor_position) < 0.02 {
                                    *clear_on_next_draw = true;
                                    index_to_remove = i;
                                    break;
                                }
                            }

                            if *clear_on_next_draw {
                                break;
                            }
                        }

                        if *clear_on_next_draw {
                            strokes.remove(index_to_remove);
                        }
                    } else if input_mode == &InputMode::Pan && pan_state.current_pan_start.is_none()
                    {
                        pan_state.current_pan_start = Some(*cursor_position);
                    }

                    *mouse_down = true;

                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                // end the current stroke
                if let Some(AppState {
                    current_stroke,
                    strokes,
                    mouse_down,
                    window,
                    pan_state,
                    ..
                }) = self.state.as_mut()
                {
                    if let Some(stroke) = current_stroke {
                        strokes.push(stroke.clone());
                        *current_stroke = None;
                    }

                    if let Some(current_pan_position) = pan_state.current_pan_position {
                        let (width, height): (u32, u32) = window.inner_size().into();
                        let origin = ndc_origin();

                        let dxdy = window_coords_to_ndc(
                            current_pan_position.0 - pan_state.current_pan_start.unwrap().0,
                            current_pan_position.1 - pan_state.current_pan_start.unwrap().1,
                            width,
                            height,
                        );

                        let dxdy = [
                            dxdy[0] as f64 - origin[0] as f64,
                            dxdy[1] as f64 - origin[1] as f64,
                        ];

                        for stroke in strokes.iter_mut() {
                            for point in stroke.points.iter_mut() {
                                point[0] += dxdy[0] as f32;
                                point[1] += dxdy[1] as f32;
                            }
                        }

                        pan_state.current_pan_start = None;
                        pan_state.current_pan_position = None;
                    }

                    window.request_redraw();

                    *mouse_down = false;
                }
            }
            _ => (),
        }
    }
}

struct App {
    template: ConfigTemplateBuilder,
    display_builder: DisplayBuilder,
    exit_state: Result<(), Box<dyn Error>>,
    not_current_gl_context: Option<NotCurrentContext>,
    renderer: Option<Renderer>,
    // NOTE: `AppState` carries the `Window`, thus it should be dropped after everything else.
    state: Option<AppState>,
    final_pixels: Option<(i32, i32, Vec<u8>)>,
}

impl App {
    fn new(template: ConfigTemplateBuilder, display_builder: DisplayBuilder) -> Self {
        Self {
            template,
            display_builder,
            exit_state: Ok(()),
            not_current_gl_context: None,
            state: None,
            renderer: None,
            final_pixels: None,
        }
    }
}

#[derive(Debug, Clone)]
struct Stroke {
    points: Vec<[f32; 2]>,
    color: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum InputMode {
    Draw,
    Select,
    Erase,
    Pan,
}

struct PanState {
    current_pan_start: Option<(f64, f64)>,
    current_pan_position: Option<(f64, f64)>,
}

struct AppState {
    gl_context: PossiblyCurrentContext,
    gl_surface: Surface<WindowSurface>,
    cursor_position: (f64, f64),
    pan_state: PanState,
    strokes: Vec<Stroke>,
    current_stroke: Option<Stroke>,
    input_mode: InputMode,
    clear_on_next_draw: bool,
    mouse_down: bool,
    // NOTE: Window should be dropped after all resources created using its
    // raw-window-handle.
    window: Window,
}

// Find the config with the maximum number of samples, so our triangle will be
// smooth.
fn gl_config_picker(configs: Box<dyn Iterator<Item = Config> + '_>) -> Config {
    configs
        .reduce(|accum, config| {
            let transparency_check = config.supports_transparency().unwrap_or(false)
                & !accum.supports_transparency().unwrap_or(false);

            if transparency_check || config.num_samples() > accum.num_samples() {
                config
            } else {
                accum
            }
        })
        .unwrap()
}

struct Renderer {
    program: gl::types::GLuint,
    vao: gl::types::GLuint,
    vbo: gl::types::GLuint,
    gl: gl::Gl,
}

impl Renderer {
    fn new<D: GlDisplay>(gl_display: &D) -> Self {
        unsafe {
            let gl = gl::Gl::load_with(|symbol| {
                let symbol = CString::new(symbol).unwrap();
                gl_display.get_proc_address(symbol.as_c_str()).cast()
            });

            if let Some(renderer) = get_gl_string(&gl, gl::RENDERER) {
                println!("Running on {}", renderer.to_string_lossy());
            }
            if let Some(version) = get_gl_string(&gl, gl::VERSION) {
                println!("OpenGL Version {}", version.to_string_lossy());
            }

            if let Some(shaders_version) = get_gl_string(&gl, gl::SHADING_LANGUAGE_VERSION) {
                println!("Shaders version on {}", shaders_version.to_string_lossy());
            }

            let vertex_shader = create_shader(&gl, gl::VERTEX_SHADER, VERTEX_SHADER_SOURCE);
            let fragment_shader = create_shader(&gl, gl::FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);

            let program = gl.CreateProgram();

            gl.AttachShader(program, vertex_shader);
            gl.AttachShader(program, fragment_shader);

            gl.LinkProgram(program);

            gl.UseProgram(program);

            gl.DeleteShader(vertex_shader);
            gl.DeleteShader(fragment_shader);

            let mut vao = std::mem::zeroed();
            gl.GenVertexArrays(1, &mut vao);
            gl.BindVertexArray(vao);

            let mut vbo = std::mem::zeroed();
            gl.GenBuffers(1, &mut vbo);
            gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl.BufferData(
                gl::ARRAY_BUFFER,
                (VERTEX_DATA.len() * std::mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                VERTEX_DATA.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            let pos_attrib = gl.GetAttribLocation(program, b"position\0".as_ptr() as *const _);
            let color_attrib = gl.GetAttribLocation(program, b"color\0".as_ptr() as *const _);
            gl.VertexAttribPointer(
                pos_attrib as gl::types::GLuint,
                2,
                gl::FLOAT,
                0,
                5 * std::mem::size_of::<f32>() as gl::types::GLsizei,
                std::ptr::null(),
            );
            gl.VertexAttribPointer(
                color_attrib as gl::types::GLuint,
                3,
                gl::FLOAT,
                0,
                5 * std::mem::size_of::<f32>() as gl::types::GLsizei,
                (2 * std::mem::size_of::<f32>()) as *const () as *const _,
            );
            gl.EnableVertexAttribArray(pos_attrib as gl::types::GLuint);
            gl.EnableVertexAttribArray(color_attrib as gl::types::GLuint);

            Self {
                program,
                vao,
                vbo,
                gl,
            }
        }
    }

    // NOTE: only draws to whatever framebuffer is currently bound
    fn draw(&mut self, strokes: &Vec<Stroke>, clear: bool) {
        if clear {
            unsafe {
                self.gl.Clear(gl::COLOR_BUFFER_BIT);
            }
        }

        for stroke in strokes.iter() {
            let mut vertices = Vec::new();
            for point in stroke.points.iter() {
                vertices.push(point[0]);
                vertices.push(point[1]);
                vertices.push(stroke.color[0]);
                vertices.push(stroke.color[1]);
                vertices.push(stroke.color[2]);
            }

            unsafe {
                self.gl.BindBuffer(gl::ARRAY_BUFFER, self.vbo);
                self.gl.BufferData(
                    gl::ARRAY_BUFFER,
                    (vertices.len() * std::mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                    vertices.as_ptr() as *const _,
                    gl::DYNAMIC_DRAW,
                );
                self.gl.UseProgram(self.program);
                self.gl.BindVertexArray(self.vao);
                self.gl.BindBuffer(gl::ARRAY_BUFFER, self.vbo);

                self.gl.LineWidth(3.0);
                self.gl
                    .DrawArrays(gl::LINE_STRIP, 0, (vertices.len() / 5) as i32);
            }
        }
    }

    fn resize(&self, width: i32, height: i32) {
        unsafe {
            self.gl.Viewport(0, 0, width, height);
        }
    }

    fn get_viewport_size(&self) -> (i32, i32) {
        let mut viewport = [0; 4];
        unsafe {
            self.gl.GetIntegerv(gl::VIEWPORT, viewport.as_mut_ptr());
        }

        (viewport[2], viewport[3])
    }

    // NOTE: this function is assumed to _only_ be called on program exit
    fn draw_to_pixels(&mut self, mut strokes: Vec<Stroke>) -> (i32, i32, Vec<u8>) {
        let (width, height) = self.get_viewport_size();

        let mut top_left = [f32::MAX, f32::MIN];
        let mut bottom_right = [f32::MIN, f32::MAX];

        // stroke points need mapped from ndc [-1, 1] to viewport coordinates
        // since not all strokes will be within the bounds of the viewport,
        // the resulting framebuffer height/width will need to be adjusted
        // to include all strokes
        for stroke in strokes.iter() {
            for point in stroke.points.iter() {
                let [x, y] = point;

                top_left[0] = top_left[0].min(*x);
                top_left[1] = top_left[1].max(*y);
            }
        }

        for stroke in strokes.iter_mut() {
            for point in stroke.points.iter_mut() {
                point[0] -= 1.0 + top_left[0];
                point[1] -= top_left[1] - 1.0;

                bottom_right[0] = bottom_right[0].max(point[0]);
                bottom_right[1] = bottom_right[1].min(point[1]);
            }
        }

        top_left = [0.0, 0.0];

        let ndc_width = std::cmp::max(((bottom_right[0] - top_left[0]) / 2.0).ceil() as usize, 1);
        let ndc_height = std::cmp::max(((top_left[1] - bottom_right[1]) / 2.0).ceil() as usize, 1);

        let mut fbo = 0;
        let mut texture = 0;

        let final_width = width as usize * ndc_width;
        let final_height = height as usize * ndc_height;
        let final_size = final_width * final_height * 3;

        println!(
            "creating image with dims ({}, {})",
            final_width, final_height
        );

        let mut final_image = vec![0u8; final_size];
        unsafe {
            self.gl.GenFramebuffers(1, &mut fbo);
            self.gl.BindFramebuffer(gl::FRAMEBUFFER, fbo);

            self.gl.GenTextures(1, &mut texture);
            self.gl.BindTexture(gl::TEXTURE_2D, texture);
            self.gl.TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGB as i32,
                width,
                height,
                0,
                gl::RGB,
                gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
            self.gl
                .TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            self.gl
                .TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

            self.gl.FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                texture,
                0,
            );

            if self.gl.CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
                panic!("Framebuffer is not complete!");
            }

            for y in 0..ndc_height {
                for x in 0..ndc_width {
                    let offset_strokes = strokes
                        .iter()
                        .map(|s| {
                            let mut points = s.points.clone();
                            for point in points.iter_mut() {
                                point[0] -= (x * 2) as f32;
                                point[1] += (y * 2) as f32;
                            }

                            Stroke {
                                points,
                                color: s.color,
                            }
                        })
                        .collect::<Vec<_>>();
                    self.draw(&offset_strokes, true);
                    self.gl.Flush();

                    let mut pixels = vec![0u8; (width * height * 3) as usize];
                    self.gl.ReadPixels(
                        0,
                        0,
                        width,
                        height,
                        gl::RGB,
                        gl::UNSIGNED_BYTE,
                        pixels.as_mut_ptr() as *mut _,
                    );

                    let row_size = (width * 3) as usize;
                    let mut temp_row = vec![0u8; row_size];
                    for y in 0..height as usize / 2 {
                        let top_row_start = y * row_size;
                        let bottom_row_start = (height as usize - 1 - y) * row_size;

                        temp_row.copy_from_slice(&pixels[top_row_start..top_row_start + row_size]);

                        pixels.copy_within(
                            bottom_row_start..bottom_row_start + row_size,
                            top_row_start,
                        );

                        pixels[bottom_row_start..bottom_row_start + row_size]
                            .copy_from_slice(&temp_row);
                    }

                    let height = height as usize;
                    let width = width as usize;
                    for r in 0..height {
                        for c in 0..width {
                            let i = ((y * height + r) * final_width) * 3;
                            let j = (x * width + c) * 3;

                            let final_index = i + j;
                            let tile_index = (r * width + c) * 3;

                            for k in 0..3 {
                                final_image[final_index + k] = pixels[tile_index + k];
                            }
                        }
                    }
                }
            }

            self.gl.BindFramebuffer(gl::FRAMEBUFFER, 0);
            println!("framebuffer unbound");
        }

        (
            (width as usize * ndc_width) as i32,
            (height as usize * ndc_height) as i32,
            final_image,
        )
    }
}

impl Deref for Renderer {
    type Target = gl::Gl;

    fn deref(&self) -> &Self::Target {
        &self.gl
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.gl.DeleteProgram(self.program);
            self.gl.DeleteBuffers(1, &self.vbo);
            self.gl.DeleteVertexArrays(1, &self.vao);
        }
    }
}

unsafe fn create_shader(
    gl: &gl::Gl,
    shader: gl::types::GLenum,
    source: &[u8],
) -> gl::types::GLuint {
    let shader = gl.CreateShader(shader);
    gl.ShaderSource(
        shader,
        1,
        [source.as_ptr().cast()].as_ptr(),
        std::ptr::null(),
    );
    gl.CompileShader(shader);
    shader
}

fn get_gl_string(gl: &gl::Gl, variant: gl::types::GLenum) -> Option<&'static CStr> {
    unsafe {
        let s = gl.GetString(variant);
        (!s.is_null()).then(|| CStr::from_ptr(s.cast()))
    }
}

#[rustfmt::skip]
static VERTEX_DATA: [f32; 15] = [
    -0.5, -0.5,  1.0,  0.0,  0.0,
     0.0,  0.5,  0.0,  1.0,  0.0,
     0.5, -0.5,  0.0,  0.0,  1.0,
];

const VERTEX_SHADER_SOURCE: &[u8] = b"
#version 100
precision mediump float;

attribute vec2 position;
attribute vec3 color;

varying vec3 v_color;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);

    v_color = color;
}
\0";

const FRAGMENT_SHADER_SOURCE: &[u8] = b"
#version 100
precision mediump float;

varying vec3 v_color;

void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
\0";
